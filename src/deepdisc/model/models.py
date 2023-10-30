from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm, nonzero_tuple
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import CascadeROIHeads
from torch import nn
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.nn import functional as F


def return_lazy_model(cfg):
    """Return a model formed from a LazyConfig.

    Parameters
    ----------
    cfg : .py file
        a LazyConfig

    Returns
    -------
        torch model
    """

    model = instantiate(cfg.model)

    model.to(cfg.train.device)
    model = create_ddp_model(model, **cfg.train.ddp)

    return model


class RedshiftPDFCasROIHeads(CascadeROIHeads):
    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components

        self.redshift_fc = nn.Sequential(
            nn.Linear(int(np.prod(self._output_size)), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
        )

    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    F.softplus(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances):
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)

        features = nn.Flatten()(features)

        fcs = self.redshift_fc(features)

        pdfs = self.output_pdf(fcs)

        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            gt_classes = cat([x.gt_classes for x in instances])
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            pdfs_fg = self.output_pdf(fcs[fg_inds, ...])

            gt_redshifts = cat([x.gt_redshift for x in instances])

            nlls_fg = -pdfs_fg.log_prob(gt_redshifts[fg_inds])

            nlls = -pdfs.log_prob(gt_redshifts)[fg_inds]
            return {"redshift_loss": torch.mean(nlls)}

        else:
            zs = torch.tensor(np.linspace(-1, 5, 200)).to(fcs.device)

            probs = torch.zeros((num_instances_per_img[0], 200)).to(fcs.device)
            for i, z in enumerate(zs):
                # probs.append(outputs.log_prob(z))
                probs[:, i] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = probs

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}
