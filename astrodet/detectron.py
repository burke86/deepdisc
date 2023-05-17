import sys, os
import numpy as np
import json
import cv2
from astropy.io import fits
import time
import datetime
import logging



import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm

import detectron2.data as data
import detectron2.data.transforms as T
from detectron2.data.transforms import Transform
from detectron2.data.transforms import Augmentation


from detectron2.data.transforms import Transform
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, TransformList

from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode
#Yufeng Dec21 more import
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import HookBase

from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.engine.hooks import LRScheduler
from fvcore.common.param_scheduler import ParamScheduler
import torch


# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

#Yufeng Dec 21 add astro_metadata into parameters
def plot_stretch_Q(dataset_dicts, astro_metadata, num=0, stretches=[0.01,0.1,0.5,1], Qs=[1,10,5,100], ceil_percentile=99.5):
    """
    Plots different normalizations of your image using the stretch, Q parameters. 
    
    Parameters
    ----------
    dataset_dicts : dict
        detectron dataset dictionary
    num : int
        Dataset number/index to use
    stretches : array
        List of stretch params you want to permutate through to find optimal image normalization.
        Default is [0.01, 0.1, 0.5, 1]
    Qs : array
        List of Q params you want to permutate through to find optimal image normalization.
        Default is [1, 10, 5, 100]
        
    Code adapted from:
        https://pmelchior.github.io/scarlet/tutorials/display.html
        
    Returns
    -------
    fig : Figure object
    
    """
    
    d = dataset_dicts[num]
    
    fig, ax = plt.subplots(len(stretches), len(Qs), figsize=(9,9))
    for i, stretch in enumerate(stretches):
        for j, Q in enumerate(Qs):
            img = read_image(d, normalize="lupton", stretch=stretch, Q=Q, ceil_percentile=ceil_percentile)
            # Scale the RGB channels for the image
            visualizer = Visualizer(img, metadata=astro_metadata)
            out = visualizer.draw_dataset_dict(d)
            ax[i][j].imshow(out.get_image(), origin='lower')
            ax[i][j].set_title("Stretch {}, Q {}".format(stretch, Q), fontsize=10)
            ax[i][j].axis('off')
            
    return fig


    
"""Note:SaveHook is in charge of saving the trained model"""
class SaveHook(HookBase):
    output_name = "model_temp"
    def set_output_name(self, name):
        self.output_name = name
    def after_train(self):
        print('saving', self.output_name)
        self.trainer.checkpointer.save(self.output_name) # Note: Set the name of the output model here
        

#Validation loss code adopted from https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        with torch.no_grad():
            for idx, inputs in enumerate(self._data_loader):            
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                    )
                loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        #print('validation_loss', mean_loss)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        self.trainer.add_val_loss(mean_loss)
        self.trainer.valloss=mean_loss
        comm.synchronize()
        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):# or (next_iter == 1):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class CustomLRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.

        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler


    def before_train(self):
        #print(self.scheduler)
        self._optimizer = self._optimizer or self.trainer.optimizer
        print(self.scheduler)
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id = LRScheduler.get_best_param_group_id(self._optimizer)


    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i


    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()


    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, _LRScheduler):
            return self.scheduler.state_dict()
        return {}


    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, _LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)
            

#Code taken from Deshwal on Stack Overflow
class GenericWrapperTransform(Transform):
    """
    Generic wrapper for any transform (for color transform only. You can give functionality to apply_coods, apply_segmentation too)
    """

    def __init__(self, custom_function):
        """
        Args:
            custom_function (Callable): operation to be applied to the image which takes in an ndarray and returns an ndarray.
        """
        if not callable(custom_function):
            raise ValueError("'custom_function' should be callable")
        
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        '''
        apply transformation to image array based on the `custom_function`
        '''
        return self.custom_function(img)

    def apply_coords(self, coords):
        '''
        Apply transformations to Bounding Box Coordinates. Currently is won't do anything but we can change this based on our use case
        '''
        return coords

    def inverse(self):
        return T.NoOpTransform()

    def apply_segmentation(self, segmentation):
        '''
        Apply transformations to segmentation. currently is won't do anything but we can change this based on our use case
        '''
        return segmentation


class CustomAug(Augmentation):
    """
    Given a probability and a custom function, return a GenericWrapperTransform object whose `apply_image`  
    will be called to perform augmentation
    """

    def __init__(self, custom_function, prob=1.0):
        """
        Args:
            custom_op: Operation to use. Must be a function takes an ndarray and returns an ndarray
            prob (float): probability of applying the function
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        '''
        Based on probability, choose whether you want to apply the given function or not
        '''
        do = self._rand_range() < self.prob
        if do:
            return GenericWrapperTransform(self.custom_function)
        else:
            return T.NoOpTransform() # it returns a Transform which just returns the original Image array only


class KRandomAugmentationList(Augmentation):
    """
    Select and Apply "K" augmentations in "RANDOM" order with "Every"  __call__ method invoke
    """
    def __init__(self, augs, k:int = -1, cropaug=None):
        """
        Args:
            augs: list of [Augmentation or Transform]
            k: Number of augment to use from the given list in range [1,len_augs].
            If None, use all. If it is -1, generate K randomly between [1,len_augs]
        """
        super().__init__()
        self.max_range = len(augs)
        self.k = k
        self.augs = augs # set augs to use as fixed if we have to use same augs everytime
        self.cropaug = cropaug

    def _setup_augs(self, augs, k:int):
        '''
        Setup the argument list. Generates the list of argument to use from the given list
        args:
            augs: list of [Augmentation or Transform])
            k: Number of augment to use from the given list in range [1,len_augs].
            If False, use all. If it is -1, generate K randomly between [1,len_augs]
        '''
        if k == -1: # Generate a random number
            k = np.random.randint(1,len(augs)+1)

        elif k is None: # use all
            k = self.max_range

        temp = np.random.choice(augs,k,replace=False) # get k augments randomly
        augl=[_transform_to_aug(x) for x in temp]
        if self.cropaug is None:
            pass
        else:
            augl.insert(0,self.cropaug)

        return augl


    def __call__(self, aug_input) -> Transform:
        tfms = []

        for x in self._setup_augs(self.augs, self.k): # generate auguments to use randomly on the fly
            #print(x)
            tfm = x(aug_input)
            tfms.append(tfm)
        return TransformList(tfms)

    def __repr__(self):
        msgs = [str(x) for x in self.augs]
        return "AugmentationList[{}]".format(", ".join(msgs))

    __str__ = __repr__

## Added for importing
class _TransformToAug(Augmentation):
    def __init__(self, tfm: Transform):
        self.tfm = tfm

    def get_transform(self, *args):
        return self.tfm

    def __repr__(self):
        return repr(self.tfm)

    __str__ = __repr__


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:
        return _TransformToAug(tfm_or_aug)

