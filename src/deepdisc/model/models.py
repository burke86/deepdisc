
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model

def return_lazy_model(cfg):
    model = instantiate(cfg.model)

    '''
    for param in model.parameters():
        param.requires_grad = False
    # Phase 1: Unfreeze only the roi_heads
    for param in model.roi_heads.parameters():
        param.requires_grad = True
    # Phase 2: Unfreeze region proposal generator with reduced lr
    for param in model.proposal_generator.parameters():
        param.requires_grad = True
    '''

    model.to(cfg.train.device)
    model = create_ddp_model(model, **cfg.train.ddp)

    return model
