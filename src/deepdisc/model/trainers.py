from deepdisc.model import models
from deepdisc.model import loaders
from astrodet.astrodet import LazyAstroTrainer
from astrodet import detectron as detectron_addons



def return_lazy_trainer(model,loader,optimizer,cfg,cfg_loader,hooklist):
    
    trainer = LazyAstroTrainer(model, loader, optimizer, cfg, cfg_loader)
    trainer.register_hooks(hooklist)

    return trainer
