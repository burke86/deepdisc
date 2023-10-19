from deepdisc.model import models
from deepdisc.model import loaders
from astrodet.astrodet import LazyAstroTrainer
from astrodet import detectron as detectron_addons



def return_lazy_trainer(model,loader,optimizer,cfg,cfg_loader,hooklist):
    
    trainer = LazyAstroTrainer(model, loader, optimizer, cfg, cfg_loader)
    trainer.register_hooks(hooklist)

    return trainer


def return_savehook(output_name):
    saveHook = detectron_addons.SaveHook()
    saveHook.set_output_name(output_name)
    return saveHook

def return_schedulerhook(optimizer):
    schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
    return schedulerHook

def return_evallosshook(val_per,model,test_loader):
    lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
    return lossHook