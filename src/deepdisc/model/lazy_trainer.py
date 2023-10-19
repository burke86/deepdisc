from deepdisc.model import models
from deepdisc.model import loaders
from astrodet.astrodet import LazyAstroTrainer
from astrodet import detectron as detectron_addons



def lazy_trainer(model,loader,optimizer,cfg,cfg_loader,output_name):
    
    saveHook = detectron_addons.SaveHook()
    saveHook.set_output_name(output_name)
    schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
    lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
    hookList = [lossHook,schedulerHook,saveHook]

    trainer = LazyAstroTrainer(model, loader, optimizer, cfg, cfg_loader)
    trainer.register_hooks(hookList)

    return trainer

