# Training script for decam data
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import argparse
import logging
import sys
import gc
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultTrainer,SimpleTrainer, HookBase, default_argument_parser, default_setup, hooks, launch
from typing import Dict, List, Optional
import detectron2.solver as solver
import detectron2.modeling as modeler
import detectron2.data as data
import detectron2.data.transforms as T
import detectron2.checkpoint as checkpointer
from detectron2.data import detection_utils as utils
import detectron2.utils.comm as comm

import weakref
import copy
import torch
import time

import imgaug.augmenters as iaa

from astrodet import astrodet as toolkit

#Custom Aug classes have been added to detectron source files
from astrodet.astrodet import CustomAug
from detectron2.data.transforms.augmentation import KRandomAugmentationList

import imgaug.augmenters.flip as flip
import imgaug.augmenters.blur as blur


# Prettify the plotting
from astrodet.astrodet import set_mpl_style
set_mpl_style()



from detectron2.structures import BoxMode
from astropy.io import fits
import glob

# ### Register Astro R-CNN dataset

def get_astro_dicts(filename_dict):
    
    """
    This can needs to be customized to your trianing data format
    
    """
        
    dataset_dicts = []
    filters = list(filename_dict.keys())
    #yufeng april5: why only 1st filter
    f = filename_dict['filters'][0] # Pick the 1st filter for now
    
    # Filename loop
    for idx, (filename_img, filename_mask) in enumerate(zip(filename_dict[f]['img'], filename_dict[f]['mask'])):
        record = {}

        # Open FITS image of first filter (each should have same shape)
        with fits.open(filename_img, memmap=False, lazy_load_hdus=False) as hdul:
            height, width = hdul[0].data.shape
            
        # Open each FITS mask image
        with fits.open(filename_mask, memmap=False, lazy_load_hdus=False) as hdul:
            hdul = hdul[1:]
            sources = len(hdul)
            # Normalize data
            data = [hdu.data for hdu in hdul]
            category_ids = [hdu.header["CAT_ID"] for hdu in hdul]
            ellipse_pars = [hdu.header["ELL_PARM"] for hdu in hdul]
            bbox = [list(map(int, hdu.header["BBOX"].split(','))) for hdu in hdul]
            area = [hdu.header["AREA"] for hdu in hdul]

        # Add image metadata to record (should be the same for each filter)
        for f in filename_dict['filters']:
            record[f"filename_{f.upper()}"] = filename_dict[f]['img'][idx]
        # Assign file_name
        record[f"file_name"] = filename_dict[filename_dict['filters'][0]]['img'][idx]
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []

        # Generate segmentation masks from model
        for i in range(sources):
            image = data[i]
            # Why do we need this?
            if len(image.shape) != 2:
                continue
            height_mask, width_mask = image.shape
            # Create mask from threshold
            mask = data[i]
            # Smooth mask
            #mask = cv2.GaussianBlur(mask, (9,9), 2)
            x,y,w,h = bbox[i] # (x0, y0, w, h)

            # https://github.com/facebookresearch/Detectron/issues/100
            contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                # contour = [x1, y1, ..., xn, yn]
                contour = contour.flatten()
                if len(contour) > 4:
                    contour[::2] += (x-w//2)
                    contour[1::2] += (y-h//2)
                    segmentation.append(contour.tolist())
            # No valid countors
            if len(segmentation) == 0:
                continue

            # Add to dict
            obj = {
                "bbox": [x-w//2, y-h//2, w, h],
                "area": w*h,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": segmentation,
                "category_id": category_ids[i],
                "ellipse_pars": ellipse_pars[i]
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
            
    return dataset_dicts

def get_dict_lists(dataset_names,dirpath,sampleNumbers=-1):
    filenames_dict_list = [] # List holding filenames_dict for each dataset
    for i, d in enumerate(dataset_names):
        data_path = os.path.join(dirpath, d)
        # Get dataset dict info
        filenames_dict = {}
        filenames_dict['filters'] = ['g', 'r', 'i']

        # Get each unqiue tract-patch in the data directory
        #file = full path name
        files = glob.glob(os.path.join(data_path, '*_scarlet_segmask.fits'))
        if sampleNumbers != -1:
            files = files[:sampleNumbers]
        # s = sample name
        s = [os.path.basename(f).split('_scarlet_segmask.fits')[0] for f in files]
        for f in filenames_dict['filters']:
            filenames_dict[f] = {}
            # List of image files in the dataset
            #Yufeng dec/21  [Errno 2] No such file or directory: '/home/shared/hsc/test/G-I-8525-4,5-c5_scarlet_img'
            #filenames_dict[f]['img'] = [os.path.join(data_path, f'{f.upper()}-{tract_patch}_scarlet_img.fits') for tract_patch in s]
            #Yufeng jan 18 f.upper() indicates filter, tract_patch[1:] removes the default I band in the front
            filenames_dict[f]['img'] = [os.path.join(data_path, f.upper() + f'{tract_patch[1:]}_scarlet_img.fits') for tract_patch in s]
            # List of mask files in the dataset
            #Yufeng jan 18 all mask files are in the I band
            filenames_dict[f]['mask'] = [os.path.join(data_path, f'{tract_patch}_scarlet_segmask.fits') for tract_patch in s]
            
        filenames_dict_list.append(filenames_dict)
    return filenames_dict_list

def data_register_and_load(dataset_names,filenames_dict_list):
    
    # Dataset loading can take a while
    print('Data loading may take a few minutes')
    dataset_dicts = {}
    for i, d in enumerate(dataset_names):
        print(f'Loading {d}')
        dataset_dicts[d] = get_astro_dicts(filenames_dict_list[i])

    return dataset_dicts

def get_data_from_json(file):
    # Opening JSON file
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def read_image(filenames, normalize='lupton', stretch=5, Q=10, m=0, ceil_percentile=99.995, dtype=np.uint8, A=1e4):
    # Read image
    #g = fits.getdata(os.path.join(dataset_dict['filename_G']), memmap=False)
    #r = fits.getdata(os.path.join(dataset_dict['filename_R']), memmap=False)
    #z = fits.getdata(os.path.join(dataset_dict['filename_I']), memmap=False)


    g = fits.getdata(os.path.join(filenames[0]), memmap=False)
    r = fits.getdata(os.path.join(filenames[1]), memmap=False)
    z = fits.getdata(os.path.join(filenames[2]), memmap=False)
    
    # Contrast scaling / normalization
    I = (z + r + g)/3.0
    
    length, width = g.shape
    image = np.empty([length, width, 3], dtype=dtype)
    
    # Options for contrast scaling
    if normalize.lower() == 'lupton':
        z = z*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        r = r*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        g = g*np.arcsinh(stretch*Q*(I - m))/(Q*I)
    
    elif normalize.lower() == 'zscore':
        #Isigma = I*np.mean([np.nanstd(g), np.nanstd(r), np.nanstd(z)])
        #z = (z - np.nanmean(z) - m)/Isigma
        #r = (r - np.nanmean(r) - m)/Isigma
        #g = (g - np.nanmean(g) - m)/Isigma
        
        zsigma = np.nanstd(z)
        rsigma = np.nanstd(r)
        gsigma = np.nanstd(g)
        
        z = A*(z - np.nanmean(z) - m)/zsigma
        r = A*(r - np.nanmean(r) - m)/rsigma
        g = A*(g - np.nanmean(g) - m)/gsigma
        
    elif normalize.lower() == 'linear':
        z = (z - m)/I
        r = (r - m)/I
        g = (g - m)/I
    else:
        print('Normalize keyword not recognized.')

    #max_RGB = np.nanpercentile([z, r, g], ceil_percentile) * 2
    # avoid saturation
    #r = r/max_RGB; g = g/max_RGB; z = z/max_RGB

    # Rescale to 0-255 for dtype=np.uint8
    #max_dtype = np.iinfo(dtype).max
    #r = r*max_dtype
    #g = g*max_dtype
    #z = z*max_dtype

    # 0-255 RGB image
    image[:,:,0] = z # R
    image[:,:,1] = r # G
    image[:,:,2] = g # B
    
    return image

# ### Augment Data

def gaussblur(image):
    aug = iaa.GaussianBlur(sigma=(0.0, np.random.random_sample()*4+2))
    return aug.augment_image(image)

def addelementwise16(image):
    aug = iaa.AddElementwise((-3276, 3276))
    return aug.augment_image(image)

def addelementwise8(image):
    aug = iaa.AddElementwise((-25, 25))
    return aug.augment_image(image)

def addelementwise(image):
    aug = iaa.AddElementwise((-image.max()*.1, image.max()*.1))
    return aug.augment_image(image)

def train_mapper(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    filenames=[dataset_dict['filename_G'],dataset_dict['filename_R'],dataset_dict['filename_I']]

    image = read_image(filenames, normalize='zscore', dtype=np.int16)
    #image = rescale_image(dataset_dict["file_name"], **read_image_args)
    
    #augs = T.AugmentationList([
    #    T.RandomCrop('relative',(0.5,0.5)),
    #    T.RandomRotation([-90, 90, 180], sample_style='choice'),
    #    T.RandomFlip(prob=0.5),
    #    T.RandomFlip(prob=0.5,horizontal=False,vertical=True),
    #    #T.Resize((512,512)),
    #    #T.FixedSizeCrop((512,512))
    #])
    
    
    augs = KRandomAugmentationList([
        # my custom augs
        T.RandomRotation([-90, 90, 180], sample_style='choice'),
        T.RandomFlip(prob=0.5),
        T.RandomFlip(prob=0.5,horizontal=False,vertical=True),
        CustomAug(gaussblur,prob=1.0),
        CustomAug(addelementwise,prob=1.0)
        #CustomAug(white),
        ],
        k=-1,
        cropaug=T.RandomCrop('relative',(0.5,0.5))
    )
    # Data Augmentation
    auginput = T.AugInput(image)
    # Transformations to model shapes
    transform = augs(auginput)
    image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]

    instances = utils.annotations_to_instances(annos, image.shape[1:])
    instances = utils.filter_empty_instances(instances)

    return {
       # create the format that the model expects
        "image": image,
        "image_shaped": auginput.image,
        "height": image.shape[0],
        "width": image.shape[1],
        "image_id": dataset_dict["image_id"],
        "instances": instances,
    }

def test_mapper(dataset_dict, **read_image_args):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    filenames=[dataset_dict['filename_G'],dataset_dict['filename_R'],dataset_dict['filename_I']]
    image = read_image(filenames, normalize="zscore", dtype=np.int16)
    
    augs = T.AugmentationList([
        T.RandomCrop('relative',(0.5,0.5))
    ])
    # Data Augmentation
    auginput = T.AugInput(image)
    # Transformations to model shapes
    transform = augs(auginput)
    image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]


    instances = utils.annotations_to_instances(annos, image.shape[1:])
    instances = utils.filter_empty_instances(instances)
    
    return {
       # create the format that the model expects
        "image": image,
        "image_shaped": auginput.image,
        "height": image.shape[0],
        "width": image.shape[1],
        "image_id": dataset_dict["image_id"],
        "instances": instances,
        #"annotations": annos
    }


# Hack if you get SSL certificate error 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main(tl,dirpath,dataset_names,train_head,output_name,cfgfile,args):




    # ### Prepare For Training
    # Training logic:
    # To replicate 2019 methodology, need to 
    # 1) run intially with backbone frozen (freeze_at=4) for 15 epochs
    # 2) unfreeze and run for [25,35,50] epochs with lr decaying by 0.1x each time


    
    trainfile=dirpath+dataset_names[0]+'.json'
    testfile=dirpath+dataset_names[1]+'.json'
    valfile=dirpath+dataset_names[2]+'.json'

    DatasetCatalog.register("astro_train", lambda: get_data_from_json(trainfile))
    MetadataCatalog.get("astro_train").set(thing_classes=["star", "galaxy","other"])
    astrotrain_metadata = MetadataCatalog.get("astro_train") # astro_test dataset needs to exist

    #DatasetCatalog.register("astro_test", lambda: get_data_from_json(testfile))
    #MetadataCatalog.get("astro_test").set(thing_classes=["star", "galaxy","other"])
    #astrotest_metadata = MetadataCatalog.get("astro_test") # astro_test dataset needs to exist


    DatasetCatalog.register("astro_val", lambda: get_data_from_json(valfile))
    MetadataCatalog.get("astro_val").set(thing_classes=["star", "galaxy","other"])
    astroval_metadata = MetadataCatalog.get("astro_val") # astro_test dataset needs to exist


    


    #treg=time.time()
    #DatasetCatalog.register("astro_train", lambda: get_astro_dicts(filenames_dict_list[0]))
    #MetadataCatalog.get("astro_train").set(thing_classes=["star", "galaxy","other"])
    #DatasetCatalog.register("astro_test", lambda: get_astro_dicts(filenames_dict_list[1]))
    #MetadataCatalog.get("astro_test").set(thing_classes=["star", "galaxy","other"])
    #DatasetCatalog.register("astro_val", lambda: get_astro_dicts(filenames_dict_list[2]))
    #MetadataCatalog.get("astro_val").set(thing_classes=["star", "galaxy","other"])

    #if comm.is_main_process():
    #    print('Data register time ', time.time()-treg)
    
    #astrotrain_metadata = MetadataCatalog.get("astro_train") # astro_test dataset needs to exist
    #astrotest_metadata = MetadataCatalog.get("astro_test") # astro_test dataset needs to exist
    #astrotval_metadata = MetadataCatalog.get("astro_val") # astro_test dataset needs to exist
    
    output_dir = '/home/shared/hsc/HSC/HSC_DR3/models/'
    #astro_metadata = MetadataCatalog.get("astro_train")
    #print(len(filenames_dict_list[0]['g']['img'][0]))
    
    #tl=len(filenames_dict_list[0]['g']['img'])
    #tl=200

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfgfile)) # Get model structure
    cfg.DATASETS.TRAIN = ("astro_train") # Register Metadata
    cfg.DATASETS.TEST = ("astro_val") # Config calls this TEST, but it should be the val dataset
    #cfg.TEST.EVAL_PERIOD = 40
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 250  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.PIXEL_MEAN = [-200,-200,-200]


    #cfg.INPUT.MIN_SIZE_TRAIN = 1025
    #cfg.INPUT.MAX_SIZE_TRAIN = 1050
    cfg.INPUT.MIN_SIZE_TRAIN = 500
    cfg.INPUT.MAX_SIZE_TRAIN = 500
    
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.SOLVER.IMS_PER_BATCH = 4   # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations*num_gpus)
    
    cfg.OUTPUT_DIR = output_dir
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    # Maximum absolute value used for clipping gradients
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0

    # iterations for 15,25,35,50 epochs
    epoch = int(tl/cfg.SOLVER.IMS_PER_BATCH)
    e1=epoch*15
    e2=epoch*10
    e3=epoch*20
    efinal=epoch*35

    
    #val_per = epoch
    val_per = 20

    if train_head:

        # Step 1)
        cfg.MODEL.BACKBONE.FREEZE_AT = 4   # Initial re-training of the head layers (i.e. freeze the backbone)
        cfg.SOLVER.BASE_LR = 0.001    
        cfg.SOLVER.STEPS = []          # do not decay learning rate for retraining
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.MAX_ITER = e1     # for DefaultTrainer

        init_coco_weights = True # Start training from MS COCO weights
        
        if init_coco_weights:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfgfile)  # Initialize from MS COCO
        else:
            cfg.MODEL.WEIGHTS = os.path.join(output_dir, 'model_temp.pth')  # Initialize from a local weights
        

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        model = modeler.build_model(cfg)
        optimizer = solver.build_optimizer(cfg, model)
        loader = data.build_detection_train_loader(cfg, mapper=train_mapper)
        test_loader = data.build_detection_test_loader(cfg,cfg.DATASETS.TEST,mapper=test_mapper)
        
        saveHook = toolkit.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = toolkit.CustomLRScheduler(optimizer=optimizer)
        lossHook = toolkit.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook,schedulerHook,saveHook]
        #hookList = [schedulerHook,saveHook]

        trainer = toolkit.NewAstroTrainer(model, loader, optimizer, cfg)
        trainer.register_hooks(hookList)
        trainer.set_period(int(epoch/2)) # print loss every n iterations
        trainer.train(0,e1)
        #trainer.set_period(10)
        #trainer.train(0,100)
        if comm.is_main_process():
            np.save(output_dir+output_name+'_losses',trainer.lossList)
            np.save(output_dir+output_name+'_val_losses',trainer.vallossList)
        return

    else:
        
        # Step 2)

        cfg.MODEL.BACKBONE.FREEZE_AT = 0  # unfreeze all backbone layers  
        cfg.SOLVER.BASE_LR = 0.0001   
        cfg.SOLVER.STEPS = [e2,e3]        # decay learning rate
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.MAX_ITER = efinal          # for LR scheduling
        cfg.MODEL.WEIGHTS = os.path.join(output_dir, output_name+'.pth')  # Initialize from a local weights

        model = modeler.build_model(cfg)
        optimizer = solver.build_optimizer(cfg, model)
        loader = data.build_detection_train_loader(cfg, mapper=train_mapper)
        test_loader = data.build_detection_test_loader(cfg,cfg.DATASETS.TEST,mapper=test_mapper)

        saveHook = toolkit.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = toolkit.CustomLRScheduler(optimizer=optimizer)
        lossHook = toolkit.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook,schedulerHook,saveHook]

        trainer = toolkit.NewAstroTrainer(model, loader, optimizer, cfg)
        trainer.register_hooks(hookList)
        trainer.set_period(int(epoch/2)) # print loss every n iterations
        trainer.train(0,efinal)
        #trainer.set_period(10) # print loss every n iterations
        #trainer.train(0,100)

        if comm.is_main_process():
            losses = np.load(output_dir+output_name+'_losses.npy')
            losses = np.concatenate((losses,trainer.lossList))
            np.save(output_dir+output_name+'_losses',losses)

            vallosses = np.load(output_dir+output_name+'_val_losses.npy')
            vallosses= np.concatenate((vallosses,trainer.vallossList))
            np.save(output_dir+output_name+'_val_losses',vallosses)

        return


def custom_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--run-name", type=str, default='baseline', help="output name for run")
    parser.add_argument("--backbone", type=str, default='res50', help="backbone architecture for model")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--sample-number", type=int, default=-1, help="number of files to use for each set")


    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = custom_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    dirpath = '/home/shared/hsc/HSC/HSC_DR3/data/' # Path to dataset
    #dirpath = '/home/shared/hsc/decam/decam_data/' # Path to dataset

    dataset_names = ['train', 'test', 'val'] 

    
    #filenames_dict_list = get_dict_lists(dataset_names,dirpath,args.sample_number)
    traind = get_data_from_json(os.path.join(dirpath,dataset_names[0])+'_full.json')
    testd = get_data_from_json(os.path.join(dirpath,dataset_names[2])+'_full.json')


    #number of total samples
    print('# of train sample: ', len(traind))
    print('# of val sample: ', len(testd))

    tl = len(traind)
    del traind,testd
    gc.collect()

    #dataset_dicts = data_register_and_load(dataset_names,filenames_dict_list)


    #for i, d in enumerate(dataset_names):
    #    filenames_dir = os.path.join(dirpath,d)
    #    DatasetCatalog.register("astro_" + d, lambda: get_astro_dicts(filenames_dir))
    #    MetadataCatalog.get("astro_" + d).set(thing_classes=["star", "galaxy"], things_colors = ['blue', 'gray'])


    output_name=args.run_name
    arch = args.backbone
    if arch=="res50":
        cfgfile="COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
    elif arch=="res101":
        cfgfile="COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
    elif arch=="res101_fpn":
        cfgfile="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    else:
        raise NameError('Choose a different architecture')
                        
    print('Training head layers')
    train_head=True
    t0=time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(tl,dirpath,dataset_names,train_head,output_name,cfgfile,args,),
    )

    print('Training full model')
    train_head=False
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(tl,dirpath,dataset_names,train_head,output_name,cfgfile,args,),
    )
    print(f'Took {time.time()-t0} seconds')

    
