'''
If you use TorchIO for your research, please cite the following paper:
Pérez-García et al., TorchIO: a Python library for efficient loading,
preprocessing, augmentation and patch-based sampling of medical images
in deep learning. Credits instructions: https://torchio.readthedocs.io/#credits
'''

import torchio
from torchio import AFFINE, DATA, PATH, TYPE, STEM
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)

import torch
import numpy as np
from unet import UNet
import multiprocessing


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

LIST_FCD =  [ 8,   10,   11,   12,   13,    16,   17,   18,  26,  47, 49,   50, 
  51,   52,   53,   54,   58,  85,  251,  252,  253,  254,  255]



def get_torchio_dataset(inputs, targets, transform):
    """
    Function creates a dataset from data from inputs and targets lists and applies transform to that dataset
    
    Arguments:
        * inputs (list): list of paths to MR images
        * targets (list):  list of paths to segmentation of MR images
        * transform (False/torchio.transforms.augmentation.composition.Compose): transformation which will be applied to MR images
    
    Output:
        * datasets (list): torchio.data.dataset.SubjectsDataset of torchio.data.subject.Subject entities

    """
    subjects = []
    for (image_path, label_path) in zip(inputs, targets ):
        subject_dict = {
            'MRI' : torchio.Image(image_path, torchio.INTENSITY),
            'LABEL': torchio.Image(label_path, torchio.LABEL), #intensity transformations won't be applied to torchio.LABEL 
        }
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
    
    if transform:
        dataset = torchio.SubjectsDataset(subjects, transform = transform)
    elif not transform:
        dataset = torchio.SubjectsDataset(subjects)
    
    return  dataset

def get_loaders(data, cv_split,
        training_transform = False,
        validation_transform = False,
        patch_size = 64,
        patches = False,
        samples_per_volume = 6,
        max_queue_length = 180,
        training_batch_size = 1,
        validation_batch_size = 1,
        mask = False):
    
    """
    Function creates dataloaders 
    
    Arguments:
        * data (data_processor.DataMriSegmentation): dataset
        * cv_split (list): list of two arrays, one with train indexes, other with test indexes
        * training_transform (bool/torchio.transforms.augmentation.composition.Compose): whether or not to 
        use transform for training images
        * validation_transform (bool/torchio.transforms.augmentation.composition.Compose): whether or not to 
        use  transform for validation images
        * patch_size (int): size of patches
        * patches (bool): if True, than patch-based training will be applied
        * samples_per_volume (int): number of patches to extract from each volume
        * max_queue_length (int): maximum number of patches that can be stored in the queue
        * training_batch_size (int): size of batches for training
        * validation_batch_size (int): size of batches for validation
        * mask (bool): if True, than masked images will be used 
    
    Output:
        * training_loader (torch.utils.data.DataLoader): loader for train
        * validation_loader (torch.utils.data.DataLoader): loader for test
    """
    
    training_idx, validation_idx = cv_split
    
    print('Training set:', len(training_idx), 'subjects')
    print('Validation set:', len(validation_idx), 'subjects')
    
    training_set = get_torchio_dataset(
        list(data.img_files[training_idx].values), 
        list(data.img_seg[training_idx].values),
        training_transform)
    
    validation_set = get_torchio_dataset(
        list(data.img_files[validation_idx].values), 
        list(data.img_seg[validation_idx].values),
        validation_transform)
    
    if mask:
        # if using masked data for training
        training_set = get_torchio_dataset(
            list(data.img_files[training_idx].values), 
            list(data.img_mask[training_idx].values),
            training_transform)
        
        validation_set = get_torchio_dataset(
            list(data.img_files[validation_idx].values), 
            list(data.img_mask[validation_idx].values),
            validation_transform)
    
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size = training_batch_size)

    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size = validation_batch_size)
    
    if patches:
        # https://niftynet.readthedocs.io/en/dev/window_sizes.html - about patch based training
        # https://torchio.readthedocs.io/data/patch_training.html - about Queue
        patches_training_set = torchio.Queue(
            subjects_dataset = training_set,
            max_length = max_queue_length,
            samples_per_volume = samples_per_volume,
            patch_size = patch_size,
            sampler_class = torchio.sampler.ImageSampler,
            num_workers = multiprocessing.cpu_count(),
            shuffle_subjects = True,
            shuffle_patches = True,
        )

        patches_validation_set = torchio.Queue(
            subjects_dataset = validation_set,
            max_length = max_queue_length,
            samples_per_volume = samples_per_volume,
            patch_size = patch_size,
            sampler_class = torchio.sampler.ImageSampler,
            num_workers = multiprocessing.cpu_count(),
            shuffle_subjects = False,
            shuffle_patches = False,
        )

        training_loader = torch.utils.data.DataLoader(
            patches_training_set, batch_size = training_batch_size)

        validation_loader = torch.utils.data.DataLoader(
            patches_validation_set, batch_size = validation_batch_size)
        
        print('Patches mode')
        print('Training loader length:', len(training_loader))
        print('Validation loader length:', len(validation_loader))
    
    return training_loader, validation_loader


def get_model_and_optimizer(device, 
                            num_encoding_blocks = 3,
                            out_channels_first_layer = 16,
                            patience = 3):
    
    '''
    Function creates model, optimizer and scheduler
    
    Arguments:
     * device (cpu or gpu): device on which calculation will be done 
     * num_encoding_blocks (int): number of encoding blocks, which consist of con3d + ReLU + conv3d + ReLu
     * out_channels_first_layer (int) : number of channels after first encoding block
     * patience (int): Number of epochs with no improvement after which learning rate will be reduced.
     
    Output:
     * model 
     * optimizer
     * scheduler
    '''
    
    # reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #https://segmentation-models.readthedocs.io/en/latest/tutorial.html
    model = UNet(
        in_channels = 1,
        out_classes = 2,
        dimensions = 3,
        num_encoding_blocks = num_encoding_blocks,
        out_channels_first_layer = out_channels_first_layer,
        normalization = 'batch',
        upsampling_type = 'linear',
        padding = True,
        activation = 'PReLU',
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode = 'min',
                                                           factor = 0.1,
                                                           patience = patience,
                                                           threshold = 0.01)
    
    return model, optimizer, scheduler

def prepare_batch(batch, device):
    """
    Function loads *nii.gz files, sending to the devise.
    For the LABEL in binarises the data.
    
    Arguments:
        * batch (dict): batch dict, contains input data and target data
        * device (torch.device): device for computation 
    
    Output:
        * inputs (torch.tensor): inputs in the appropriate format for model 
        * targets (torch.tensor): targets in the appropriate format for model 
    """
    inputs = batch['MRI'][DATA].to(device)
    targets = batch['LABEL'][DATA]
    targets[0][0][(np.isin(targets[0][0], LIST_FCD))] = 1 # WHAT!??
    targets[targets >= 900] = 1
    targets[targets != 1] = 0
    targets = targets.to(device)   
    
    return inputs, targets