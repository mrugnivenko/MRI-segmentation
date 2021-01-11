import matplotlib.pyplot as plt
import torch
import numpy as np
import nibabel

def plot_central_cuts(img):
    
    """
    Function plots central slices of MRI
    
    Arguments:
        * img (torch.Tensor): MR image (1xDxHxW)
    
    Output:
        * picture of central slices of MRI
    """
    
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if (len(img.shape) > 3):
            img = img[0,:,:,:]
                
    elif isinstance(img, nibabel.nifti1.Nifti1Image):    
        img = img.get_fdata()
   
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    
    axes[0].imshow(img[ img.shape[0] // 2, :, :])
    axes[1].imshow(img[ :, img.shape[1] // 2, :])
    axes[2].imshow(img[ :, :, img.shape[2] // 2])
    
    plt.show()

def plot_certain_cuts(img, coordinates):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if (len(img.shape) > 3):
            img = img[0,:,:,:]
                
    elif isinstance(img, nibabel.nifti1.Nifti1Image):    
        img = img.get_fdata()
   
    coordinate_sagital, coordinate_coronal, coordinate_axial = coordinates
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    axes[0].imshow(img[ coordinate_sagital, :, :])
    axes[1].imshow(img[ :, coordinate_coronal, :])
    axes[2].imshow(img[ :, :, coordinate_axial])
    
    plt.show()