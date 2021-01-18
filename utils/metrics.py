import torch

def get_dice_score(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), smoothing = 1e-9):
    '''
    WORKS ONLY IF BATCH_SIZE = 1, I WILL IMPROVE IT LSTER
    
    Function get dice score on output and target tensors  
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    
    Arguments:
        * output (torch.tensor):  (1,2,X,Y,Z) probabilities tensor, one component 
        is probability-tensor (1,X,Y,Z) to be the brain, another component 
        is probability-tensor (1,X,Y,Z) to be background.
        * target (torch.tensor): (1,1,X,Y,Z) binary tensor. 1 - brain mask, 0 - background
        * SPATIAL_DIMENSIONS (typle): typle with indexes corresponding to spatial parts of tensors
        * smoothing (float): a small number to avoid division on zero  
    
    Outputs:
        * dice score (torch.tensor): dice score 
    '''
#     output[output >= 0.5] = 1
#     output[output != 1] = 0
    #output = torch.argmin(output, dim=1).float() # because 0 class = 1, 1 class = 0
    output = output[:,0,:].reshape(target.shape)
    
    tp_brain = (output * target).sum(dim=SPATIAL_DIMENSIONS).float()
    tp_bg = ((1-output) * (1-target)).sum(dim=SPATIAL_DIMENSIONS).float()
    
    volume_sum_brain = output.sum(dim=SPATIAL_DIMENSIONS).float() + target.sum(dim=SPATIAL_DIMENSIONS).float()
    volume_sum_bg = (1-output).sum(dim=SPATIAL_DIMENSIONS).float() + (1-target).sum(dim=SPATIAL_DIMENSIONS).float()
    
    dice_brain = 2*tp_brain/(2*volume_sum_brain + smoothing)
    dice_bg = 2*tp_bg/(2*volume_sum_bg + smoothing)
    
    return (dice_brain+dice_bg)/2
    
#     p0 = output
#     g0 = target
#     p1 = 1 - p0
#     g1 = 1 - g0
#     tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
#     fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
#     fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
#     num = 2 * tp
#     denom = 2 * tp + fp + fn + epsilon
#     dice_score = num / denom
#     return dice_score

def get_dice_loss(output, target):
    '''
    WORKS ONLY IF BATCH_SIZE = 1, I WILL IMPROVE IT LSTER
    
    Function get dice score loss on output and target tensors  
    
    Arguments:
        * output (torch.tensor):  (1,2,X,Y,Z) probabilities tensor, one component 
        is probability-tensor (1,X,Y,Z) to be the brain, another component 
        is probability-tensor (1,X,Y,Z) to be background.
        * target (torch.tensor): (1,1,X,Y,Z) binary tensor. 1 - brain mask, 0 - background
    
    Outputs:
        * dice score loss (torch.tensor): 1 - dice score 
    '''
    return 1 - get_dice_score(output, target)
