import torch

def get_dice_score(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon = 1e-9):
    '''
    WORKS ONLY IF BATCH_SIZE = 1, I WILL IMPROVE IT LSTER
    
    Function get dice score on output and target tensors  
    https://www.jeremyjordan.me/semantic-segmentation/#loss
    
    Arguments:
        * output (torch.tensor):  (1,2,X,Y,Z) probabilities tensor, one component 
        is probability-tensor (1,X,Y,Z) to be the brain, another component 
        is probability-tensor (1,X,Y,Z) to be background.
        * target (torch.tensor): (1,2,X,Y,Z)  binary tensor, one component 
        is binary-mask (1,X,Y,Z) for the brain, another component 
        is binary-mask (1,X,Y,Z) for the background.
        * SPATIAL_DIMENSIONS (typle): typle with indexes corresponding to spatial parts of tensors
        * epsilon (float): a small number used for numerical stability to avoid divide by zero errors  
    
    Outputs:
        * dice score (torch.tensor): dice score 
    '''
    
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    
    num = 2 * tp
    denom = 2 * tp + fp + fn
    
    dice_score = (num + epsilon)/(denom + epsilon)
    
    return dice_score

def get_dice_loss(output, target):
    '''
    WORKS ONLY IF BATCH_SIZE = 1, I WILL IMPROVE IT LSTER
    
    Function get dice score loss on output and target tensors  
    
    Arguments:
        * output (torch.tensor):  (1,2,X,Y,Z) probabilities tensor, one component 
        is probability-tensor (1,X,Y,Z) to be the brain, another component 
        is probability-tensor (1,X,Y,Z) to be background.
        * target (torch.tensor): (1,2,X,Y,Z) binary tensor, one component 
        is binary-mask (1,X,Y,Z) for the brain, another component 
        is binary-mask (1,X,Y,Z) for the background.
    
    Outputs:
        * dice score loss (torch.tensor): 1 - dice score 
    '''
    return 1 - get_dice_score(output, target)
