import torch
import torch.nn as nn


INF = 1000000

class BCEWithLogitsLoss2(nn.Module):
    def __init__(self, weight=None, reduction='elementwise_mean'):
        super(BCEWithLogitsLoss2, self).__init__()
        self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return bce_with_logits(input, target, weight=self.weight, reduction=self.reduction)


def bce_with_logits(input, target, weight=None, reduction='elementwise_mean'):
    """
    This function differs from F.binary_cross_entropy_with_logits in the way 
    that if weight is not None, the loss is normalized by weight
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))
    if weight is not None:
        if not (weight.size() == input.size()):
            raise ValueError("Weight size ({}) must be the same as input size ({})".format(
                weight.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
        ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        if weight is not None:
            # different from F.binary_cross_entropy_with_logits
            return loss.sum() / weight.sum()
        else:
            return loss.mean()
    else:
        return loss.sum()



class ChamfersDistance(nn.Module):
    '''
    Extensively search to compute the Chamfersdistance. 
    '''

    def forward(self, input1, input2):

        # input1, input2: Nx3, Nx3

        N, K = input1.shape
        
        # Repeat (x,y,z) M times in a row
        input11 = input1.unsqueeze(1)           # Nx1x3
        input11 = input11.expand(N, N, K)       # NxNx3
        # Repeat (x,y,z) N times in a column
        input22 = input2.unsqueeze(0)           # 1xNx3
        input22 = input22.expand(N, N, K)       # NxNx3
        # compute the distance matrix
        D = input11 - input22                   # NxNx3
        D = torch.norm(D, p=2, dim=2)           # NxN

        dist0, _ = torch.min(D, dim=0)        # M
        dist1, _ = torch.min(D, dim=0)        # N

        
        dist0 = torch.mean(dist0, 0)
        dist1 = torch.mean(dist1, 0)

        loss = dist0 + dist1  
        #loss = torch.mean(loss)                             # 1
        return loss


def registration_loss(source_pc, template_pc):
    """
    Registration consistency
    
    source_pc: <Nx3>
    template_pc: <Nx3>
    """
    criternion = ChamfersDistance()

    loss = criternion(template_pc, source_pc)
    return loss


def chamfer_loss(source, template):

    # Source: <bxNx3>
    batch_size = source.shape[0]
    #total_step = bs - seq + 1
    loss = 0.

    for b in range(batch_size):
        current_loss = registration_loss(source[b, :, :], template[b, :, :])
        loss += current_loss

    loss = loss / batch_size

    return loss


def bce_ch(pred, targets, source, template, bce_weight=None, gamma=0.99, lam=0):
    """
    pred: <Bx(n+1)Lx1>, occupancy probabiliry from M-Net
    targets: <Bx(n+1)Lx1>, occupancy label
    
    bce_weight: <Bx(n+1)Lx1>, weight for each point in computing bce loss
    """
    #lam = 5
    bce_loss = bce(pred, targets, bce_weight)
    ch_loss = chamfer_loss(source, template)

    #return gamma*ch_loss
    return gamma * bce_loss + (1 - gamma) * ch_loss

def bce(pred, targets, weight=None):
    criternion = BCEWithLogitsLoss2(weight=weight)
    loss = criternion(pred, targets)
    return loss