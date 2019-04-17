import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def cross_entropy2d(input, target, weights=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim = 1)
    loss = F.nll_loss(log_p, Variable(target.cuda()), weight=weights, size_average=size_average)
    
    return loss


def soft_iou(input, target, class_average=True, ignore=None):
    n, c, h, w = input.size()
    input = F.softmax(input, dim = 1)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1).unsqueeze(1)

    target_one_hot = None

    if ignore is None:
        target_one_hot = torch.FloatTensor(n*h*w, c).zero_().cuda()
        target_one_hot.scatter_(1, target, 1)
    
    if ignore is not None:
        target_one_hot = torch.FloatTensor(n*h*w, c+1).zero_().cuda()
        target_one_hot.scatter_(1, target, 1)
        target_one_hot = target_one_hot[:,:ignore]

    target_one_hot = Variable(target_one_hot.cuda())

    tp = torch.mul(input, target_one_hot)

    numerator = torch.sum(tp, dim = 0)
    denom = torch.sum(input + target_one_hot - tp, dim = 0)
    iou = torch.sum(torch.mul(numerator, 1/denom), dim = 0)

    return iou/c