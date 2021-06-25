import torch
import numpy as np


def eval_mae(y_pred, y):
    return torch.abs(y_pred - y).mean()

def eval_pr(y_pred, y):
    y_temp = (y_pred >= 0.5).float()

    y = y.cpu().clone().detach().contiguous().numpy()
    pred = y_pred.cpu().clone().detach().contiguous().numpy()
    temp = y_temp.cpu().clone().detach().contiguous().numpy()
    tp = (temp * y).sum()

    prec, recall = tp / (y_temp.sum()), tp / y.sum()
    return prec, recall

def get_average(list):
    sum = 0
    list = np.array(list)
    sum = np.nansum(list)
    average = sum/(len(list)-np.count_nonzero(list != list))
    return average

def eval_Fscore(y_pred, y):
    prec, recall = eval_pr(y_pred, y)
    beta = 0.7
    return (1 + beta ** 2) * prec.numpy() * recall / (beta ** 2 * prec.numpy() + recall)

def eval_iou(y_pred, y):
    y_temp = (y_pred >0.7).float()

    y = y.cpu().clone().detach().contiguous().numpy()
    pred = y_pred.cpu().clone().detach().contiguous().numpy()
    temp = y_temp.cpu().clone().detach().contiguous().numpy()
    intersection = (temp * y).sum()
    u = temp + y
    u[u < 0] = 0
    u[u > 0] = 1
    union = u.sum()
    iou = intersection/union
    return iou