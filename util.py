import torch
import numpy as np


def my_binary_loss(output, target):
    return (output and target).mean

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def get_metrics_count(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_tag, dim=1)
    y_pred_tags = y_pred_tags.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    true_positives = np.logical_and(y_pred_tags, y_test).sum()
    false_positives = np.logical_and(y_pred_tags, np.logical_not(y_test)).sum()
    true_negatives = np.logical_and(np.logical_not(y_pred_tags), np.logical_not(y_test)).sum()
    false_negatives = np.logical_and(np.logical_not(y_pred_tags), y_test).sum()
    return true_positives, false_positives, true_negatives, false_negatives




def get_world_size():
    return 1

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

loss_dict_template = {
    'loss_sum' : [],
    'loss_classifier' : [],
    'loss_box_reg' : [],
    'loss_objectness' : [],
    'loss_rpn_box_reg' : []
}