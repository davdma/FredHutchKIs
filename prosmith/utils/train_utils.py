import os
import torch
import torch.distributed as dist
import logging
import numpy as np

class EarlyStopper:
    """Stops the training early if validation loss doesn't improve after a given number of epochs.
    
    Parameters
    ----------
    patience : int
        Number of epochs without improvement before training is stopped.
    min_delta : float
        Loss above lowest loss + min_delta counts towards patience.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = True
        self.min_validation_loss = float('inf')
        self.metric = None

    def step(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best = True
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.best = False
        else:
            self.best = False

    def is_stopped(self):
        return self.counter >= self.patience

    def is_best_epoch(self):
        return self.best

    def store_metric(self, metric):
        self.metric = metric

    def get_metric(self):
        return self.metric

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def is_cuda(device):
    return device == torch.device('cuda')

def accuracy(y_pred, y_true):
    # Calculate accuracy
    correct = (y_pred == y_true).sum().item()
    total = y_true.shape[0]
    acc = correct / total    
    return acc

def multi_accuracy(y_pred, y_true, labels=None):
    if labels is None:
        raise Exception('Requires list of class labels in ascending order e.g. [0, 1, 2].')

    multi_acc = []
    # calculate accuracy for each label
    for label in labels:
        class_indices = (y_true == label)
        correct = np.sum(y_pred[class_indices] == y_true[class_indices])
        total = np.sum(class_indices)
        class_acc = correct / total
        multi_acc.append(class_acc)

    return multi_acc

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0
    
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig, ys_line):
    ys_orig = np.array(ys_orig).reshape(-1)
    ys_line = np.array(ys_line).reshape(-1)
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))
