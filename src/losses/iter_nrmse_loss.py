import numpy as np
import torch
import torch.nn.functional as F

from .utils import compute_nmse
from .hornsmethod import horns_method, robust_horn
from .base_loss import BaseLoss


class IterNrmseLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
    
    def compute_one_iter(self, t_pred, t_gt, valid_inds=None):
        if self.args['align']:
            if valid_inds is not None:
                t_pred = t_pred[valid_inds]
                t_gt = t_gt[valid_inds]
            t_pred_np = t_pred.cpu().detach().numpy().astype(np.float64)
            t_gt_np = t_gt.cpu().numpy().astype(np.float64)
            if self.args['robust_align']:
                R, r0, s, _ = robust_horn(t_gt_np, t_pred_np, ransac_rounds=100)
            else:
                R, r0, s, _ = horns_method(t_gt_np, t_pred_np)
            R = torch.from_numpy(R).float().to(t_pred.device)
            r0 = torch.from_numpy(r0).float().to(t_pred.device)
            t_pred = s*(R@t_pred.T).T + r0.reshape(1, 3)
        nrmse_loss = compute_nmse(t_pred, t_gt)**0.5
        return nrmse_loss

    def compute(self, outputs_model, inputs_data):
        t_gt = inputs_data['t_gt'][0]
        t_pred_all = outputs_model['t_iters']
        valid_inds = inputs_data.get('valid_inds', None)
        
        gamma = self.args['gamma']

        n = len(t_pred_all)
        loss = 0
        for i in range(n):
            w = gamma**(n-i-1)
            loss_i = self.compute_one_iter(t_pred_all[i], t_gt, valid_inds)
            loss = loss + loss_i*w
            

        return loss, {"nrmse": loss_i}
