import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.utils  import get_homograpy, gps2distance
from models.utils.Mercator import get_latlon_tensor, get_pixel_tensor

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=10, sample = True):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.sample = sample
        self.F_LogSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, sim_matrix, positive_indices):       
        batch_size, h, w = sim_matrix.shape      
        sim_matrix_logexp = -self.F_LogSoftmax(sim_matrix.view(batch_size,-1)/self.temperature)
        positive_indices = 2*positive_indices/(w-1)- 1
        sim_matrix_exp = F.grid_sample(sim_matrix_logexp.view(batch_size,1,h, w), positive_indices.unsqueeze(1).unsqueeze(1), align_corners=True)
            
        loss = torch.mean(sim_matrix_exp)
        
        return loss

def corr_loss(grd_gps, sat_gps, corr_fn, infoLoss, args,  sat_delta=None, transformed_center = None, sz = [512,512]):
    zoom = args.zoom
    sat_size = args.sat_size
    batch_sz =  grd_gps.shape[0]
    h,w = corr_fn.shape[-2:]    
    if transformed_center is not None:
        corr_map = corr_fn.view(batch_sz,h,w,h,w).permute(0,3,4,1,2).view(batch_sz,h*w,h,w) 
        transformed_center_ = transformed_center[:,:,0]*h/sz[0] # /4
        transformed_center_ = 2*transformed_center_/(corr_map.shape[-1]-1) -1 
        corr_map = F.grid_sample(corr_map, transformed_center_.unsqueeze(1).unsqueeze(1), align_corners=True) # [x,y]
        corr_map = corr_map.view(batch_sz,h,w)
    else: 
        corr_map = corr_fn.view(batch_sz,h,w,h,w)[:,h//2,w//2,:,:]  

    if len(grd_gps.shape) == 3:
        y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], grd_gps[:,0,0],grd_gps[:,0,1], zoom, sat_size)
    else:
        y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], grd_gps[:,0],grd_gps[:,1], zoom, sat_size) # get ground truth pixel coords
    y = torch.cat((y[0].reshape(-1,1),y[1].reshape(-1,1)),dim = 1)
    if sat_delta is not None:
        y = sat_delta/sz[0]*sat_size
    loss = infoLoss(corr_map, y/sat_size*corr_map.shape[-1])

    return loss

def vigor_gps_loss(four_pred, grd_gps, sat_gps, args, sat_delta=None, orien = False, transformed_center = None, sz = [512,512], gamma = 0.85, ori_angle = 0, w3 = 10.0):
    """
    loss: pixel_wise
    grd_gps: ground truth GPS of grd images
    """
    zoom = args.zoom
    sat_size = args.sat_size

    sz = [grd_gps.shape[0] ]+ [1] + sz
    n_predictions = len(four_pred) if type(four_pred) == list else 1
    v_loss = 0.0

    # print(x[0,0].item(), x[0,1].item())
    
    pot_num = 0
    if len(grd_gps.shape) == 3:
        batch_num, pot_num = grd_gps.shape[:2]
        sat_gps = sat_gps.unsqueeze(1).repeat(1,pot_num,1).reshape(-1,2)
        grd_gps = grd_gps.reshape(-1,2)

    y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], grd_gps[:,0],grd_gps[:,1], zoom, sat_size=sat_size) # get ground truth pixel coords
    y = torch.cat((y[0].reshape(-1,1),y[1].reshape(-1,1)),dim = 1) #  [batch*pot_num, 2]
    if sat_delta is not None:
        y = sat_delta/sz[2]*sat_size

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        H = get_homograpy(four_pred[i], sz) if type(four_pred) == list else get_homograpy(four_pred, sz)
        if transformed_center is None:
            points = torch.cat((torch.ones((1,1))*sz[3]//2.0, torch.ones((1,1))*sz[2]//2.0, torch.ones((1,1))),
                            dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
        else:
            points = torch.cat((transformed_center, torch.ones((sz[0],1,transformed_center.shape[-1])).to(grd_gps.device)), dim = 1).to(grd_gps.device)
        x = H.bmm(points)
        x = x / x[:, 2, :].unsqueeze(1)        
        x[:,:2,:] = x[:,:2,:]
        if orien:
            dx = x[:,0, 1]- x[:,0, 0]
            dy = x[:,1, 0]- x[:,1, 1]     
            ori = -torch.rad2deg(torch.atan2(dx,dy))
            ori_loss = (ori-ori_angle).abs()
            ori_loss = ori_loss.nanmean()
        x = x[:, 0:2, :]/sz[2]*sat_size      # [batch, 2, put_num]
        if pot_num!= 0:
            x=x.permute(0,2,1).reshape(-1,2)
        else:
            x = x[:,:,0]
        i_loss = torch.nanmean((x-y)**2)/sat_size*sz[2]/sat_size*sz[2]
        i_loss += ori_loss*w3 if orien else 0
        v_loss += i_weight * i_loss    

    est_lat, est_lon = get_latlon_tensor(sat_gps[:,0], sat_gps[:,1], x[:,0], x[:,1], zoom, sat_size)
    epe = gps2distance(grd_gps[:,0],grd_gps[:,1], est_lat, est_lon)

    metrics = { # for people
        'epe': epe.nanmean().item(), 
        '1px': (epe < 1).float().mean().item(), # R@1
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return v_loss, metrics 

def kitti_ori_loss(four_pred, grd_gps, args, ori_angle, sz = [512,512], gamma = 0.85):
    sz = [grd_gps.shape[0] ]+ [1] + sz
    n_predictions = len(four_pred) if type(four_pred) == list else 1
    v_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        H = get_homograpy(four_pred[i], sz) if type(four_pred) == list else get_homograpy(four_pred, sz)
        points = torch.cat((torch.ones((1,1))*sz[3]//2.0, torch.ones((1,1))*sz[2]//2.0, torch.ones((1,1))),
                                    dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
        points_ = torch.cat((torch.ones((1,1))*sz[3]//2.0, torch.ones((1,1))*sz[2]//2.0-10, torch.ones((1,1))),
                                    dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
        points = torch.cat((points,points_), dim = 2)
        x = H.bmm(points)
        x = x / x[:, 2, :].unsqueeze(1)        
        x = x[:, 0:2, :]/sz[2]*args.sat_size
        dx = x[:,0, 1]- x[:,0, 0]
        dy = x[:,1, 0]- x[:,1, 1]     
        ori = -torch.rad2deg(torch.atan2(dx,dy))
        ori_epe = (ori-ori_angle).abs()
        ori_loss = ori_epe # - torch.min(torch.ones_like(ori_loss), ori_epe)
        ori_loss = ori_loss.nanmean()
        i_loss = ori_loss
        # torch.nanmean((x-y)**2)
        v_loss += i_weight * i_loss    
    return v_loss