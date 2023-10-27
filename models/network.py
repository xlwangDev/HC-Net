import torch
import torch.nn as nn
from .update import GMA
from .corr import CorrBlock
from .utils.utils import *
from .efficientnet_pytorch.model import EfficientNet
from .utils.torch_geometry import get_perspective_transform
# import torchgeometry as tgm


autocast = torch.cuda.amp.autocast

class HCNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        intput_dim = 3
        self.sat_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', circular=False, in_channels = intput_dim)
        self.grd_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', circular=False, in_channels = intput_dim) if args.p_siamese else None
        
        in_dim = 164 if args.flow else 166
        sz = 16 if args.CNN16 else 32
        self.update_block_4 = GMA(self.args, sz, in_dim)
    
    def get_flow_now_k(self, four_point, k = 4):
        N,_,h,w = self.sz
        h, w = h//k, w//k
        four_point = four_point / k # four_point is at original size， coordinate is at feature map size
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
        four_point_org[:, 1, 1] = torch.Tensor([w -1, h-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(N, 1, 1, 1)
        four_point_new = torch.autograd.Variable(four_point_org) + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        # H = tgm.get_perspective_transform(four_point_org, four_point_new)
        H = get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, w-1, steps=w), torch.linspace(0,h-1, steps=h),indexing='ij')
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, w * h))),
                           dim=0).unsqueeze(0).repeat(N, 1, 1).to(four_point.device)
        points_new = H.bmm(points) # (N,3,3) (N,3,w*h)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(N, w, h).unsqueeze(1),
                          points_new[:, 1, :].reshape(N, w, h).unsqueeze(1)), dim=1)
        return flow, H

    def initialize_flow_k(self, img, k= 4):
        N, C, H, W = img.shape
        coords0 = coords_grid(N,H//k, W//k).to(img.device) # [batch,2, H, W]
        coords1 = coords_grid(N, H//k, W//k).to(img.device)

        return coords0, coords1 # [x,y]

    def forward(self, image1, image2, sat_gps = [], iters_lev0 = 6, test_mode=False):
        # 0. Normalize input data
        image1 = 2 * (image1 / 255.0) - 1.0 # 【-1，1】
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()        
        self.sz = image1.shape # [N, 3, 128, 128]
        
        # 1. Using Backbone to Obtain Feature Maps
        grd_feature_volume, multiscale_grd = self.sat_efficientnet.extract_features_multiscale(image1) \
            if self.grd_efficientnet is None else self.grd_efficientnet.extract_features_multiscale(image1)
        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(image2)

        if self.args.CNN16:
            fmap1 = multiscale_grd[15] # [320, 16, 16]]
            fmap2 = multiscale_sat[15] # [320, 16, 16]
        else:
            fmap1 = multiscale_grd[10] # [112, 32, 32]
            fmap2 = multiscale_sat[10] # [112, 32, 32]
        sz = fmap1.shape

        fmap1 = fmap1.float()
        fmap2 = fmap2.float() 

        # 2. Calculate Correlation Matrix      
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4)
        coords0, coords1 = self.initialize_flow_k(image1, k=self.sz[-1]//sz[-1])         
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)

        # 3. Recurrent Homography Estimation
        flow_predictions =[] ## for train 
        for itr in range(iters_lev0):
            corr = corr_fn(coords1) # batch,channel,H,W  correlation
            flow = coords1 - coords0 if self.args.flow else torch.cat((coords1,coords0),dim=1) # [batch,2, H, W] mean x, y
            with autocast(enabled=self.args.mixed_precision):
                delta_four_point = self.update_block_4(corr, flow) # input shape: [b,2+channel,H,W] , output shape [b,2,2,2]

            four_point_disp =  four_point_disp + delta_four_point

            coords1, _ = self.get_flow_now_k(four_point_disp, k=self.sz[-1]//sz[-1])
            flow_predictions.append(four_point_disp) ## for train 
        
        # 4. Output
        coords1,H = self.get_flow_now_k(four_point_disp, k=1) 
        points = torch.cat((torch.ones((1,1))*self.sz[3]//2.0, torch.ones((1,1))*self.sz[2]//2.0, torch.ones((1,1))),
                        dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point_disp.device)# [N,2,1] only one point
        offset = torch.zeros_like(points[:,:2,0])
        self.corr_fn = corr_fn
        if test_mode:
            return four_point_disp #, offset
        else:
            return flow_predictions, corr_fn.corr_map #, offset