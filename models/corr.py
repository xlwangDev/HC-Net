import torch
import torch.nn.functional as F
from .utils.utils import *

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4): 
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock.corr(fmap1, fmap2)
        self.corr_map = corr
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)  

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords, r = None):
        if r is None:
            r = self.radius
        coords = coords.permute(0, 2, 3, 1) # batch,h,w,2 
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)  # [-4,-3,-2., -1.,  0.,  1.,  2.,3,4]
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx,indexing='xy'), axis=-1).to(coords.device) # 9*9*2  'i,j' (y,x) 'x,y' (x,y)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i # [1*32*32,1,1,2]
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2) # [1, 9, 9, 2]
            coords_lvl = centroid_lvl + delta_lvl # [1024, 9, 9, 2]

            corr = bilinear_sampler(corr, coords_lvl) # [1024, 1,9,9]
            corr = corr.view(batch, h1, w1, -1) # [1, 32, 32, 81]
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1) # [1, 32, 32, 162]   local correlation 
        return out.permute(0, 3, 1, 2).contiguous().float() # batch*channel*H*W

    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.relu(torch.matmul(fmap1.transpose(1, 2), fmap2)) # [batch, h*w, h*w]
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        return corr #/ torch.sqrt(torch.tensor(dim).float())
