import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
# import torchgeometry as tgm
from models.utils.torch_geometry import get_perspective_transform
from models.utils.utils import *
from models.utils.loss_factory import *
import time
import cv2

@torch.no_grad()
def validate_process(model,total_steps, val_dataset, args):
    """ Perform evaluation on the validation split """
    model.eval()
    mace_list = []
    timeall=[]
    logger = Logger(args)
    for i_batch, data_blob in enumerate(val_dataset):
        time_start = time.time()
        if args.dataset == 'kitti':
            image1, image2, grd_gps,  sat_gps, transformed_center,sat_delta,_  = [x.to(model.device) for x in data_blob]
        else:
            image1, image2, grd_gps,  sat_gps, transformed_center,sat_delta, ori_angle  = [x.to(model.device) for x in data_blob]
        sat_delta = None

        # Forward pass
        # if args.dataset == 'vigor':
        four_pr  = model(image1, image2, sat_gps=sat_gps.float(), iters_lev0=args.iters_lev0, test_mode=True)       
        _,metrics = vigor_gps_loss(four_pr, grd_gps = grd_gps, transformed_center=transformed_center, sat_delta = sat_delta, \
            sat_gps=sat_gps, sz = [image1.shape[2],image1.shape[3]], args=args)
        logger.push(metrics)
       
        if i_batch == 0:
            if not os.path.exists('watch'):
                os.makedirs('watch')
            pona = cv2.imread("figure/VIGOR_exp/pano.png",  1)[:,:,::-1]   # 全景图
            img1 = get_BEV_tensor(pona,image1.shape[-1],image1.shape[-1],dty = 0, dy = 0, dataset=True).to(model.device).unsqueeze(0)
            sat = cv2.imread('figure/VIGOR_exp/sat.png', 1)[:,:,::-1] # 卫星图
            sat = cv2.resize(sat, (image1.shape[-1], image1.shape[-1]))
            img2 = torch.from_numpy(sat).float().permute(2, 0, 1).to(model.device).unsqueeze(0)
            four_pr  = model(img1, img2, sat_gps=[], iters_lev0=args.iters_lev0, test_mode=True)       

            H = get_homograpy(four_pr, image1.shape)
            H = H.detach().cpu().numpy()
            image1 = img1[0].permute(1, 2,0).detach().cpu().numpy()
            image0 = img2[0].permute(1, 2,0).detach().cpu().numpy()
            plt.figure(figsize=(10,10))
            result = show_overlap(image1, image0, H[0])
            # cv2.imwrite('./watch/' + "result_" + str(total_steps).zfill(5) + '.png',result[:,:,::-1])
            print("save at: {}".format('./watch/' + "result_" + str(total_steps).zfill(5) + '.png'))

        # if args.dataset == 'vigor':
        mace_list.append(metrics['epe'])
        torch.cuda.empty_cache()
        time_end = time.time()
        timeall.append(time_end-time_start)
    
    # if args.dataset == 'vigor':
    mace = np.mean(np.array(mace_list))
    logger._print_training_status()
    print("Validation MDIS: %f" % mace)
    print("Average use time:  {:.2f} ms. All  use time: {:.3f}s".format(np.mean(np.array(timeall[1:-1]))*1000, np.sum(np.array(timeall))))
    torch.cuda.empty_cache()
    model.train()
    return {'val_mace': mace}

@torch.no_grad()
def test_process(model,total_steps, args):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    mace_list = []
    args.batch_size = 1
    logger = Logger(args)
    if args.dataset=='vigor':
        print("Dataset is VIGOR!")
        import dataset as datasets
    val_dataset = datasets.fetch_dataloader(args, split='validation')
    for i_batch, data_blob in enumerate(val_dataset):
        image1, image2, flow_gt,  sat_gps  = [x.to(model.device) for x in data_blob]
        if args.dataset!='vigor':
            flow_gt = flow_gt.squeeze(0)
            flow_4cor = torch.zeros((2, 2, 2))
            flow_4cor[:, 0, 0] = flow_gt[:, 0, 0]
            flow_4cor[:, 0, 1] = flow_gt[:, 0, -1]
            flow_4cor[:, 1, 0] = flow_gt[:, -1, 0]
            flow_4cor[:, 1, 1] = flow_gt[:, -1, -1]

        image1 = image1.to(model.device)
        image2 = image2.to(model.device)
        # Forward pass
        if args.dataset == 'vigor':
            four_pr,_  = model(image1, image2, sat_gps=sat_gps, iters_lev0=args.iters_lev0, test_mode=True)       
            _,metrics = vigor_gps_loss(four_pr, grd_gps = flow_gt, sat_gps=sat_gps, sz = [image1.shape[2],image1.shape[3]])
            logger.push(metrics)
        else:
            four_pr,_,_  = model(image1, image2, iters_lev0=args.iters_lev0, test_mode=True)    
        if i_batch == 0:
            if not os.path.exists('watch'):
                os.makedirs('watch')
            N,C,H0,W0 = image1.shape
            four_point_org = torch.zeros((2, 2, 2)).to(four_pr.device)
            four_point_org[:, 0, 0] = torch.Tensor([0, 0])
            four_point_org[:, 0, 1] = torch.Tensor([W0-1, 0])
            four_point_org[:, 1, 0] = torch.Tensor([0, H0-1])
            four_point_org[:, 1, 1] = torch.Tensor([W0-1, H0-1])
            four_point_org = four_point_org.repeat(four_pr.shape[0], 1, 1, 1)
            four_point_new = four_point_org + four_pr
            H = get_perspective_transform(four_point_org.flatten(2).permute(0,2,1), four_point_new.flatten(2).permute(0,2,1))
            H = H.detach().cpu().numpy()
            image1 = image1[0].permute(1, 2,0).detach().cpu().numpy()
            image0 = image2[0].permute(1, 2,0).detach().cpu().numpy()
            plt.figure(figsize=(10,10))
            result = show_overlap(image1, image0, H[0])
            cv2.imwrite('./watch/' + "result_" + str(total_steps).zfill(5) + '.png',result[:,:,::-1])

        if args.dataset == 'vigor':
            mace_list.append(metrics['epe'])
            torch.cuda.empty_cache()
        else: 
            mace = torch.sum((four_pr[0, :, :, :].cpu() - flow_4cor) ** 2, dim=0).sqrt()
            mace_list.append(mace.view(-1).numpy())
        torch.cuda.empty_cache()
        if i_batch>300:
            break

    model.train()
    if args.dataset == 'vigor':
        mace = np.mean(np.array(mace_list))
    else: 
        mace = np.mean(np.concatenate(mace_list))
    logger._print_training_status()
    print("Validation MACE: %f" % mace)
    return {'chairs_mace': mace}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()
