import numpy as np
import os
import torch
import argparse
from models.network import HCNet
from models.utils.utils import *
import numpy as np
import dataset as datasets
from models.utils.loss_factory import *
import time
import warnings
warnings.filterwarnings("ignore")

SUM_FREQ = 100

setup_seed(2023)

@torch.no_grad()
def evaluate_HCNet(model, val_dataset, args = None):
    torch.cuda.empty_cache()
    logger = Logger(args)
    timeall=[]
    distance_in_meters = []
    orientation_error = []
    probability_at_gt = []
    for i_batch, data_blob in enumerate(val_dataset):
        img1, img2, grd_gps,  sat_gps,ori_angle, sat_delta  = [x.to(model.device) for x in data_blob]
        sat_delta = sat_delta if args.orig_label and args.dataset=='vigor' else None

        time_start = time.time()
        four_pr  = model(img1, img2, sat_gps=sat_gps.float(), iters_lev0=args.iters_lev0, test_mode=True)       
        time_end = time.time()

        zoom = args.zoom
        sat_size=args.sat_size
        # get ground truth pixel coords
        y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], grd_gps[:,0],grd_gps[:,1], zoom, sat_size=sat_size) 
        y = torch.cat((y[0].reshape(-1,1),y[1].reshape(-1,1)),dim = 1)

        if sat_delta is not None:
            y = sat_delta/img1.shape[2]*sat_size

        # get predicted truth pixel coords
        H = get_homograpy(four_pr[-1], img1.shape) if type(four_pr) == list else get_homograpy(four_pr, img1.shape)        
        points = torch.cat((torch.ones((1,1))* img1.shape[3]//2.0, torch.ones((1,1))* img1.shape[2]//2.0, torch.ones((1,1))),
                        dim=0).unsqueeze(0).repeat( img1.shape[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
        x = H.bmm(points)
        x = x / x[:, 2, :].unsqueeze(1)
        x[:,:2,0] = x[:,:2,0]
        x = x[:, 0:2, 0]/img1.shape[2]*sat_size      # [batch, 2]

        # get predicted [lat, lon]
        est_lat, est_lon = get_latlon_tensor(sat_gps[:,0], sat_gps[:,1], x[:,0], x[:,1], zoom, sat_size = sat_size)
        epe = gps2distance(grd_gps[:,0],grd_gps[:,1], est_lat, est_lon)
        timeall.append(time_end-time_start)

        # Get probability of the result
        corr_fn = model.corr_fn
        corr_map = corr_fn.corr_pyramid[0]
        h,w = corr_map.shape[-2:]    
        corr_map = corr_map.view((-1,h,w,h,w))
        temp = h//2
        sim_matrix = corr_map[:,temp,temp,:,:]
        temperature = 400
        batch_size = corr_map.shape[0]    
        y_ = x.reshape(y.shape)
        positive_indices = y_/args.sat_size*corr_map.shape[-1]
        sim_matrix_logexp = nn.Softmax(dim=1)(sim_matrix.reshape(batch_size,-1)/temperature)
        positive_indices = 2*positive_indices/(w-1)- 1
        sim_matrix_exp = F.grid_sample(sim_matrix_logexp.reshape(batch_size,1,h, w), positive_indices.unsqueeze(1).unsqueeze(1), align_corners=True)


        metrics = { 
        'epe': epe.nanmean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),}
        
        logger.push(metrics)
        torch.cuda.empty_cache()

        if True in torch.isnan(epe):
            print('\033[1;91m'+"There is a nan at {}!\033[0m".format(len(distance_in_meters)+torch.isnan(epe).nonzero()))
            print(x[torch.isnan(epe).nonzero()[0].item()], y[torch.isnan(epe).nonzero()[0].item()],
                est_lat[torch.isnan(epe).nonzero()[0].item()].item(), est_lon[torch.isnan(epe).nonzero()[0].item()].item(),
                grd_gps[torch.isnan(epe).nonzero()[0].item()][0].item(), grd_gps[torch.isnan(epe).nonzero()[0].item()][1].item())

        sz =  img1.shape
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

        for i in range(epe.shape[0]):
            distance_in_meters.append(epe[i].item())
        for i in range(ori_epe.shape[0]):
            orientation_error.append(ori_epe[i].item())
        sim_matrix_exp = sim_matrix_exp.reshape(batch_size)
        for i in range(sim_matrix_exp.shape[0]):
            probability_at_gt.append(sim_matrix_exp[i].item())

        if i_batch%SUM_FREQ == SUM_FREQ-1:        
            logger._print_training_status()
            print('mean localization error (m): ', np.nanmean(distance_in_meters))   
            print('median localization error (m): ', np.nanmedian(distance_in_meters))
            print('num of predicted pairs: ', len(distance_in_meters))

            print('mean orientation error (m): ', np.nanmean(orientation_error))   
            print('median orientation error (m): ', np.nanmedian(orientation_error))
            print('mean probability_ error (m): ', np.nanmean(probability_at_gt))   
            print('median probability_ error (m): ', np.nanmedian(probability_at_gt))


    logger._print_training_status()
    print('mean localization error (m): ', np.nanmean(distance_in_meters))   
    print('median localization error (m): ', np.nanmedian(distance_in_meters))
    print('num of predicted pairs: ', len(distance_in_meters))
    print('mean orientation error (m): ', np.nanmean(orientation_error))   
    print('median orientation error (m): ', np.nanmedian(orientation_error))
    print('mean probability_ error (m): ', np.nanmean(probability_at_gt))   
    print('median probability_ error (m): ', np.nanmedian(probability_at_gt))
    print("Average use time:  {:.2f} ms. All  use time: {:.3f}s".format(np.mean(np.array(timeall[1:-1]))*1000, np.sum(np.array(timeall))))
    if not os.path.exists("res_npy"):
        os.makedirs("res_npy")
    np.save('res_npy/' +'val_results.npy', distance_in_meters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None,help="restore model")
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--dataset', type=str, default='vigor', help='dataset')    
    parser.add_argument('--ori_noise', type=float, default=45.0, help='orientation noise for VIGOR')

    parser.add_argument('--lev0', default=True, action='store_true',
                        help='warp no')
    parser.add_argument('--flow', default=False, action='store_true',
                        help='GMA input shape')      # 
    parser.add_argument('--augment', default=False, action='store_true',
                        help='Use albumentations to augment data')      # 
    parser.add_argument('--orien', default=False, action='store_true',
                        help='Add orientation loss')      # 
    parser.add_argument('--p_siamese', default=False, action='store_true',
                        help='Use siamese or pseudo-siamese backbone')      # Siamese
    parser.add_argument('--cross_area', default=False, action='store_true',
                        help='Cross_area or same_area')      # Siamese
    parser.add_argument('--CNN16', default=False, action='store_true',
                        help='Feature map size')      # 
    parser.add_argument('--orig_label', default=False, action='store_true',
                        help='Choose label for VIGOR')      # 

    parser.add_argument('--name', default='HC-Net', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, default='validation') # train or validation

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--sat_size', type=int, default=640)
    parser.add_argument('--zoom', type=int, default=20)

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    model = HCNet(args)
    model_dict = model.state_dict()
    
    if args.model is not None:
        model_med = torch.load(args.model, map_location='cuda:0')
        print("Have load state_dict from: {}".format(args.model))
    elif args.restore_ckpt is not None:
        if os.path.isfile(args.restore_ckpt):
            checkpoint = torch.load(args.restore_ckpt)
            best_dis = checkpoint['best_dis']
            args.start_step = checkpoint['steps']
            model_med = checkpoint['model']
            print('Best distance so far {}.'.format(best_dis))
            print('Load checkpoint at steps {}.'.format(args.start_step))        
            print("Have load state_dict from: {}".format(args.restore_ckpt))    
    
    print('\033[1;91m'+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) +"\033[0m")
    print('\033[1;91m'+args.name +"\033[0m")
    for k, v in model_med.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict:
            model_dict[k].copy_(v)
        else:
            print('Warning: key %s not found in model' % k)
    # missing_keys, unexpected_keys = model.load_state_dict(model_med, strict=True)
    model.load_state_dict(model_dict, strict=True)

    model.to(device) 
    model.eval()

    val_dataset = datasets.fetch_dataloader(args, split=args.validation) #validation
    evaluate_HCNet(model, val_dataset, args=args)