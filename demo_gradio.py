import numpy as np
import os
import torch
import argparse
from models.network import HCNet
from models.utils.utils import *
import scipy.io as io
import numpy as np
import dataset as datasets
from models.utils.loss_factory import *
import warnings
import gradio as gr
from PIL import Image
import re

# setting GRADIO_TEMP_DIR
os.environ['GRADIO_TEMP_DIR'] = './tmp'

def is_valid_gps_format(gps_str):
    gps_str = gps_str.replace(" ", "")
    gps_pattern = r'^[-]?(\d+\.\d{6,}),[-]?(\d+\.\d{6,})$'
    match = re.match(gps_pattern, gps_str)
    return match is not None

warnings.filterwarnings("ignore")

setup_seed(2023)

# Define a function to load the model checkpoint
def load_model(checkpoint_path):
    checkpoint_path = "./checkpoints/" + checkpoint_path

    model_dict = model.state_dict()    

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_med = checkpoint['model']

    for k, v in model_med.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict:
            model_dict[k].copy_(v)
        else:
            print('Warning: key %s not found in model' % k)
    model.load_state_dict(model_dict, strict=True)
    device = torch.device('cuda:'+ str(args.gpuid[0]))
    model.to(device) 
    model.eval()
    print(checkpoint_path)

    return None

def get_BEV(pano_image, fov, roll, pitch, yaw, dty):
    dty = int(dty)
    Hp, Wp = pano_image.shape[:2]
    pano_image = np.roll(pano_image,int(yaw/360*pano_image.shape[1]), axis=1)
    dx = pitch/360 * Wp 
    dy = -roll/180 * Hp 
    bev_image = get_BEV_tensor(pano_image,500,500,Fov = fov, dty = dty,  dx = dx, dy = dy).numpy().astype(np.uint8)
    return bev_image

def model_process(BEV_output, Sat_input, alpha,grd_gps, sat_gps):
    patch_size = 512
    h1,w1,_ = BEV_output.shape
    h2,w2,_ = Sat_input.shape

    if is_valid_gps_format(grd_gps) and is_valid_gps_format(sat_gps):
        grd_gps = np.array(grd_gps.replace(" ", "").split(',')).astype(float)
        sat_gps = np.array(sat_gps.replace(" ", "").split(',')).astype(float)
        print("Both strings are valid GPS coordinates.")
        grd_gps = torch.from_numpy(grd_gps).unsqueeze(0).float().cuda() # [batch, 2]
        sat_gps = torch.from_numpy(sat_gps).unsqueeze(0).float().cuda()
        zoom = 20
        y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], grd_gps[:,0],grd_gps[:,1], zoom) # get ground truth pixel coords
        g_u, g_v = [ite.item()/640*w2 for ite in y]
    else:
        grd_gps, sat_gps = None, None
        print("At least one of the strings is not a valid GPS coordinate.")


    corners1 = np.array([[0, 0], [0, h1], [w1, 0], [w1, h1]], dtype=np.float32)
    corners2 = np.array([[0, 0], [0, h2], [w2, 0], [w2, h2]], dtype=np.float32)
    corners_patch = np.array([[0, 0], [0, patch_size], [patch_size, 0], [patch_size, patch_size]], dtype=np.float32)
    H1 = cv2.getPerspectiveTransform(corners1, corners_patch)
    H2 = cv2.getPerspectiveTransform(corners2, corners_patch)

    img1_ = cv2.resize(BEV_output, (patch_size, patch_size))
    img2_ = cv2.resize(Sat_input, (patch_size, patch_size))

    img1_ = torch.from_numpy((img1_)).float().permute(2, 0, 1).unsqueeze(0)
    img2_ = torch.from_numpy((img2_)).float().permute(2, 0, 1).unsqueeze(0)

    img1_ = img1_.to(model.device)
    img2_ = img2_.to(model.device)

    # run Model
    four_pred = model(img1_, img2_, iters_lev0=args.iters_lev0, test_mode=True)

    H = get_homograpy(four_pred, img1_.shape)
    H = H[0].detach().cpu().numpy()
    H_ = np.linalg.inv(H2).dot(H).dot(H1)

    # get overlap
    h,w = Sat_input.shape[0],Sat_input.shape[1]
    pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    center = np.float32( [w1/2, h1/2]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    dst_center = cv2.perspectiveTransform(center, H).reshape(-1, 2)   
    warped = cv2.warpPerspective(cv2.drawMarker(BEV_output,(int(h1//2),int(w1//2)), color=(0, 0, 255), markerType=cv2.MARKER_STAR, thickness=4), H_, (w, h)) 
    warped_ = draw_markers(np.ascontiguousarray(warped.copy()), [int(dst_center[0][0]),int(dst_center[0][1])], size=4, thickness=2, color=(0, 0, 255), shape = 2)
    cv2.polylines(warped_, [np.int32(dst)], True, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.circle(warped_,(int(dst_center[0][0]),int(dst_center[0][1])),15,(0,255,0),2)
    overlaped = cv2.addWeighted(Sat_input, alpha,  warped_, 1-alpha,0)

    corr_fn = model.corr_fn
    h_temp = corr_fn.corr_pyramid[0].shape[-1]
    corr_map = F.interpolate(corr_fn.corr_pyramid[0], size=(h2, w2), mode='bilinear', align_corners=True)

    # get heatmap
    fig, ax = plt.subplots()
    ax.imshow(Sat_input)
    ax.scatter(dst_center[0][0], dst_center[0][1], s=200, color=(1,1,0), alpha = 1, marker = "*",  edgecolor='black')
    ax.axis('off')

    # Save the figure as a temporary image file
    temp_image_path = "temp_image.png"  
    plt.savefig(temp_image_path,dpi=200, bbox_inches='tight', pad_inches=0)
    pil_image = Image.open(temp_image_path)
    heatmap_result = np.array(pil_image)


    # Get final result
    fig, ax = plt.subplots()
    ax.imshow(Sat_input)
    heatmap = corr_map[h_temp//2*h_temp+h_temp//2, 0, :, :].cpu().detach().numpy()
    vmin = np.min(heatmap)
    vmax = np.max(heatmap)
    im = ax.imshow(heatmap, cmap='jet', alpha=(heatmap - vmin) / (vmax - vmin)/1.4)

    if grd_gps is not None:
        ax.scatter(g_u, g_v, s=200, c='g', alpha = 1, marker = "^",  edgecolor='white', label='GT_label')  

        sz =  img1_.shape
        points = torch.cat((torch.ones((1,1))*sz[3]//2.0, torch.ones((1,1))*sz[2]//2.0, torch.ones((1,1))),
                            dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
        points_ = torch.cat((torch.ones((1,1))*sz[3]//2.0, torch.ones((1,1))*sz[2]//2.0-10, torch.ones((1,1))),
                                    dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(grd_gps.device) # [N,2,1] only one point
        points = torch.cat((points,points_), dim = 2)
        x = get_homograpy(four_pred, img1_.shape).bmm(points)
        x = x / x[:, 2, :].unsqueeze(1)      
        x = x[:, 0:2, :]/sz[2]*args.sat_size
        est_lat, est_lon = get_latlon_tensor(sat_gps[:,0], sat_gps[:,1], x[:,0,0], x[:,1,0], zoom, args.sat_size)
        predicted_GPS = f"{est_lat.item():.6f},{est_lon.item():.6f}" # :.6f

        dx = x[:,0, 1]- x[:,0, 0]
        dy = x[:,1, 0]- x[:,1, 1]     
        # ori_loss = torch.rad2deg(torch.atan2(dx,dy)).abs().nanmean()
        ori = -torch.rad2deg(torch.atan2(dx,dy)).item()
        ori = f"{ori:.2f}"
        x = x[:,0,:]

        dis = f"{gps2distance(grd_gps[:,0],grd_gps[:,1], est_lat, est_lon).item():.4f}"
        infoLoss = InfoNCELoss(temperature=4, sample = True)
        loss2 = corr_loss(grd_gps, sat_gps, corr_fn.corr_pyramid[0], infoLoss,  args=args, transformed_center = None, sz = [img1_.shape[2],img1_.shape[3]])
    else:
        predicted_GPS = None
        ori = None
        dis = None

    ax.scatter(int(dst_center[0][0]),int(dst_center[0][1]), s=200, color=(1,1,0), marker = "*",  edgecolor='white', label='Ours '+f"({torch.exp(-loss2).item():.1e})" if dis is not None else 'Ours ')
    ax.axis('off')
    plt.legend(labelspacing=1)
    plt.savefig(temp_image_path,dpi=200, bbox_inches='tight', pad_inches=0)
    pil_image = Image.open(temp_image_path)
    result = np.array(pil_image)

    return warped, overlaped, heatmap_result, result, predicted_GPS, dis, ori

def run_all(pano_input, Sat_input, fov, roll, pitch, yaw, alpha,sat_GPS, grd_GPS, dty):

    BEV_output = get_BEV(pano_input, fov, roll, pitch, yaw)
    warp_output, overlap_output, heatmap_output = model_process(BEV_output, Sat_input, alpha)

    return BEV_output, warp_output, overlap_output, heatmap_output

def load_data(city, idx):
    idx = int(idx)
    city_num = {'NewYork':0, 'Seattle':13884, 'SanFrancisco':13884+11875, 'Chicago':13884+11875+14107}
    idx = city_num[city] + idx    

    pona_path = val_dataset.dataset.pano_list[idx]
    sat_path = val_dataset.dataset.pano_label[idx][0]
    pano_gps = np.array(pona_path[:-5].split(',')[-2:])
    pano_gps = f"{pano_gps[0]},{pano_gps[1]}" 
    sat_gps = np.array(sat_path[:-4].split('_')[-2:])
    sat_gps =  f"{sat_gps[0]},{sat_gps[1]}" 

    sat = cv2.imread(sat_path, 1)[:,:,::-1] # 
    pona = cv2.imread(pona_path,  1)[:,:,::-1]   # 

    return pona, sat,sat_gps, pano_gps

@torch.no_grad()
def evaluate_HCNet(model, val_dataset, args = None):
    torch.cuda.empty_cache()


    # List of available checkpoints in your specified folder
    checkpoint_folder = "./checkpoints"
    checkpoint_files = [f"{filename}" for filename in os.listdir(checkpoint_folder)]    
    load_model('/VIGOR/best_checkpoint_same.pth')

    
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # HC-Net: Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator
            """)
        # Create a dropdown with available checkpoint files
        # with gr.Row():
        #     checkpoint_dropdown = gr.Dropdown(choices=checkpoint_files, label="Select a checkpoint")
        #     # btn_load = gr.Button(value="🔄")
        # checkpoint_dropdown.change(load_model, checkpoint_dropdown, None)
        with gr.Row():
            with gr.Column():
                city_choice = gr.Radio(['NewYork', 'Seattle', 'SanFrancisco', 'Chicago'], value='Seattle', interactive=True, label='Data City')
                idx_input = gr.Textbox(visible = True,lines=1, label='Data Index', placeholder = 'Press Enter to upload the data')
                # btn_upload = gr.Button(value="Upload the Data")
            gr.Markdown(
                """
                ### Usage:
                1. Upload your Panorama image and its corresponding Satellite image, or choose data from the [VIGOR](https://github.com/Jeff-Zilence/VIGOR) dataset (using the **'same_area_balanced_test'** split).
                2. Click **Get BEV Image** to generate the transformed bird's-eye view image.
                3. Click **Run the Model** to align the BEV image with the corresponding Satellite image and obtain the localization result.
                """)

        with gr.Row():
            with gr.Column():
                pano_input = gr.Image(label="Pona Image").style(height=450) # shape=(1000, 500), 
                with gr.Row():
                    Sat_input = gr.Image(shape=(500, 500), label="Satellite Image").style(height=450)
                    BEV_output = gr.Image(shape=(500, 500), label="BEV Image").style(height=450)

                fov = gr.Slider(10,180, value=170, label="FOV")
                roll = gr.Slider(-180, 180, value=0, label="Roll")
                pitch = gr.Slider(-180, 180, value=0, label="Pitch")
                yaw = gr.Slider(-180, 180, value=0, label="Yaw")
                dty = gr.Slider(-200, 200, value=0, label="Panoramic image completion")
                btn = gr.Button(value="Get BEV Image")
                btn.click(get_BEV,inputs= [pano_input, fov, roll, pitch, yaw, dty], outputs=BEV_output, queue=False)
            with gr.Column():
                with gr.Row():
                    warp_output = gr.Image(shape=(500, 500), label="Warped Image").style(height=450)
                    heatmap_output = gr.Image(shape=(500, 500), label="Heatmap Image").style(height=450)
                with gr.Row():
                    overlap_output = gr.Image(shape=(500, 500), label="Overlaped Image").style(height=450)
                    reault_output = gr.Image(shape=(500, 500), label="Result Image").style(height=450)

                
                alpha = gr.Slider(0, 1, value=0.5, label="alpha for overlap")
                with gr.Row():
                    grd_GPS = gr.Textbox(visible = True,lines=1, label='GPS of ground camera', placeholder='Latitude, longitude') # 47.576563,-122.298433
                    sat_GPS = gr.Textbox(visible = True,lines=1, label='GPS of satellite image center', placeholder='Latitude, longitude') # 47.57667712980035_-122.29857212879756

                with gr.Row():
                    grd_GPS_pre = gr.Textbox(visible = True,lines=1, label='Predicted GPS of ground camera', scale=2)
                    dis_error = gr.Textbox(visible = True,lines=1, label='Localization error (m)', scale=1)
                    ori_error = gr.Textbox(visible = True,lines=1, label='Predicted Orientation', scale=1)

                run_btn = gr.Button(value="Run the Model")
                run_btn.click(get_BEV,inputs= [pano_input, fov, roll, pitch, yaw, dty], outputs=BEV_output, queue=False).then(model_process,inputs= [BEV_output, Sat_input, alpha,grd_GPS, sat_GPS], outputs=[warp_output, overlap_output, heatmap_output, reault_output, grd_GPS_pre, dis_error, ori_error], queue=False)
                alpha.change(get_BEV,inputs= [pano_input, fov, roll, pitch, yaw, dty], outputs=BEV_output, queue=False).then(model_process,inputs= [BEV_output, Sat_input, alpha,grd_GPS, sat_GPS], outputs=[warp_output, overlap_output, heatmap_output, reault_output, grd_GPS_pre, dis_error, ori_error], queue=False)
        
        
        # btn_upload.click(load_data, [city_choice, idx_input], [pano_input, Sat_input,sat_GPS, grd_GPS])
        idx_input.submit(load_data, [city_choice, idx_input], [pano_input, Sat_input,sat_GPS, grd_GPS])
        gr.Markdown(
            """
            ### Note: 
            - If you wish to acquire **quantitative localization error results** for your uploaded data, kindly supply the real GPS for the ground image as well as the corresponding GPS for the center of the satellite image.
            - When inputting GPS coordinates, please make sure their precision extends to **at least six decimal places**.
            """)

        gr.Markdown("## Image Examples")
        gr.Examples(
            examples=[
                      [val_dataset.dataset.pano_list[13929], val_dataset.dataset.pano_label[13929][0], 170, 0, 0, 0,0.5,'47.58619123,-122.32484011', '47.586118,-122.324936',0],],
            inputs= [pano_input, Sat_input, fov, roll, pitch, yaw, alpha,sat_GPS, grd_GPS, dty],
            outputs=[BEV_output, warp_output, overlap_output, heatmap_output, reault_output, dis_error, ori_error, grd_GPS_pre],
            fn=run_all,
            cache_examples=False,
        )
    demo.launch(server_name="0.0.0.0", server_port=7860)

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
    parser.add_argument('--flow', default=True, action='store_true',
                        help='GMA input shape')      # 
    parser.add_argument('--augment', default=False, action='store_true',
                        help='Use albumentations to augment data')      # 
    parser.add_argument('--orien', default=False, action='store_true',
                        help='Add orientation loss')      # 
    parser.add_argument('--p_siamese', default=True, action='store_true',
                        help='Use siamese or pseudo-siamese backbone')      # Siamese
    parser.add_argument('--cross_area', default=False, action='store_true',
                        help='Cross_area or same_area')      # Siamese
    parser.add_argument('--CNN16', default=True, action='store_true',
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

    val_dataset = datasets.fetch_dataloader(args, split=args.validation) #validation
    # val_dataset = datasets.fetch_dataloader(args, split='train') #validation
    evaluate_HCNet(model, val_dataset, args=args)