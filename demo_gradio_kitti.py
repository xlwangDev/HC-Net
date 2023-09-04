import numpy as np
import torch
import warnings
import gradio as gr
from models.utils.torch_geometry import get_perspective_transform, warp_perspective

warnings.filterwarnings("ignore")

def get_BEV_kitti(front_img, fov, pitch, scale, out_size): 
    Hp, Wp  = front_img.shape[:2]

    Wo,Ho = int(Wp*scale),int(Wp*scale)  

    fov = fov *torch.pi/180                               # 
    theta = pitch*torch.pi/180             # Camera pitch angle


    f = Hp/2/torch.tan(torch.tensor(fov))
    phi = torch.pi/2 - fov
    delta = torch.pi/2+theta - torch.tensor(phi)
    l = torch.sqrt(f**2+(Hp/2)**2)
    h = l*torch.sin(delta)
    f_ = l*torch.cos(delta)

    ######################

    frame = torch.from_numpy(front_img).to(device)  

    out = torch.zeros((2, 2,2)).to(device)

    y = (torch.ones((2, 2)).to(device).T  *(torch.arange(0,Ho, step=Ho-1)).to(device)).T
    x = torch.ones((2, 2)).to(device)  *torch.arange(0, Wo, step=Wo-1).to(device)  
    l0 = torch.ones((2, 2)).to(device)*Ho - y
    l1 = torch.ones((2, 2)).to(device) * f_+ l0

    f1_0 =  torch.arctan(h/l1)
    f1_1 =  torch.ones((2, 2)).to(device)*(torch.pi/2+theta) - f1_0
    y_ = l0*torch.sin(f1_0)/torch.sin(f1_1)
    j_p = torch.ones((2, 2)).to(device) * Hp - y_
    i_p = torch.ones((2, 2)).to(device) * Wp/2 -(f_+torch.sin(torch.tensor(theta))*(torch.ones((2, 2)).to(device)*Hp-j_p))*(Wo/2*torch.ones((2, 2)).to(device)-x)/l1

    out[:,:,0] = i_p.reshape((2, 2))
    out[:,:,1] = j_p.reshape((2, 2))

    four_point_org = out.permute(2,0,1)
    four_point_new = torch.stack((x,y), dim = -1).permute(2,0,1)
    four_point_org = four_point_org.unsqueeze(0).flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.unsqueeze(0).flatten(2).permute(0, 2, 1)
    H = get_perspective_transform(four_point_org, four_point_new)

    scale1,scale2 = out_size/Wo,out_size/Ho
    T3 = np.array([[scale1, 0, 0], [0, scale2, 0], [0, 0, 1]])
    Homo = torch.matmul(torch.tensor(T3).unsqueeze(0).to(device).float(), H) 
    BEV = warp_perspective(frame.permute(2,0,1).unsqueeze(0).float(), Homo, (out_size,out_size))

    BEV = BEV[0].cpu().int().permute(1,2,0).numpy().astype(np.uint8)

    return BEV

@torch.no_grad()
def KittiBEV():
    torch.cuda.empty_cache()
    
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # HC-Net: Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator
            ## Get BEV from front-view image. 
            [[Paper](https://arxiv.org/abs/2308.16906)]  [[Code](https://github.com/xlwangDev/HC-Net)]
            """)

        with gr.Row():
            front_img = gr.Image(label="Front-view Image").style(height=450)
            BEV_output = gr.Image(label="BEV Image").style(height=450)

        fov = gr.Slider(1,90, value=20, label="FOV")
        pitch = gr.Slider(-180, 180, value=0, label="Pitch")
        scale = gr.Slider(1, 10, value=1.0, label="Scale")
        out_size = gr.Slider(500, 1000, value=500, label="Out size")
        btn = gr.Button(value="Get BEV Image")
        btn.click(get_BEV_kitti,inputs= [front_img, fov, pitch, scale, out_size], outputs=BEV_output, queue=False)
        gr.Markdown(
            """
            ### Note: 
            - 'FOV' represents the field of view in the camera's vertical direction, please refer to section A.2 in the [paper](https://arxiv.org/abs/2308.16906)'s Supplementary.
            - By default, the camera faces straight ahead, with a 'pitch' of 0 resulting in a top-down view. Increasing the 'pitch' tilts the BEV view upwards.
            - 'Scale' affects the field of view in the BEV image; a larger 'Scale' includes more content in the BEV image.
            """
        )


        gr.Markdown("## Image Examples")
        gr.Examples(
            examples=[['./figure/exp1.jpg', 27, 7, 6, 1000],
                      ['./figure/exp2.png', 17.5, 0.8, 4, 1000]],
            inputs= [front_img, fov, pitch, scale, out_size],
            outputs=[BEV_output],
            fn=get_BEV_kitti,
            cache_examples=False,
        )
    demo.launch()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    KittiBEV()