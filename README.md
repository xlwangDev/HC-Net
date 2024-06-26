<h1 align="center"><strong>HC-Net: Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator</strong></h1>

<p align="center">
  <a href="https://arxiv.org/abs/2308.16906" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2308.16906-blue?">
  </a> 
  <a href="https://arxiv.org/pdf/2308.16906.pdf" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📖-blue?">
  </a> 
  <a href="http://101.230.144.196:7860/" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-&#x1f917-blue">
  </a>
  <a href="https://huggingface.co/spaces/Xiaolong-Wang/HC-Net" target='_blank'>
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
  </a>
</p>

## 🏠 About

![image-20230831214545912](./figure/pipeline.png)

We introduce a novel approach to fine-grained cross-view geo-localization. Our method **aligns a warped ground image with a corresponding GPS-tagged satellite image covering the same area using homography estimation.** We first employ a differentiable **spherical transform**, adhering to geometric principles, to **accurately align the perspective of the ground image with the satellite map.** To address challenges such as occlusion, small overlapping range, and seasonal variations, we propose a robust correlation-aware homography estimator to align similar parts of the transformed ground image with the satellite image. Our method achieves **sub-pixel resolution and meter-level GPS accuracy** by mapping the center point of the transformed ground image to the satellite image using a homography matrix and determining the orientation of the ground camera using a point above the central axis. Operating at a speed of 30 FPS, our method outperforms state-of-the-art techniques, **reducing the mean metric localization error by 21.3% and 32.4%** in same-area and cross-area generalization tasks on the VIGOR benchmark, respectively, and by **34.4% on the KITTI benchmark in same-area evaluation.**

## 🔥 News

- [2024-04-27] We release the training codes for the VIGOR dataset. 
- [2023-10-27] We release the inferencing codes with [checkpoints](https://drive.google.com/drive/folders/1EL6RISnR5lOgz0WtWUYtFhKGcU_nROX9?usp=sharing) as well as the [demo script](https://github.com/xlwangDev/HC-Net/blob/main/demo_gradio.py). You can test HC-Net with your own machines.
- [2023-10-01] We release the code for implementing the spherical transform. For usage instructions, please refer to [Spherical_transform.ipynb](https://github.com/xlwangDev/HC-Net/blob/main/demo/Spherical_transform.ipynb).
- [2023-09-21] HC-Net has been accepted by NeurIPS 2023! 🔥🔥🔥
- [2023-08-30] We release the [paper](https://arxiv.org/abs/2308.16906) of HC-Net and an online gradio [demo](http://101.230.144.196:7860).

## 🤖 Online Demo

~~**HC-Net is online! Try it at [this url](http://101.230.144.196:7860/).**~~

**Please use the [demo script](https://github.com/xlwangDev/HC-Net/blob/main/demo_gradio.py) to deploy the demo locally for testing.**

You can test our model using the data from the **'same_area_balanced_test.txt'** split of the [VIGOR](https://github.com/Jeff-Zilence/VIGOR) dataset, or by providing your own Panorama image along with its corresponding Satellite image.

<img src="./figure/Demo.png" alt="image-20230831204530724" style="zoom: 80%;" />

## 📦 Training and Evaluation

### Installation

We train and test our codes under the following environment:

- Ubuntu 18.04
- CUDA 12.0
- Python 3.8.16
- PyTorch 1.13.0

To get started, follow these steps: 

1. Clone this repository.

```bash
git clone https://github.com/xlwangDev/HC-Net.git
cd HC-Net
```

2. Install the required packages.

```bash
conda create -n hcnet python=3.8 -y
conda activate hcnet
pip install -r requirements.txt
```

### Training

```bash
sh train.sh
```

### Evaluation

To evaluate the HC-Net model, follow these steps:

1. Download the [VIGOR](https://github.com/Jeff-Zilence/VIGOR) dataset and set its path to '/home/< usr >/Data/VIGOR'.
2. Download the [pretrained models](https://drive.google.com/drive/folders/1EL6RISnR5lOgz0WtWUYtFhKGcU_nROX9?usp=sharing) and place them in the './checkpoints/VIGOR '.
3. Run the following command:

````bash
chmod +x val.sh
# Usage: val.sh [same|cross]
# For same-area in VIGOR
./val.sh same 0
# For cross-area in VIGOR
./val.sh cross 0
````

4. You can also observe the visualization results of the model through a demo based on gradio. Use the following command to start the demo, and open the local URL: [http://0.0.0.0:7860](http://0.0.0.0:7860/).

```bash
python demo_gradio.py
```

## 🏷️ Label Correction for [VIGOR](https://github.com/Jeff-Zilence/VIGOR) Dataset

<img src="./figure/VIGOR_label.png" alt="image-20230831204530724" style="zoom: 60%;" />

We propose the use of [Mercator projection](https://en.wikipedia.org/wiki/Web_Mercator_projection#References) to directly compute the pixel coordinates of ground images on specified satellite images using the GPS information provided in the dataset. You can find the specific code at [Mercator.py](https://github.com/xlwangDev/HC-Net/blob/main/models/utils/Mercator.py).

To use our corrected label, you can add the following content to the `__getitem__` method of the `VIGORDataset` class in `datasets.py` file in the [CCVPE](https://github.com/tudelft-iv/CCVPE) project:

```python
from Mercator import *

pano_gps = np.array(self.grd_list[idx][:-5].split(',')[-2:]).astype(float)   
pano_gps = torch.from_numpy(pano_gps).unsqueeze(0) 

sat_gps = np.array(self.sat_list[self.label[idx][pos_index]][:-4].split('_')[-2:]).astype(float)
sat_gps = torch.from_numpy(sat_gps).unsqueeze(0)     

zoom = 20
y = get_pixel_tensor(sat_gps[:,0], sat_gps[:,1], pano_gps[:,0],pano_gps[:,1], zoom) 
col_offset_, row_offset_ = y[0], y[1]

width_raw, height_raw = sat.size
col_offset, row_offset = width_raw/2 -col_offset_.item(), row_offset_.item() - height_raw/2
```

## 📷 Get BEV Image from front-view

We have released the code corresponding to section A.2 in the [paper](https://arxiv.org/abs/2308.16906)'s Supplementary, along with an online testing platform [<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">](https://huggingface.co/spaces/Xiaolong-Wang/HC-Net).

Compared to traditional **Inverse Perspective Mapping (IPM)**, our approach does not require calibration of camera parameters. Instead, it allows for manual tuning to achieve an acceptable BEV projection result.

You can use Hugging Face Spaces for online testing, or run our code locally. Online testing utilizes CPU for computation, which is slower. If you run it locally with a GPU, the projection process takes less than 10ms.

```bash
python demo_gradio_kitti.py
```

Our projection process is implemented entirely in PyTorch, which means **our projection method is differentiable and can be directly deployed in any network for gradient propagation.**

##### Example of KITTI

<img src="./figure/Example_kitti.png" alt="image-20230904150231834" style="zoom:80%;" />

##### Example of a Random Network Image

<img src="./figure/Example_random.png" alt="image-20230904150208550" style="zoom:80%;" />

## 📝 TODO List

- [x] Add data preparation codes.
- [x] Add inferencing and serving codes with checkpoints.
- [x] Add evaluation codes.
- [x] Add training codes.

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{wang2024fine,
  title={Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator},
  author={Wang, Xiaolong and Xu, Runsen and Cui, Zhuofan and Wan, Zeyu and Zhang, Yu},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## 👏 Acknowledgements

- This work is mainly based on [IHN](https://github.com/imdumpl78/IHN) and [RAFT](https://github.com/princeton-vl/RAFT), we thank the authors for the contribution.
