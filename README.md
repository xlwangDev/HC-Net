# HC-Net: Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator

![image-20230831214545912](./figure/pipeline.png)

### Paper Abstract

In this paper, we introduce a novel approach to fine-grained cross-view geo-localization. Our method aligns a warped ground image with a corresponding GPS-tagged satellite image covering the same area using homography estimation. We first employ a differentiable spherical transform, adhering to geometric principles, to accurately align the perspective of the ground image with the satellite map. This transformation effectively places ground and aerial images in the same view and on the same plane, reducing the task to an image alignment problem. To address challenges such as occlusion, small overlapping range, and seasonal variations, we propose a robust correlation-aware homography estimator to align similar parts of the transformed ground image with the satellite image. Our method achieves sub-pixel resolution and meter-level GPS accuracy by mapping the center point of the transformed ground image to the satellite image using a homography matrix and determining the orientation of the ground camera using a point above the central axis. Operating at a speed of 30 FPS, our method outperforms state-of-the-art techniques, reducing the mean metric localization error by 21.3% and 32.4% in same-area and cross-area generalization tasks on the VIGOR benchmark, respectively, and by 34.4% on the KITTI benchmark in same-area evaluation.

### [Demo of our project](http://101.230.144.196:7860/)

#### Visualization

<img src="./figure/Demo.png" alt="image-20230831204530724" style="zoom: 80%;" />

### Label Correction in VIGOR Dataset

<img src="./figure/VIGOR_label.png" alt="image-20230831204530724" style="zoom: 60%;" />

We propose the use of [Mercator projection](https://en.wikipedia.org/wiki/Web_Mercator_projection#References) to directly compute the pixel coordinates of ground images on specified satellite images using the GPS information provided in the dataset.



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

