# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

#### Code from the course [Introduction to NeRF, volumetric rendering, and 3D reconstruction](https://www.udemy.com/course/neural-radiance-fields-nerf/?referralCode=DD33817D57404AF048DF).

## Usage

**Dataset:** [Download the training and testing datasets](https://drive.google.com/drive/folders/18bwm-RiHETRCS5yD9G00seFIcrJHIvD-?usp=sharing).
```commandline
$ pip3 install -r requirements.txt
$ python3 nerf.py
```

## Results



#### Novel views rendered from the optimized NeRF representation



 ![](novel_views/img_0_hf_6.png)              |  ![](novel_views/img_10_hf_6.png) 
:-------------------------:|:-------------------------:
![](novel_views/img_20_hf_6.png)  |  ![](novel_views/img_30_hf_6.png)


![](novel_views/novel_views.gif)

## Credits

- The datasets were generated with the code from [kwea123/nerf_pl](https://github.com/kwea123/nerf_pl/blob/master/datasets/blender.py).
- The training hyperparameters were retrieved from [kwea123/nerf_pl](https://github.com/kwea123/nerf_pl).

## What is not implemented

- Hierarchical volume sampling