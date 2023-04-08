# K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

Implementation in 100 lines of code of the paper [K-Planes: Explicit Radiance Fields in Space, Time, and Appearance](https://arxiv.org/abs/2301.10241).

## Usage

**Dataset:** [Download the training and testing datasets](https://drive.google.com/drive/folders/18bwm-RiHETRCS5yD9G00seFIcrJHIvD-?usp=sharing).
```commandline
$ pip3 install -r requirements.txt
$ python3 kplanes.py
```

## Results



#### Novel views rendered from the optimized models



 ![](novel_views/img_0.png)              |  ![](novel_views/img_60.png) 
:-------------------------:|:-------------------------:
![](novel_views/img_120.png)  |  ![](novel_views/img_180.png)


## What is not implemented

- Multiscale Bilinear Interpolation