# SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction
---

**Xinran Yang**, **Donghao ji**, **Yuanqi Li**, **Jie Guo**, **Yanwen Guo**, **Junyuan Xie**


This repository contains the official Pytorch implementation for **SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction** (CVPR2025).

![GitHub Logo](./assets/teaser0_clear.png)


## Installation
---

```
git clone https://github.com/Martinyxr/SGCR.git
cd SGCR
conda env create --file environment.yml 
conda activate SGCR
```

## Demo
---
```
# Training Spherical Gaussians 
python train.py -s ./example/00000006 -m ./output/Gaussain/00000006

# 3D Parametric Curve Reconstrcution 
python ./parametric_curve/curve_fitting.py --object_id 00000006

```
The training command is similar as [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). 
After curve reconstruction, the results will be saved in `./output/curve/`.



## Citation
---
if you find the code useful, please consider the following BibTeX entry.
```bibtex
@InProceedings{yang2025sgcr,
  title        = {SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction},
  author       = {Yang, Xinran and Ji, Donghao and Li, Yuanqi and Guo, Jie and Guo, Yanwen and Xie, Junyuan},
  booktitle    = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2025}
}
```



## Acknowledgments
---
This project is built upon [<u>3DGS</u>](https://github.com/graphdeco-inria/gaussian-splatting). The evaluation ABC-NEF dataset is from [<u>NEF</u>](https://github.com/yunfan1202/NEF_code). We use pretrained [<u>PidiNet</u>](https://github.com/hellozhuo/pidinet) for edge map extraction. We thank all the authors for their great work and repos.
