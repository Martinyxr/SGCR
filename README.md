<p align="center">

  <h1 align="center">SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction</h1>
  <p align="center">
    <strong>Xinran Yang</strong></a>
    ·
    <strong>Donghao ji</strong></a>
    ·
    <strong>Yuanqi Li</strong></a>
    ·
    <strong>Jie Guo</strong></a>
    ·
    <strong>Yanwen Guo</strong></a>
    ·
    <strong>Junyuan Xie</strong></a>
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2505.04668" rel="external nofollow noopener" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2505.04668-B31B1B" alt='Arxiv Link'></a>
  </p>
  <br>


  <p>This repository contains the official <b>Pytorch</b> implementation for <b>SGCR: Spherical Gaussians for 
     Efficient 3D Curve Reconstruction</b> (CVPR2025).</p>
  <div align="center">
    <img src="./assets/teaser0_clear.png" alt="teaser" width="100%">
  </div>
</p>



## Installation


```
git clone https://github.com/Martinyxr/SGCR.git
cd SGCR
conda env create --file environment.yml 
conda activate SGCR
```

## Demo

```
# Training Spherical Gaussians 
python train.py -s ./example/00000006 -m ./output/Gaussain/00000006

# 3D Parametric Curve Reconstrcution 
python ./parametric_curve/curve_fitting.py --object_id 00000006
```
The training command is similar as [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). 
After curve reconstruction, the results will be saved in `./output/curve/`.



## Citation

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

This project is built upon [<u>3DGS</u>](https://github.com/graphdeco-inria/gaussian-splatting). The evaluation ABC-NEF dataset is from [<u>NEF</u>](https://github.com/yunfan1202/NEF_code). We use pretrained [<u>PidiNet</u>](https://github.com/hellozhuo/pidinet) for edge map extraction. We thank all the authors for their great work and repos.
