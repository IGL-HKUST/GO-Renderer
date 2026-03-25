# GO-Renderer: Generative Object Rendering with 3D-aware Controllable Video Diffusion Models

<p align="center">
  <a href="https://scholar.google.com/citations?user=Y8AU3RkAAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Zekai Gu</a><sup>1,2</sup>,
  Shuoxuan Feng<sup>3</sup>,
  Yansong Wang<sup>4</sup>,
  <a href="https://openreview.net/profile?id=~Hanzhuo_Huang1" target="_blank" rel="noopener noreferrer">Hanzhuo Huang</a><sup>5</sup>,
  Zhongshuo Du<sup>1</sup>,
  <a href="https://afterjourney00.github.io/" target="_blank" rel="noopener noreferrer">Chengfeng Zhao</a><sup>1</sup>,
  <a href="https://github.com/ChernweiRen" target="_blank" rel="noopener noreferrer">Chengwei Ren</a><sup>1</sup>,
  <a href="https://totoro97.github.io/" target="_blank" rel="noopener noreferrer">Peng Wang</a><sup>2‡</sup>,
  <a href="https://liuyuan-pal.github.io/" target="_blank" rel="noopener noreferrer">Yuan Liu</a><sup>1†</sup>
</p>

<p align="center">
  <sup>1</sup>HKUST &nbsp;&nbsp;
  <sup>2</sup>VAST &nbsp;&nbsp;
  <sup>3</sup>Nanyang Technological University &nbsp;&nbsp;
  <sup>4</sup>Tsinghua University &nbsp;&nbsp;
  <sup>5</sup>ShanghaiTech University
</p>

<p align="center">
  <sup>†</sup> Corresponding author &nbsp;&nbsp;
  <sup>‡</sup> Project leader
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.23246">ArXiv</a> &nbsp;|&nbsp;
  <a href="https://igl-hkust.github.io/GO-Renderer/">Project Page</a> &nbsp;|&nbsp;
  <a href="https://github.com/IGL-HKUST/GO-Renderer">Code</a>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="GO-Renderer teaser" width="92%">
</p>

GO-Renderer integrates reconstructed 3D proxies with controllable video diffusion models to render objects on arbitrary viewpoints under arbitrary lighting conditions. The 3D proxy provides accurate structural guidance and viewpoint control, while the generative model supplies realistic appearance priors for novel environments, relighting, and object insertion.

## Abstract

Reconstructing a renderable 3D model from images is a useful but challenging task. Recent feedforward 3D reconstruction methods have demonstrated remarkable success in efficiently recovering geometry, but still cannot accurately model the complex appearances of these 3D reconstructed models. Recent diffusion-based generative models can synthesize realistic images or videos of an object using reference images without explicitly modeling its appearance, which provides a promising direction for object rendering, but lacks accurate control over the viewpoints. In this paper, we propose GO-Renderer, a unified framework integrating the reconstructed 3D proxies to guide the video generative models to achieve high-quality object rendering on arbitrary viewpoints under arbitrary lighting conditions. Our method not only enjoys the accurate viewpoint control using the reconstructed 3D proxy but also enables high-quality rendering in different lighting environments using diffusion generative models without explicitly modeling complex materials and lighting. Extensive experiments demonstrate that GO-Renderer achieves state-of-the-art performance across the object rendering tasks, including synthesizing images on new viewpoints, rendering the objects in a novel lighting environment, and inserting an object into an existing video.

## TODO

- [ ] Release project code and checkpoints.
- [ ] Release additional assets and documentation.
