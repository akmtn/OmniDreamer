# Diverse Plausible 360-Degree Image Outpainting for Efficient 3DCG Background Creation (CVPR 2022)
![teaser](assets/teaser.png)

[**Diverse Plausible 360-Degree Image Outpainting for Efficient 3DCG Background Creation**](https://akmtn.github.io/omni-dreamer/)<br/>
[Naofumi Akimoto](https://akmtn.github.io/resume.pdf), 
[Yuhi Matsuo](https://ishyuhi.github.io/ImsoHappyYuhi),
[Yoshimitsu Aoki](https://aoki-medialab.jp/home-en/)<br/>


[arXiv](http://arxiv.org/abs/2203.14668) | [BibTeX](#bibtex) | [Project Page](https://akmtn.github.io/omni-dreamer/)

## Requirements
A suitable [conda](https://conda.io/) environment named `omnidreamer` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate omnidreamer
```


## Trained models
Please send us an email. We will send you the URL for downloading. You may distribute the trained models to others, but please do not reveal the URL.


## Running trained models
- Put trained weights under `logs/`
- Comment out ckpt_path to VQGAN models from each `{*}-project.yaml`




# Command examples
## Images sampling for the comparison against 360IC
```
CUDA_VISIBLE_DEVICES=0 python sampling.py \
--config_path logs/2021-07-27T05-57-41_sun360_basic_transformer/configs/2021-07-27T05-57-41-project.yaml \
--ckpt_path logs/2021-07-27T05-57-41_sun360_basic_transformer/checkpoints/last.ckpt \
--config_path_2 logs/2021-07-27T10-49-57_sun360_refine_net/configs/2021-07-27T10-49-57-project.yaml \
--ckpt_path_2 logs/2021-07-27T10-49-57_sun360_refine_net/checkpoints/last.ckpt \
--outdir outputs/test
```

## Images sampling for the comparison against SIG-SS
```
CUDA_VISIBLE_DEVICES=0 python sampling.py \
--config_path logs/2021-07-27T05-57-41_sun360_basic_transformer/configs/2021-07-27T05-57-41-project.yaml \
--ckpt_path logs/2021-07-27T05-57-41_sun360_basic_transformer/checkpoints/last.ckpt \
--config_path_2 logs/2021-07-27T10-49-57_sun360_refine_net/configs/2021-07-27T10-49-57-project.yaml \
--ckpt_path_2 logs/2021-07-27T10-49-57_sun360_refine_net/checkpoints/last.ckpt \
--mask_path assets/90binarymask.png \
--outdir outputs/test
```


## Images sampling for the comparison against EnvMapNet
```
CUDA_VISIBLE_DEVICES=0 python sampling.py \
--config_path logs/2021-08-12T03-27-04_sun360_basic_transformer/configs/2021-08-12T03-27-04-project.yaml \
--ckpt_path logs/2021-08-12T03-27-04_sun360_basic_transformer/checkpoints/last.ckpt \
--config_path_2 logs/2021-08-12T03-42-53_sun360_refine_net/configs/2021-08-12T03-42-53-project.yaml \
--ckpt_path_2 logs/2021-08-12T03-42-53_sun360_refine_net/checkpoints/last.ckpt \
--mask_path assets/90binarymask.png \
--outdir outputs/test
```


## Development environment
- Ubuntu 18.04
- Titan RTX or RTX 3090
- CUDA11


## License
This repo is built on top of VQGAN. See the [license](https://github.com/CompVis/taming-transformers/blob/master/License.txt).


## BibTeX
```
@inproceedings{akimoto2022diverse,
    author    = {Akimoto, Naofumi and Matsuo, Yuhi and Aoki, Yoshimitsu},
    title     = {Diverse Plausible 360-Degree Image Outpainting for Efficient 3DCG Background Creation},
    booktitle   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
}
```