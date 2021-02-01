# Deforming meshes by reinforcement learning

This repository is based on [3DModelingRL](https://github.com/clinplayer/3DModelingRL) that is the source code for the ECCV 2020 paper [Modeling 3D Shapes by Reinforcement Learning](https://arxiv.org/abs/2003.12397).

```
@article{lin2020modeling,
  title={Modeling 3D Shapes by Reinforcement Learning},
  author={Lin, Cheng and Fan, Tingxiang and Wang, Wenping and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2003.12397},
  year={2020}
}
```


## How to install Pytorch3D ?

Everything is explained in the [README](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) of the pytorch3D lib.

The following command are for my GC only with CUDA 11.2

```bash
curl -LO https://github.com/NVIDIA/cub/archive/1.11.0.tar.gz
tar xzf 1.11.0.tar.gz
export CUB_HOME=$PWD/cub-1.11.0
```

I installed Pytorch3D, with the following command :

```python
git clone "https://github.com/facebookresearch/pytorch3d.git"
cd pytorch3d
pip install -e .
```

## Aim of this project

Our aim is to transform a 3D model (mesh) with deep learning in order to apply an artistic style to it after processing.

**Input:**
Primitive-based shape representation of our wanted model (box)

**Method:** 
Supervised with examples of transformations (box input => 3d mesh output)

**Output:**
3D model (mesh)

## Research papers

Name of the article| Authors | Link
--- | --- | --- |
Modeling 3D Shapes by Reinforcement Learning | Cheng Lin, Tingxiang Fan, Wenping Wang, Matthias Nießner | https://arxiv.org/abs/2003.12397 |
PyTorch3D | Nikhila Ravi, Jeremy Reizenstein, David Novotny, Taylor Gordon, Wan-Yen Lo, Justin Johnson, Georgia Gkioxari | https://arxiv.org/pdf/2007.08501.pdf |
Mesh R-CNN | Georgia Gkioxari, Jitendra Malik, Justin Johnson | https://arxiv.org/pdf/1906.02739.pdf |
Kaolin: A PyTorch Library for Accelerating 3D Deep Learning Research | Krishna Murthy Jatavallabhula, Edward Smith, Jean-Francois Lafleche, Clement Fuji Tsang, Artem Rozantsev, Wenzheng Chen, Tommy Xiang, Rev Lebaredian, Sanja Fidler | https://arxiv.org/pdf/1911.05063.pdf |
Variational Autoencoders for Deforming 3D Mesh Models | Qingyang Tan, Lin Gao1, Yu-Kun Lai, Shihong Xia1 | https://qytan.com/publication/vae/ |
Learning a Neural 3D Texture Space from 2D Exemplars | Philipp Henzler, Niloy J. Mitra, Tobias Ritschel | https://geometry.cs.ucl.ac.uk/projects/2020/neuraltexture/ |
