DiffusionNet is a general-purpose deep learning approach for learning on surfaces, like 3D triangle meshes and point clouds. It's well-suited for tasks like segmentation, classification, feature extraction

Compared to other approaches, DiffusionNet has several advantages:
- It is _efficient_ and _scalable_. On a single consumer GPU, we can easily train on meshes of 20k vertices, and infer on meshes with 200k vertices. One-time preprocessing takes a few seconds in the former case, and about a minute in the latter.
- It is _sampling agnostic_. Many graph-based mesh learning approaches tend to overfit to mesh connectivity, and can output nonsense when you run them on meshes that are triangulated differently. Similarly, with DiffusionNet can intermingle very coarse or vary fine meshes without issue.
- It is _representation agnostic_. For instance, you can train on a mesh and infer on a point cloud, or mix meshes and point clouds in the training set.
- It is _robust_. DiffusionNet avoids potentially-brittle geometric operations, and does not impose any assumptions such as manifoldness, etc.
- It is _data efficient_. DiffusionNet can learn from 10s of models, without any augmentation policies needed.

See the [Tips and Tricks]() section below for a more in-depth discussion of the properties of DiffusionNet, and whether it makes sense for you problem or not.

DiffusionNet is described in the paper ["DiffusionNet: Discretization Agnostic Learning on Surfaces"](https://arxiv.org/abs/2012.00888), by 
- [Nicholas Sharp](https://nmwsharp.com/)
- Souhaib Attaiki
- [Keenan Crane](http://keenan.is/here)
- [Maks Ovsjanikov](http://www.lix.polytechnique.fr/~maks/)

**NOTE:** the linked paper is a _preprint_, and this code should be viewed similarly. 

![network diagram](https://github.com/nmwsharp/diffusion-net/blob/master/media/diagram.jpg)


## Files outline

  - `README.md` This file.
  - `diffusion_net/utils.py` Utilities and helper functions used by the code.
  - `diffusion_net/geometry.py` Core geometric routines, mainly to build the Laplacian and gradient matrices, as well as computing the corresponding spectral basis. Includes a caching mechanism.
  - `diffusion_net/layers.py` The implemention of the DiffusionNetBlock, including pointwise MLPs, learned diffusion, and learned gradient features.
  - `environment.yml` A conda environment file which can be used to install packages.


## Prerequisites

DiffusionNet depends on pytorch, as well as a handful of other fairly typical numerical packages. These can be installed manually without too much trouble, but alternately a conda environment file is provided with known-working package versions (see conda documentation for additional instructions).

```
conda env create -f environment.yml
```

The code assumes a GPU with CUDA support. DiffusionNet has minimal memory requirements; >4GB GPU memory should be sufficient. 

### Tips and Tricks
TODO

Some possible "gotcha"s:
- DiffusionNet (with _spectral acceleration_, the usual mode) performs some precomputation on the CPU for each input. It only takes a few seconds, but this can be a significant runtime hit in a setting where every input is a different shape (such as applying the model . However, this can be worked around in some cases, such as the case where all inputs are deformations of a template (see below).

## Thanks
Snippets of code for dataset loaders come from [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric), [HSN](https://github.com/rubenwiersma/hsn), and probably other sources too. Thank you!

## Human mesh segmentation example

We include machinery to run one example from the paper, segmenting meshes of humans.

Our dataloader bootstraps off the dataloader graciously provided by the authors of HSN at https://github.com/rubenwiersma/hsn. The corresponding dataset can also be downloaded from the link in that repository: https://surfdrive.surf.nl/files/index.php/s/L68uSYpHtfO6dLa.

- Install dependencies (see above).
- Place the dataset in `data/seg/raw/`, like `data/human_seg/raw/ShapeSeg/Adobe/`, etc. The dataset can be obtained from the link above; note that the files you need are nested within the downloaded archive.
- Call `python src/run_human_seg.py` to train the model. Note that on the first pass through the code will precompute the necessary operators and store them in `cache/*`.
