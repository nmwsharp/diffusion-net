Source code for [Diffusion is All You Need for Learning on Surfaces](https://arxiv.org/abs/2012.00888)", by 
- [Nicholas Sharp](https://nmwsharp.com/)
- Souhaib Attaiki
- [Keenan Crane](http://keenan.is/here)
- [Maks Ovsjanikov](http://www.lix.polytechnique.fr/~maks/) at ECCV 2020.

**NOTE:** the linked paper is a _preprint_, and this code should be viewed similarly. A full code release, including experimental details, will be provided after publication.


![network diagram](https://github.com/nmwsharp/diffusion-net/blob/master/media/diagram.jpg)


## Files outline

  - `README.md` This file.
  - `src/utils.py` Utilities and helper functions used by the code.
  - `src/geometry.py` Core geometric routines, mainly to build the Laplacian and gradient matrices, as well as computing the corresponding spectral basis. Includes a caching mechanism.
  - `src/layers.py` The implemention of the DiffusionNetBlock, including pointwise MLPs, learned diffusion, and learned gradient features.
  - `src/human_seg_dataset.py` A dataset loader for the human mesh segmentation dataset.
  - `src/run_human_seg.py` A main script to fit mesh segmentations on the humans dataset.
  - `environment.yml` A conda environment file which can be used to install packages.


## Prerequisites

DiffusionNet depends on pytorch, as well as a handful of other fairly typical numerical packages. These can be installed manually without too much trouble, but alternately a conda environment file is provided with known-working package versions (see conda documentation for additional instructions).

```
conda env create -f environment.yml
```

The code assumes a GPU with CUDA support. DiffusionNet has minimal memory requirements; >4GB GPU memory should be sufficient. 


## Human mesh segmentation example

We include machinery to run one example from the paper, segmenting meshes of humans.

Our dataloader bootstraps off the dataloader graciously provided by the authors of HSN at https://github.com/rubenwiersma/hsn. The corresponding dataset can also be downloaded from the link in that repository: https://surfdrive.surf.nl/files/index.php/s/L68uSYpHtfO6dLa.

- Install dependencies (see above).
- Place the dataset in `data/seg/raw/`, like `data/human_seg/raw/ShapeSeg/Adobe/`, etc. The dataset can be obtained from the link above; note that the files you need are nested within the downloaded archive.
- Call `python src/run_human_seg.py` to train the model. Note that on the first pass through the code will precompute the necessary operators and store them in `cache/*`.
