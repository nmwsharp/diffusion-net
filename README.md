**DiffusionNet** is a general-purpose method for deep learning on surfaces such as 3D triangle meshes and point clouds. It is well-suited for tasks like segmentation, classification, feature extraction, etc.

Why try DiffusionNet?
- It is _efficient_ and _scalable_. On a single GPU, we can easily train on meshes of 20k vertices, and infer on meshes with 200k vertices. One-time preprocessing takes a few seconds in the former case, and about a minute in the latter.
- It is _sampling agnostic_. Many graph-based mesh learning approaches tend to overfit to mesh connectivity, and can output nonsense when you run them on meshes that are triangulated differently from the training set. With DiffusionNet we can intermingle different triangulations and very coarse or fine meshes without issue. No special regularization or data augmentation needed!
- It is _representation agnostic_. For instance, you can train on a mesh and infer on a point cloud, or mix meshes and point clouds in the training set.
- It is _robust_. DiffusionNet avoids potentially-brittle geometric operations, and does not impose any assumptions such as manifoldness, etc.
- It is _data efficient_. DiffusionNet can learn from 10s of models, even without any data augmentation.

DiffusionNet is described in the paper ["DiffusionNet: Discretization Agnostic Learning on Surfaces"](https://arxiv.org/abs/2012.00888), by 
- [Nicholas Sharp](https://nmwsharp.com/)
- Souhaib Attaiki
- [Keenan Crane](http://keenan.is/here)
- [Maks Ovsjanikov](http://www.lix.polytechnique.fr/~maks/)

![network diagram](https://github.com/nmwsharp/diffusion-net/blob/master/media/diagram.jpg)

## Outline

  - `diffusion_net/src` implementation of the method, including preprocessing, layers, etc
  - `experiments` examples and scripts to reproduce experiments from the DiffusionNet paper
  - `environment.yml` A conda environment file which can be used to install packages.


## Prerequisites

DiffusionNet depends on pytorch, as well as a handful of other fairly typical numerical packages. These can usually be installed manually without much trouble, but alternately a conda environment file is also provided (see conda documentation for additional instructions). These package versions were tested with CUDA 10.1 and 11.1. 

```
conda env create --name diffusion_net -f environment.yml
```

The code assumes a GPU with CUDA support. DiffusionNet has minimal memory requirements; >4GB GPU memory should be sufficient. 

## Applying DiffusionNet to your task

The `DiffusionNet` class can be applied to meshes or point clouds. The basic recipe looks like:

```python
import diffusion_net

# Here we use Nx3 positions as features. Any other features you might have will work!
# See our experiments for the use of of HKS features, which are naturally 
# invariant to (isometric) deformations.
C_in = 3

# Output dimension (e.g., for a 10-class segmentation problem)
C_out = 10 

# Create the model
model = diffusion_net.layers.DiffusionNet(
            C_in=C_in,
            C_out=n_class,
            C_width=128, # internal size of the diffusion net. 32 -- 512 is a reasonable range
            last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1), # apply a last softmax to outputs 
                                                                                  # (set to default None to output general values in R^{N x C_out})
            outputs_at='vertices')

# An example epoch loop.
# For a dataloader example see experiments/human_segmentation_original/human_segmentation_original_dataset.py
for sample in your_dataset:
    
    verts = sample.vertices  # (Vx3 array of vertices)
    faces = sample.faces     # (Fx3 array of faces, None for point cloud) 
    
    # center and unit scale
    verts = diffusion_net.geometry.normalize_positions(verts)
    
    # Get the geometric operators needed to evaluate DiffusionNet. This routine 
    # automatically populates a cache, precomputing only if needed.
    # TIP: Do this once in a dataloader and store in memory to further improve 
    # performance; see examples.
    frames, mass, L, evals, evecs, gradX, gradY = \
        get_operators(verts, faces, op_cache_dir='my/cache/directory/')
    
    # this example uses vertex positions as features 
    features = verts
    
    # Forward-evaluate the model
    # preds is a NxC_out array of values
    outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
    
    # Now do whatever you want! Apply your favorite loss function, 
    # backpropgate with loss.backward() to train the DiffusionNet, etc. 
```

See the examples in `experiments/` for complete examples, including dataloaders, other features, optimizers, etc. Please feel free to file an issue to discuss applying DiffusionNet to your problem!

### Tips and Tricks

By default, DiffusionNet uses _spectral acceleration_ for fast performance, which requires some CPU-based precomputation to compute operators & eigendecompositions for each input, which can take a few seconds for moderately sized inputs. DiffusionNet will be fastest if this precomputation only needs to be performed once for the dataset, rather than for each input. 

- If you are learning on a **template mesh**, consider precomputing operators for the _reference pose_ of the template, but then using xyz the coordinates of the _deformed pose_ as inputs to the network. This is a slight approximation, but will make DiffusionNet very fast, since the precomputed operators are shared among all poses.
- If  you need **data augmentation**, try to apply augmentations _after_ computing operators whenever possible. For instance, in our examples, we apply random rotation to positions, but only _after_ computing operators. Note that we find common augmentations such as slightly skewing/scaling/subsampling inputs are generally unnecessary with DiffusionNet.

### Thanks

Parts of this work were generously supported by the Fields Institute for Mathematics, the Vector Institute, ERC Starting Grant No. 758800 (EXPROTEA) the ANR AI Chair AIGRETTE, a Packard Fellowship, NSF CAREER Award 1943123, an NSF Graduate Research Fellowship, and gifts from Activision Blizzard, Adobe, Disney, Facebook, and nTopology. The dataset loaders mimic code from [HSN](https://github.com/rubenwiersma/hsn), [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric), and probably indirectly from other sources too. Thank you!
