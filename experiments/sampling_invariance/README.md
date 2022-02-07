This benchmark measures the ability of models to automatically generalize to meshes which are tessellated very differently from the training inputs (or even sampled to a point cloud). As a sample task, we use a simple notion of correspondence where each test mesh vertex is to be labelled with corresponding closest vertex index from a template mesh. We train on the ordinary FAUST template meshes, and evaluate on variant of FAUST remeshed via several different strategies.

This benchmark is described in detail here: https://github.com/nmwsharp/discretization-robust-correspondence-benchmark

![example image of data](https://github.com/nmwsharp/discretization-robust-correspondence-benchmark/blob/main/data_image.png?raw=true)

### Data

  The **training** data comes from the registered templates in the MPI-FAUST dataset, which can be [accessed here](http://faust.is.tue.mpg.de/).

  The **testing** data is the FAUST test meshes, remeshed according to several strategies. They can be accessed from [this repository](https://github.com/nmwsharp/discretization-robust-correspondence-benchmark).

  Quick data download: (run from this directory)
  ```sh
  mkdir data
  mkdir data/train
  unzip MPI-FAUST.zip -d data/train
  git clone git@github.com:nmwsharp/discretization-robust-correspondence-benchmark.git data/test
  ```
 
  The data should then be laid out such that the train and test meshes are located in
    - `data/train/MPI-FAUST/training/registrations/tr_reg_000.ply`
    - `data/test/data/iso/tr_reg_iso_080.ply`
  etc. This folder structure is a bit weird, we use it because it is the default resulting from unzipping/cloning the datasets above.

  (Note that we do not use most of the data from the MPI-FAUST dataset, just the registered templates.)


### Training from scratch

To train the models, use:

```python
python sampling_invariance.py
```
which trains on the ordinary FAUST registered templates, then tests on the remeshed meshes & point cloud as described above.  The training script prints the fraction of vertices which were classified exacatly correctly; this is not a very meaningful metric, just something to give feedback during training. After the last epoch, error on the test set is reported as measured by geodesic distance along the surface between the predicted correspondence and correct correspondence, a much more meaningful metric.

### Evaluating pretrained models

Likewise, a pretrained model is included in `/pretrained_models`. You can load it and evaluate on the test set like:

```python
python sampling_invariance.py --evaluate
```
