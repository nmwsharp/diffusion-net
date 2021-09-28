This benchmark measures the ability of models to automatically generalize to meshes which are tessellated very differently from the training inputs (or even sampled to a point cloud). As a sample task, we use a simple notion of correspondence where each test mesh vertex is to be labelled with corresponding closest vertex index from a template mesh. We train on the ordinary FAUST template meshes, and evaluate on variant of FAUST remeshed via several different strategies.

This benchmark is described in detail here: https://github.com/nmwsharp/discretization-robust-correspondence-benchmark

![example image of data](https://github.com/nmwsharp/discretization-robust-correspondence-benchmark/blob/main/data_image.png?raw=true)

### Data

  The **training** data comes from the MPI-FAUST dataset, which can be [accessed here](http://faust.is.tue.mpg.de/).

  The **testing** data is the FAUST test meshes, remeshed according to several strategies. They can be accessed from [this repository](https://github.com/nmwsharp/discretization-robust-correspondence-benchmark).

  Quick setup: (run from this directory)
  ```sh
  mkdir data
  mkdir data/train
  unzip MPI-FAUST.zip -d data/train
  git clone git@github.com:nmwsharp/discretization-robust-correspondence-benchmark.git data/test
  ```
 
  The data should then be laid out such that the train and test meshes are located in
    - `data/train/MPI-FAUST/training/registrations/tr_reg_000.ply`
    - `data/test/data/iso/tr_reg_iso_080.ply`
  etc.

  (we do not use most of the data from the MPI-FAUST dataset, just the aligned templates)


### Training from scratch

To train the models, use

```python
python human_segmentation_original.py --input_features=xyz  
```
or, with heat kernel signature features
```python
python human_segmentation_original.py --input_features=hks  
```

Note that since we do not use a validation set on this dataset (to match prior work), and simply take the accuracy at the last epoch, there is some decent variance in the final accuracy from run to run.

### Evaluating pretrained models

Pretrained models are included in `/pretrained_models`. You can load them and evaluate on the test set like:

```python
python human_segmentation_original.py --input_features=xyz --evaluate  
```
or, with heat kernel signature features
```python
python human_segmentation_original.py --input_features=hks --evaluate  
```

