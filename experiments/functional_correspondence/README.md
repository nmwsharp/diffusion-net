These experiments demonstrate the use of DiffusionNet for shape correspondence using functional maps. Here, a DiffusionNet is trained as a feature extractor, and the resulting features are used to extract a functional map encoding the correspondence. As elsewhere, DiffusionNet offers several benefits in this setting, due to discretization invariance and scalability.

![functional correspondence results](https://github.com/nmwsharp/diffusion-net/blob/master/media/func_corr_results.png)

We use two distinct datasets (FAUST and SCAPE), and show results within each dataset, as well as training on one and testing on the other.

### Data

Data can be downloaded at [this link](https://nuage.lix.polytechnique.fr/index.php/s/LJFXrsTG22wYCXx), provided by the authors of [GeomFMaps](https://github.com/LIX-shape-analysis/GeomFmaps). Please note the citations and license restrictions of the original data: [FAUST](http://faust.is.tue.mpg.de/), [SCAPE](http://ai.stanford.edu/~drago/Projects/scape/scape.html).

**FAUST**

  - [Download](https://nuage.lix.polytechnique.fr/index.php/s/LJFXrsTG22wYCXx) the `FAUST_r.zip` and `FAUST_r_vts.zip` files
  - Unzip them to the `/data` directory

```sh
unzip FAUST_r.zip -d data/faust/
unzip FAUST_r_vts.zip -d data/faust/
```


**SCAPE**
  
  - [Download](https://nuage.lix.polytechnique.fr/index.php/s/LJFXrsTG22wYCXx) the `SCAPE_r.zip` and `SCAPE_r_vts.zip` files
  - Unzip them to the `/data` directory

```sh
unzip SCAPE_r.zip -d data/scape/
unzip SCAPE_r_vts.zip -d data/scape/
```

### Training from scratch

To train the models, use

```python
python functional_correspondence.py --train_dataset=faust --input_features=xyz
```
or, with heat kernel signature features
```python
python functional_correspondence.py --train_dataset=faust --input_features=hks
```

Passing `--train_dataset=scape` trains on the SCAPE dataset instead.

After training, the fitted model will be saved in `saved_models/[dataset]_[features].pth`.

During training, geodesic error metrics are computed on the test set after each iteration; see the note below.

### Pretrained models and evaluating geodesic accuracy

> **NOTE:** Geodesic error metrics are computed via the all-pairs geodesic distance matrix, which is giant and expensive to compute! These distances will be computed and cached the first time evaluation is run, which may take several minutes per model.

Compute geodesic accuracy of a trained model by running with the `--evaluate` flag. Load any pretrained model from disk with `--load_model=PATH_TO_MODEL.pth`. In this case, the `train_dataset` argument is ignored.

```python
python functional_correspondence.py --test_dataset=faust --input_features=xyz --load_model=pretrained_models/faust_xyz.pth
```

We include several pretrained models in the `pretrained_models/` directory for both FAUST and SCAPE, as well as `xyz` and `hks` features.
