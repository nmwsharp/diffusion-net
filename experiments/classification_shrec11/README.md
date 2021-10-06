### Data

  The **original SHREC11 models** were previously distributed via NIST [here](https://www.nist.gov/itl/iad/shrec-2011-datasets), but that page seems to have been lost to the sands of time. We provide a zip of the old dataset page here: https://drive.google.com/uc?export=download&id=1O_P03aAxhjCOKQH2n71j013-EfSmEp5e. The relevant files are in `SHREC11_test_database_new.zip`, which is password protected with the password `SHREC11@NIST`. 

  ```sh
  unzip -P SHREC11@NIST SHREC11_test_database_new.zip -d data/orig/raw
  ```


  The **simplified models** from MeshCNN can be downloaded here (link from the MeshCNN authors): https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz

  - Download the human segmentation dataset of "Convolutional Neural Networks on Surfaces via Seamless Toric Covers" by Maron et. al. 2017 from here (link by the original authors): https://www.dropbox.com/sh/cnyccu3vtuhq1ii/AADgGIN6rKbvWzv0Sh-Kr417a?dl=0
  - Unzip it in to the `data` subdirectory of this folder like (e.g. on unix run `unzip human_benchmark_sig_17.zip -d data/`)

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

