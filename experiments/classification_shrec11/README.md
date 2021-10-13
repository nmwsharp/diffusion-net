### Data

  The **original SHREC11 models** were previously distributed via NIST [here](https://www.nist.gov/itl/iad/shrec-2011-datasets), but that page seems to have been lost to the sands of time. We provide a zip of the old dataset page here: https://drive.google.com/uc?export=download&id=1O_P03aAxhjCOKQH2n71j013-EfSmEp5e. The relevant files are in `SHREC11_test_database_new.zip`, which is password protected with the password `SHREC11@NIST`. 

  ```sh
  unzip -P SHREC11@NIST SHREC11_test_database_new.zip -d data/orig/raw
  ```

  The **simplified models** from MeshCNN can be downloaded here (link from the MeshCNN authors): https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz. Note that despite the filename, this really is the shapes from the SHREC 2011 dataset. Extract it to the `data/simplified/raw/` directory.

  ```sh
  tar -xf shrec_16.tar.gz -C data/simplified/raw
  ```

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

**Note:** This experiment is configured to generate a random test/train split on each run. For this reason, no evaluation mode or pretrained models are provided, to avoid potential errors of testing on a set which overlaps with the train set which was used for the model.
