This experiment classifies meshes from the SHREC2011 dataset in to 30 categories ('ant', 'pliers', 'laptop', etc). The dataset contains 20 meshes from each category, for a total of 600 inputs. The variants of each mesh are nonrigid deformed versions of one another, which makes intrinsic surface-based methods highly effective.

As with past work, we use this dataset to test the effectiveness of our model with very small amounts of training data, and train on just 10 inputs per class, selected via random split. DiffusionNet gets nearly perfect accuracy, without any data augmentation (except random rotations to learn a rigid-invariant model when positional features are used as input).

The original dataset contained meshes of about 10,000 vertices, with imperfect mesh quality (some degenerate faces, etc). In the MeshCNN paper, these were simplified to high-quality meshes of <1000 vertices, which have been widely used in subsequent work. DiffusionNet is tested on both variants of the dataset, with similar results on each but a small improvement on the original high-resolution data. This repositiory has code and instructions for running on either dataset.

### Data

  The **original SHREC11 models** were previously distributed via NIST [here](https://www.nist.gov/itl/iad/shrec-2011-datasets), but that page seems to have been lost to the sands of time. We provide a zip of the old dataset page here: https://drive.google.com/uc?export=download&id=1O_P03aAxhjCOKQH2n71j013-EfSmEp5e. The relevant files are inside that archive, in the `SHREC11_test_database_new.zip` file, which is password protected with the password `SHREC11@NIST`. We also include the `data/original/categories.txt` file in this repositiory, giving ground truth labels.

  ```sh
  unzip SHREC2011_NonRigid.zip 
  unzip -P SHREC11@NIST NonRigid/SHREC11_test_database_new.zip -d data/original/raw
  ```

  The **simplified models** from MeshCNN can be downloaded here (link from the MeshCNN authors): https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz. Despite the filename, this really is the shapes from the SHREC 2011 dataset. Extract it to the `data/simplified/raw/` directory.

  ```sh
  tar -xf shrec_16.tar.gz -C data/simplified/raw
  ```

### Training from scratch

On each training run, we generate a random train/test split with 10 training inputs per-class.

To train the models on the **original** SHREC meshes, use

```python
python classification_shrec11.py --dataset_type=original --input_features=hks
```
or, with positional coordinates as features
```python
python classification_shrec11.py --dataset_type=original --input_features=xyz
```

And likewise, to train on the simplified meshes

```python
python classification_shrec11.py --dataset_type=simplified --input_features=hks
python classification_shrec11.py --dataset_type=simplified --input_features=xyz
```

There will be variance in the final accuracy, because the networks generally predict just 0-3 test models incorrectly, and the test split is randomized. Perform multiple runs to get a good sample!

**Note:** This experiment is configured to generate a random test/train split on each run. For this reason, no evaluation mode or pretrained models are included to avoid potential mistakes of testing on a set which overlaps with the train set which was used for the model.
