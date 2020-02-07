# Lead Detection Algorithm

Algorithm for sea ice lead mapping of Sentinel-1 SAR images taken over the Arctic.

Requirements:
Python packages:
* scikit-learn=0.18.1
* opencv
* weave
* gdal
* pillow
* joblib

System:
* linux (not tested under Windows or Mac)
* 64Gb RAM (might work with 32Gb, but not tested)
* multiple core/processor systems are recommended

Before any of the scripts is run it is highly recommended to install an [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment from the file environment.yml.

Run
```conda env create --name lead_detection --file environment.yml```.
And then activate the environment with
```conda activate lead_detection```.


Sentinel-1 reading routines from [Sentinel-1 routines](https://github.com/d-murashkin/sentinel1_routines) are required.
Clone the repository into your work directory (a work directory can be added to pythonpath with ```conda develop <path>```.

Since files with classifiers are uploaded with Git Large File Storage, one needs to install git lfs from [Git Large File Storage](https://git-lfs.github.com/) and run
``` git lfs clone``` instead of ``` git clone ``` to clone this repository.

To perform lead mapping on a single Sentinel-1 scene use process_single_product.py script.

The script has three inputs: source folder (-s <source_folder>), destination folder (-d <destination_folder>), and name of the zip-file with the scene.

For example
```python process_single_product.py -s <source_folder> -d <destination_folder> S1A_EW_GRDM_1SDH_20200101T025017_20200101T025117_030601_038174_2410.zip```

Unpacked products are also supported (at least under Linux) - specify the *.SAFE forlder instead of the *.zip file for this case.

Output of the script is a GeoTiff file with two bands that correspond to two classifications (as described in the paper, link below).
Numbers correspond to probability of a given pixel to be a lead.
To create a binary classification a threshold of 50% can be applied to the result.

For any questions contact murashkin@uni-bremen.de.
Details of the algorithm can be found in Method for detection of leads from Sentinel-1 SAR images in [Annals of Glaciology](https://www.cambridge.org/core/journals/annals-of-glaciology/article/method-for-detection-of-leads-from-sentinel1-sar-images/3FC47FE6D90A3B9021CD753DC37184B9).
