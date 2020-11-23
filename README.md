# Lead Detection Algorithm

Algorithm for sea ice lead mapping of Sentinel-1 SAR images taken over the Arctic.

Requirements:
Python packages:
* scikit-learn
* opencv
* weave
* gdal >= 2.3
* pillow
* joblib

System:
* Linux (not tested under Windows or Mac)
* 64Gb RAM (might work with 32Gb, but not tested)
* multiple core/processor systems are recommended

To avoid possible conflicts with the requires libraries it is recommended to install [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and create a new environment as described below.

Run
```bash
conda env create --name lead_detection python=2
```
And then activate the environment with
```bash
conda activate lead_detection
```
Now install required packages:
```bash
conda install -c conda-forge weave
conda install -c defaults pillow joblib scikit-learn opencv
conda install -c defaults gdal>=2.3
```


Sentinel-1 reading routines from [Sentinel-1 routines](https://github.com/d-murashkin/sentinel1_routines) are required.
Clone the repository into your work directory (a work directory can be added to pythonpath with ```conda develop <path>```).

To perform lead mapping on a single Sentinel-1 scene use ```process_single_product.py``` script.

The script has three inputs: source folder (-s <source_folder>), destination folder (-d <destination_folder>), and name of the zip-file with the scene.

For example
```python process_single_product.py -s <source_folder> -d <destination_folder> S1A_EW_GRDM_1SDH_20200101T025017_20200101T025117_030601_038174_2410.zip```

Unpacked products are also supported (at least under Linux) - just specify the *.SAFE forlder instead of the *.zip file.

Output of the script is a GeoTiff file with two bands that correspond to the two classifications described in the publication, link below.
One classification is based on HH band or band product, another one is based on band ratio.
Numbers correspond to probability(integers, in %) of a given pixel to be a lead.
To create a binary classification a threshold of 50% can be applied to the result.

For any questions contact murashkin@uni-bremen.de.
Details of the algorithm can be found in Method for detection of leads from Sentinel-1 SAR images in [Annals of Glaciology](https://www.cambridge.org/core/journals/annals-of-glaciology/article/method-for-detection-of-leads-from-sentinel1-sar-images/3FC47FE6D90A3B9021CD753DC37184B9).
