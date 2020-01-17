# Lead Detection Algorithm

Algorithm for sea ice lead mapping of Sentinel-1 SAR images taken over the Arctic.

Before any of the scripts is run it is highly recommended to install an [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment from the file lead_detection.yml
Run ```conda env create --name lead_detection --file environment.yml```.
And then activate the environment with
```conda activate lead_detection```.

To perform lead mapping on a single Sentinel-1 scene use process_single_product.py script.
The script has three inputs: source folder (-s <source_folder>), destination folder (-d <destination_folder>), and name of the zip-file with the scene.
For example
```python process_single_product.py -s <source_folder> -d <destination_folder> S1A_EW_GRDM_1SDH_20200101T025017_20200101T025117_030601_038174_2410.zip```
Unpacked products are also supported (at least under Linux) - specify the *.SAFE forlder instead of the *.zip file for this case.

Details of the algorithm can be found in Method for detection of leads from Sentinel-1 SAR images in [Annals of Glaciology](https://www.cambridge.org/core/journals/annals-of-glaciology/article/method-for-detection-of-leads-from-sentinel1-sar-images/3FC47FE6D90A3B9021CD753DC37184B9).
