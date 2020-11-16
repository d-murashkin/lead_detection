"""
The script processes single Sentinel-1 products.
Sourse folder, output folder and filename should be specified.

The script is made to be used with Linux parallel function, for example:
ls <sourse_folder> | parallel python single_product_TF -s <sourse_folder> -d <destination_folder> {}
or if 'parallel' is not available:
ls <sourse_folder> | xargs -n1 --max-procs=4 -I {} python single_product_TF -s <sourse_folder> -d <destination_folder> {}
will pass list of products from <sourse_folder> to parallel/xargs utility which will run several (by
default equal to number of cores) python scripts in parallel.
A mask can be applied to ls. Pipe with head and/or tail can be used as well. For instance,
ls *.zip | head -3
will return first 3 files in the directory.

Some parameters of the processing chain can be modified in this script.
These parameters can also be added as agruments to the script in the future.

@author: Dmitrii Murashkin (murashkin@uni-bremen.de)
"""

import argparse
import sys
import time
import os

from lead_detection import lead_classification as process_single_product
from sentinel1_routines.utils import scene_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate TF for a single Sentinel-1 product.\
                                        You might want to take a look at the file description.')
    parser.add_argument('-s', help='sourse folder')
    parser.add_argument('-d', help='destination folder')
    parser.add_argument('filename', help='product file name')
    parser.add_argument('--data_root', help='path to the data folder. Inside the data folder the structure should be the following: <data_folder>/<year>/<month>/<day>/PRODUCT/<scene>')
    args = parser.parse_args()
    
    if not args.filename:
        print('Please, specify product name.')
        sys.exit()
    scene_name = args.filename

    if not args.s:
        if not args.data_root:
            print('Please, specify source folder using -s key or data root folder with --data_root.')
            sys.exit()
        else:
            scn_time = scene_time
            input_folder = os.path.join(args.data_root, scn_time.strftime('%Y'), scn_time.strftime('%m'), scn_time.strftime('%d'), 'PRODUCT')
            """ Search for the exect scene name since last 4 characters might differ. """
            name = scene_name.split('.')[0][:-3]
            for item in os.listdir(input_folder):
                if name in item:
                    scene_name = item
                    break

    if not args.d:
        print('Destination folder is not specified(-d key), so the current working directory is used for output.')

    t = time.time()

    process_single_product(inp_fld=args.s, out_fld=args.d, product_name=args.filename,
                           dec=2, first_band='hh', nolv=True, classifier_name='RFC_nolv')
    print('Product {0} is processed in {1} sec.'.format(args.filename, time.time() - t))
