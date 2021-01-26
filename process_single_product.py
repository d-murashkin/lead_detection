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
import shutil

from lead_detection import lead_classification as process_single_product
from sentinel1_routines.utils import scene_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate TF for a single Sentinel-1 product.\
                                        You might want to take a look at the file description.')
    parser.add_argument('-s', help='sourse folder')
    parser.add_argument('-d', help='destination folder')
    parser.add_argument('filename', help='product file name')
    parser.add_argument('--data_root', help='path to the data folder. Inside the data folder the structure should be the following: <data_folder>/<year>/<month>/<day>/PRODUCT/<scene>')
    parser.add_argument('--search_existing_results', default=None, help='If result already exists in the specified root folder, just copy it in the destination folder')
    args = parser.parse_args()
    
    if not args.filename:
        print('Please, specify product name.')
        sys.exit()
    scene_name = args.filename

    if not args.s:
        if not args.data_root:
            print('Neither source folder (-s key) nor data root folder (--data_root key) is specified.')
            print('Current working directory is used as the source folder.')
            input_folder = os.getcwd()
        else:
            scn_time = scene_time(args.filename)
            input_folder = os.path.join(args.data_root, scn_time.strftime('%Y'), scn_time.strftime('%m'), scn_time.strftime('%d'), 'PRODUCT')
            """ Search for the exect scene name since last 4 characters might differ. """
            name = scene_name.split('.')[0][:-4]
            for item in os.listdir(input_folder):
                if name in item:
                    scene_name = item
                    break
    else:
        input_folder = args.s

    if not args.d:
        print('Destination folder is not specified(-d key), so the current working directory is used for output.')
        output_folder = os.getcwd()
    else:
        output_folder = args.d

    if args.search_existing_results:
        expected_path = os.path.join(args.search_existing_results, scn_time.strftime('%Y'), scn_time.strftime('%m'), scn_time.strftime('%d'), 'single_scenes', scene_name + '.tiff')
        print(expected_path)
        if os.path.exists(expected_path):
            shutil.copyfile(expected_path, os.path.join(output_folder, scene_name + '.tiff'))
            print('Scene {0} already exists, result is copied in the destination folder'.format(scene_name))
            sys.exit()

    t = time.time()
    process_single_product(inp_fld=input_folder, out_fld=output_folder, product_name=scene_name,
                           dec=2, first_band='hh', nolv=True, classifier_name='RFC_nolv')
    print('Product {0} is processed in {1} sec.'.format(args.filename, time.time() - t))
