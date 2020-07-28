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

from lead_detection import lead_classification as process_single_product

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate TF for a single Sentinel-1 product.\
                                        You might want to take a look at the file description.')
    parser.add_argument('-s', help='sourse folder')
    parser.add_argument('-d', help='destination folder')
    parser.add_argument('filename', help='product file name')
    args = parser.parse_args()
    
    if not args.s:
        print 'Please, specify source folder using -s key.'
        sys.exit()
    if not args.d:
        print 'Please, specify destination folder using -d key.'
        sys.exit()
    if not args.filename:
        print 'Please, specify product name.'
        sys.exit()

    t = time.time()

    process_single_product(inp_fld=args.s, out_fld=args.d, product_name=args.filename,
                           dec=2, first_band='hh', nolv=True, classifier_name='RFC_nolv')
    print 'Product {0} is processed in {1} sec.'.format(args.filename, time.time() - t)
