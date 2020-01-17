# -*- coding: utf-8 -*-
"""
The script containes some functions for lead detection.
calculate_features - texture features calculation for a given product
texture_features_single_product - for parallel calculation of TF. Saves results on disk
classify - classification basen on texture features of products from a given folder
"""

import numpy as np
import os
import time
import gdal
import shutil
import joblib

from cv2 import bilateralFilter

from haralick import haralick
from sentinel1 import Sentinel1Product
            

def lead_classification(inp_fld, out_fld, product_name, leads_fileID='leads', dec=3, first_band='hh', classifier_fld=os.path.dirname(os.path.realpath(__file__)) + '/classifiers/', classifier_name='RFC', nolv=True):
    """ Function for classification of leads on a given product of Sentinel-1 SAR data """
    t_start = time.time()
    """ Read data """
    try:
        p = Sentinel1Product(inp_fld + product_name)
    except:
        print "Can't open product {0}.".format(product_name)
        return False
    
    p.read_data_p(keep_useless_data=False)

    """ Perform classification """
    if first_band == 'hh':
        result_product = classify_(p.HH.data, band='hh', dec=dec, inp_fld=classifier_fld, classifier=classifier_name, nolv=nolv)
    elif first_band == 'product':
        product = p.HH.data + p.HV.data
        result_product = classify_(product, band='product', dec=dec, inp_fld=classifier_fld, classifier=classifier_name, nolv=nolv)
    else:
        print "Wrong first band name: {0}. Options are 'hh' and 'product'.".format(first_band)
        return False

    ratio = p.HH.data - p.HV.data
    ratio[p.HV.data > -5.2] = 0
    result_ratio = classify_(ratio, band='ratio', dec=dec, inp_fld=classifier_fld, classifier=classifier_name, nolv=nolv)
    
    """ Create output GeoTiff file with geolocation grid points """
    if product_name.split('.')[-1] == 'zip':
        p_gdal = gdal.Open('/tmp/' + product_name[:-3] + 'SAFE')
    else:
        p_gdal = gdal.Open(inp_fld + product_name)
    driver = gdal.GetDriverByName('GTiff')
    X = p_gdal.GetRasterBand(1).XSize
    Y = p_gdal.GetRasterBand(1).YSize
    result = driver.Create(out_fld + product_name + '.tiff', X / dec + 1 if np.remainder(X, dec) else X / dec, Y / dec + 1 if np.remainder(Y, dec) else Y / dec, 2, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
    proj = p_gdal.GetGCPProjection()
    gcps = p_gdal.GetGCPs()
    for gcp in gcps:
        gcp.GCPLine /= dec
        gcp.GCPPixel /= dec

    result.SetGCPs(gcps, proj)
    band_prod = result.GetRasterBand(1)
    band_ratio = result.GetRasterBand(2)
    
    x_min = p.x_min / dec if np.remainder(p.x_min, dec) else p.x_min / dec + 1
    x_max = p.x_max / dec if np.remainder(p.x_max, dec) else p.x_max / dec + 1

    """ Create output array for data to be written in the GeoTiff file """
    data_to_write_prod = np.zeros((band_prod.YSize, band_prod.XSize), dtype=np.int8)
    data_to_write_prod[:, x_min:x_min + result_product.shape[1]] = (result_product * 100).astype(np.int8)
    data_to_write_ratio = np.zeros((band_ratio.YSize, band_ratio.XSize), dtype=np.int8)
    data_to_write_ratio[:, x_min:x_min + result_ratio.shape[1]] = (result_ratio * 100).astype(np.int8)
                 
    no_data_val = 101
    data_to_write_prod[0, :] = data_to_write_prod[-1, :] = data_to_write_prod[:, 0] = data_to_write_prod[:, -1] = no_data_val
    data_to_write_prod[:, :x_min + 1] = no_data_val
    data_to_write_prod[:, x_max:] = no_data_val
    data_to_write_prod[p.HH.nodata_mask[::dec, ::dec]] = no_data_val
    data_to_write_ratio[0, :] = data_to_write_ratio[-1, :] = data_to_write_ratio[:, 0] = data_to_write_ratio[:, -1] = no_data_val
    data_to_write_ratio[:, :x_min + 1] = no_data_val
    data_to_write_ratio[:, x_max:] = no_data_val
    data_to_write_ratio[p.HH.nodata_mask[::dec, ::dec]] = no_data_val
    
    """ Write results in the GeoTiff file and flush it on hard drive """
    band_prod.WriteArray(data_to_write_prod)
    band_ratio.WriteArray(data_to_write_ratio)
    result.FlushCache()
                 
    """ Remove temporal data created when data is unzipped and read """
    if os.path.isdir('/tmp/' + product_name[:-3] + 'SAFE'):
        shutil.rmtree('/tmp/' + product_name[:-3] + 'SAFE')
    
    print 'Current product is processed in {0} sec.'.format(time.time() - t_start)
    return time.time() - t_start
    
    
def classify_(data, band, dec=1, inp_fld='out/', classifier='RFC', nolv=True):
    """ Implementation of the main classification algorithm """
    thresholds = np.array([((-6., -1.),
                            (-7.5, -4.5),
                            (-14., -4.),
                            (0., 6.)),
                           ((-3., 2.),
                            (-2., 2.),
                            (-3., 3.),
                            (-3., 3.))])

    X, Y = data[::dec, ::dec].shape

    clf = joblib.load(inp_fld + classifier + '_' + band + '.pkl')
    if band == 'product':
        [bmin, bmax], [nmin, nmax] = thresholds[:, 2, :]
        if nolv:
            feature_elimination_list = [0, 1, 5, 7, 4, 6, 2, 10, 12, 3, 11, 9, 8][:6]
        else:
            feature_elimination_list = [5, 19, 10, 23, 1, 14, 20, 0, 18, 16, 7, 4, 2, 6, 24, 17, 13, 22, 15, 21, 3, 11, 8, 9, 12][:16]
    elif band == 'ratio':
        [bmin, bmax], [nmin, nmax] = thresholds[:, 3, :]
        if nolv:
            feature_elimination_list = [0, 5, 10, 7, 2, 1, 4, 6, 3, 11, 8, 9, 12][:5]
        else:
            feature_elimination_list = [14, 18, 13, 5, 17, 23, 0, 20, 1, 7, 4, 19, 6, 22, 21, 3, 16, 24, 10, 2, 11, 15, 12, 8, 9][:17]
    elif band == 'hh':
        [bmin, bmax], [nmin, nmax] = thresholds[:, 0, :]
        if nolv:
            feature_elimination_list = [1, 5, 10, 0, 7, 4, 6, 2, 3, 11, 8, 9, 12][:6]
        else:
            feature_elimination_list = [1, 0, 23, 10, 5, 14, 16, 13, 7, 20, 19, 4, 17, 15, 2, 24, 21, 6, 18, 22, 3, 11, 8, 9, 12][:16]

    if not nolv:
        local_variations = data - bilateralFilter(data, 25, 15, 15)
        clip_normalize(local_variations, nmin, nmax)
        local_variations = (15 * local_variations).astype(np.uint8)
        TF_local_variations = np.zeros((13, X, Y))
        haralick(local_variations, nthreads=16, result=TF_local_variations, step=dec)
    
    data = bilateralFilter(data, 5, 15, 15)
    clip_normalize(data, bmin, bmax)
    data = (15 * data).astype(np.uint8)
    TF_data = np.zeros((13, X, Y))
    haralick(data, nthreads=16, result=TF_data, step=dec)

    if not nolv:
        TF = np.hstack([TF_data.reshape((13, X * Y)).T, TF_local_variations[:-1].reshape((12, X * Y)).T])
    else:
        TF = TF_data.reshape((13, X * Y)).T

    TF = np.delete(TF, feature_elimination_list, 1)
    TF_local_variations, TF_data = None, None
    result = clf.predict_proba(TF)
    return result[:, 1].reshape(X, Y)


def clip_normalize(x, xmin, xmax):
    x -= xmin
    x /= (xmax - xmin)
    x[x > 1] = 1
    x[x < 0] = 0


if __name__ == '__main__':
    pass
