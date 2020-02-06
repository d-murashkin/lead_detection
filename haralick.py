#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This module implements efficient algorithm for Haralick Texture Features (TF) computation
    with the use of a C function built in python code which is executed with scipy.weave
    
    haralick - function for TF computation.
    
    save_haralick_features - function that saves calculated features as numpy binary data, preview
        of features and full resolution image if needed.
        
    features_glr_direct - reduces amount of gray levels directly by multiplication and float to int
        conversion; calculate TF using haralick function.
        
    features_glr_kmeans - reduces amount of gray levels using kmeans algorythm in 1D space;
        calculate TF using haralick function.
        
    calculate_features - calculates TF for all products from given folder and saves them.
    
    @author: Dmitrii Murashkin
"""

import numpy as np
try:
    from scipy import weave
except:
    import weave
from scipy import cluster
from os import path
from os import mkdir
import time


def haralick(img, window_size=9, n_levels=16, d=1, nthreads=1, result=None, step=1):
    """
        The function calculates haralick texture features for given image.
        
        It returns np.array of 7 features and the image inself in the following order:
        ASM, entropy, contrast, homogenity, MAX, correlation, cluster tendency, image itself.
        The features mentioned can have 'a bit' different definitions in literature.
        4 GLC Matrixes are calculated, each feature then is computed for sum of the Matrixes.
        
        input:
        
        img - image, must be a 2D matrix of int8 values;
        
        window_size must be odd;
        
        n_levels - amount of brightness levels (max(img) must be less than n_levels);
        
        d - displacement;
        
        nthreads - number of threads for openMP to use, must be integer;
        
        result - buffer for result, should be 3d array size of (8, X, Y), where X, Y = img.shape
        
        tp - 'mean' by default and this is the only option since 'max' increased noise,
            'sqrt' is not very usefull // tp is not used anymore!
        
        output:
        
        normalized values, np.float32 array of (7, img.X, img.Y) size
    """
    
    """ Check that maximal brightness is less than number of gray levels """
    if img.max() > n_levels:
        print (img.max() > n_levels)
        return 0
        
    X, Y = img.shape
    ws = window_size / 2
    Xx, Yy = img[::step, ::step].shape
    size = Xx * Yy
    ret = False
    if result is None:
        result = np.zeros((14, Xx, Yy), np.float32)
        ret = True
   
    code_C_mean = \
    """
    #include <math.h>
    #include <stdio.h>
    #include <string.h>
    
    double ch[13];
    double tmp[6];
    double glcm[n_levels][n_levels];
    double pxy[n_levels*2], px_y[n_levels], px[n_levels];
    double p;
//    const double mu = 1/n_levels;
    double R = 0;
//    double n_log_n[ws*ws*16];
    unsigned int ind;
//    unsigned int n_square[ws*ws*16];
    int k, l;
    int i, j, is, js;
//    int weight;
    uint16_t glcm_i[n_levels][n_levels];
    const int dY = d * Y;
    int weight[ws*2+1][ws*2+1];
//    int step = 1;
    
    // Lookup tables    
//    for (i = 0; i < ws*ws*16; i++)
//    {
//       n_log_n[i] = i * log10(i+1);
//       n_square[i] = pow(i, 2);
//    }

    // Set kernel weights
    for (k = 0; k <= 2*ws; k++)
        for (l = 0; l <= 2*ws; l++)
            {
            weight[k][l] = fmin(ws - abs(k - ws), ws - abs(l - ws)) + 1;
            //weight[k][l] = 1;
            R += weight[k][l];
            }
    
    omp_set_num_threads(nthreads);
    
    #pragma omp parallel shared(img, weight, R, step) private(i, j, is, js, glcm, k, l, ind, ch, p, glcm_i, pxy, px_y, px, tmp)
    #pragma omp for
    for (i = ws; i < X - ws; i += step)
    {
        for (j = ws; j < Y - ws; j += step)
        {
            // set GLCM and features value to be zero
            memset(glcm_i, 0, sizeof(glcm_i));
            memset(ch, 0, sizeof(ch));
            memset(pxy, 0, sizeof(pxy));
            memset(px_y, 0, sizeof(px_y));
            memset(px, 0, sizeof(px));
            memset(tmp, 0, sizeof(tmp));
            // calculate GLCM
            // main part of sliding window
            for (k = -ws; k < ws; k++)
            {
                for (l = -ws; l < ws; l++)
                {
                    ind = (i+k) * Y + (j+l);
                    glcm_i[img[ind]][img[ind + dY]] += weight[k+ws][l+ws];     // horisontal-up
                    glcm_i[img[ind]][img[ind + d]] += weight[k+ws][l+ws];    // vertical-down
                    glcm_i[img[ind]][img[ind + dY + d]] += weight[k+ws][l+ws];   // diagonal down-right  
                    glcm_i[img[ind+1]][img[ind + dY]] += weight[k+ws][l+ws];// diagonal down-left
                }
                glcm_i[img[(i+ws)*Y + (j+k)]][img[(i+ws)*Y + (j+k+1)]]++;
                glcm_i[img[(i+k)*Y + (j+ws)]][img[(i+k+1)*Y + (j+ws)]]++;
            }
            
            // Make GLCM symmetrical
            for (k = 0; k < n_levels; k++)
                for (l = 0; l <= k; l++)
                {
                    glcm_i[k][l] += glcm_i[l][k];
                    //tmp[0] += glcm_i[k][l];
                }
                
            // Normalize glcm - summ of values in the matrix should be equal to 1                                        
            // Calculate pxy, px_y, px
            // Calculate standard deviation for px (tmp[1] is variance, tmp[2] is std)
            for (k = 0; k < n_levels; k++)
            {
                for (l = 0; l < k; l++)
                {
                    //glcm[k][l] = glcm_i[k][l] / tmp[0];
                    glcm[k][l] = glcm_i[k][l] / R;
                    pxy[k + l] += 2 * glcm[k][l];
                    px_y[k - l] += 2 * glcm[k][l];
                    px[k] += glcm[k][l];
                    tmp[0] += px[k] * k;
                }
                //glcm[k][k] = glcm_i[k][k] / tmp[0];
                glcm[k][k] = glcm_i[k][k] / R;
                pxy[k + k] += glcm[k][k];
                px_y[0] += glcm[k][k];
                px[k] += glcm[k][k];
                tmp[0] += px[k] * k;
                //tmp[1] += (mu - px[k])*(mu - px[k]);
            }
            for (k = 0; k < n_levels; k++)
            {
                tmp[1] += (tmp[0] - px[k]) * (tmp[0] - px[k]);           
            }
            tmp[2] = sqrt(tmp[1]);
            
            //calculate characteristics
            for (k = 0; k < n_levels; k++)
            {
                for (l = 0; l < k; l++)
                {
                    p = glcm[k][l];
                    if (p)
                    {
                        // ASM
                        //ch[0] += 2 * n_square[p];
                        ch[0] += 2 * p * p;

                        // Entropy
                        //ch[1] += 2 * n_log_n[p];
                        ch[1] += 2 * p * log10(p + 1.);
                        
                        // Contrast
                        ch[2] += 2 * (l-k)*(l-k) * p;

                        // Sum of squares: Variance
                        //ch[3] += 2 * p * (k-mu)*(k-mu);
                        ch[3] += 2 * p * (k-tmp[0])*(k-tmp[0]);
                        
                        // Inverse Difference Moment
                        ch[5] += 2 * p / (1 + (k - l) * (k - l));
                        
                        // Correlation
                        //ch[6] += 2 * p * (mu - k) * (mu - l);
                        //ch[6] += 2 * (k*l*p - mu*mu) / (tmp[2]);
                        ch[6] += 2 * (k*l*p - tmp[0]*tmp[0]) / (tmp[2]);
                        
                        // Cluster Tendency
                        //ch[7] += 2 * p * ((k - mu) + (l - mu));
                        ch[7] += 2 * p * ((k - tmp[0]) + (l - tmp[0]));
                        // HXY1 (tmp[3]) and HXY2 (tmp[4])
                        tmp[3] += 2 * p * log10(px[k] * px[l] + 1.);
                        tmp[4] += px[k] * px[l] * log10(px[k] * px[l] + 1.);
                    }
                }
                p = glcm[k][k];
                if (p)
                {
                    // ASM
                    //ch[0] += n_square[p];
                    ch[0] += p * p;

                    // Entropy
                    //ch[1] += n_log_n[p];
                    ch[1] += p * log10(p + 1.);
                    
                    // Contrast
                    ch[2] += (l-k)*(l-k) * p;

                    // Sum of squares: Variance
                    //ch[3] += p * (k-mu)*(k-mu);
                    ch[3] += p * (k-tmp[0])*(k-tmp[0]);
                    
                    // Inverse Difference Moment (Homogeneity)
                    ch[5] += p / (1 + (k - l) * (k - l));

                    // Correlation
                    //ch[6] += p * (mu - k) * (mu - l);
                    //ch[6] += (k*l*p - mu*mu) / (tmp[2]);
                    ch[6] += (k*l*p - tmp[0]*tmp[0]) / (tmp[2]);
                    
                    // Cluster Tendency
                    //ch[7] += p * (k - mu) * (k - mu);
                    ch[7] += p * (k - tmp[0]) * (k - tmp[0]);
                    // HXY1 (tmp[3]) and HXY2 (tmp[4])
                    tmp[3] += p * log10(px[k] * px[k] + 1.);
                    tmp[4] += px[k] * px[k] * log10(px[k] * px[k] + 1.);
                }
            }
            
            for (k = 1; k < 2*n_levels - 1; k++)
            {
                // Sum Average == Cluster Tendency that is p * ((k - mu) + (l - mu))
                ch[8] += k * pxy[k];
                
                // Sum Variance
                ch[9] += (k - ch[8])*(k - ch[8]) * pxy[k];
                
                // Sum entropy
                ch[10] += pxy[k] * log10(pxy[k] + 1.);
            }
            for (k = 0; k < n_levels; k++)
            {
                // Difference variance
                //ch[11] += mu - px_y[k];
                ch[11] += tmp[0] - px_y[k];
                
                //Difference entropy
                ch[12] += px_y[k] * log10(px_y[k] + 1.);
                
                // Probability of a run of length
                //ch[7] += (px[k] - glcm[k][k])*(px[k] - glcm[k][k]) * pow(glcm[k][k], 2) / (pow(px[k], 3)+0.001);
                
                // px entroty (tmp[5])
                tmp[5] += px[k] * log10(px[k] + 1.);
            }
            // Information measure of correlation (f12 from the paper by Haralick)
            ch[4] = (-ch[1] - tmp[3]) / (tmp[5] + 1.);
            //ch[4] = tmp[5];
            //ch[7] = sqrt(1 - exp(-2.0 * (tmp[4] + ch[1])));
            //ch[7] = (tmp[4] + ch[1]);

            is = i / step;
            js = j / step;
            result[is*Yy + js] = ch[0];
            result[is*Yy + js + size] = ch[1];
            result[is*Yy + js + 2*size] = ch[2];
            result[is*Yy + js + 3*size] = ch[3];
            result[is*Yy + js + 4*size] = ch[4];
            result[is*Yy + js + 5*size] = ch[5];
            result[is*Yy + js + 6*size] = ch[6];
            //result[is*Yy + js + 7*size] = ch[7];
            result[is*Yy + js + 7*size] = ch[12];
            result[is*Yy + js + 8*size] = ch[8];
            result[is*Yy + js + 9*size] = ch[9];
            result[is*Yy + js + 10*size] = ch[10];
            result[is*Yy + js + 11*size] = ch[11];
            //result[is*Yy + js + 12*size] = ch[12];
        }
    }
    """
    weave.inline(code_C_mean, ['img', 'X', 'Y', 'n_levels', 'result', 'ws', 'd', 'nthreads', 'step', 'size', 'Yy'],
                 extra_compile_args=['-O3 -fopenmp -funroll-loops'], compiler='gcc',
                 libraries=['gomp'], headers=['<omp.h>'])

    result[-1, :, :] = img[::step, ::step]
    if ret:
        return result.astype(np.float32)
    return True
    
    
def save_haralick_features(data, directory):
    """
        Save Haralick texture features given by 'data' in numpy binary format in 'directory',
        Save image preview (5 times smaller) in png format. Do not save it if 'preview' is False
        If full_image is True, save full image in png format
    """
    if not path.isdir(directory):
        mkdir(directory)
    for n, i in enumerate(data):
        np.save(directory + 'f{0}'.format(n), i.astype(np.float32))


def features_glr_direct(data, n_levels, nthreads=1, distance=1, win_size=9, result=None):
    """
    Apply direct gray level reduction and calculate Haralick texture features
    """
    X, Y = data.shape
    ret = False
    if result is None:
        result = np.zeros((14, X, Y), np.float32)
        ret = True
    data1 = ((n_levels - 1) * data).astype(np.int8)
    data1 = haralick(data1, win_size, n_levels=n_levels, d=distance, nthreads=nthreads)
    data1 = data1.astype(np.float32)
    if ret:
        return data1
    return True

    
def features_glr_kmeans(data, n_levels, nthreads=1, distance=1, win_size=9, result=None):
    """
    Apply k-means for gray level reduction and calculate Haralick texture features
    """
    X, Y = data.shape
    ret = False
    if result is None:
        result = np.zeros((14, X, Y), np.float32)
        ret = True
    tmp = cluster.vq.kmeans2(data.ravel(), n_levels, check_finite=False)
    a = tmp[0].copy()
    a.sort()
    b = tmp[0].copy().tolist()  
    
    tmp = tmp[1].reshape((X, Y))
    data1 = np.empty((X, Y), np.int8)
    for nb, i in enumerate(a):
        data1[tmp == b.index(i)] = nb
    haralick(data1, win_size, n_levels=n_levels, d=distance, nthreads=nthreads, result=result)
    if ret:
        return data1
    return True

    
if __name__ == '__main__':
    pass
    from sentinel1 import Sentinel1Product
    p = Sentinel1Product('/exzell2/d.murashkin/S2_raw/S1A_EW_GRDM_1SDH_20160917T070354_20160917T070454_013089_014C4A_49EB.zip')
    p.read_data_p()
    data = p.HH.data[:5000, :] - p.HV.data[:5000, :]
    data -= data.min()
    data /= data.max()
    data = (15 * data).astype(np.int8)
    X, Y = data[::3, ::3].shape
    r = np.zeros((14, X, Y))
    t = time.time()
    haralick(data, nthreads=4, result=r, window_size=9, step=3)
    print time.time() - t
