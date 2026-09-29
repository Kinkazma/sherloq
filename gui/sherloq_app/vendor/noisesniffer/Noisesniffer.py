#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:24:58 2021

@author: marina
"""

import argparse
import numpy as np
from skimage.util import view_as_windows
import functions



def do_one_channel(ch, n, m, img, w, img_means, valid_blocks_indices, b):
    
    V = []
    S = []
    
    # extract image blocks in channel ch
    img_blocks = view_as_windows(img[:,:,ch], w).reshape(-1, w, w)

    # compute low freq variance of the blocks
    low_freq_var = functions.compute_low_freq_var(img_blocks, w).reshape(-1)
    
    # sort valid blocks according to their mean
    blocks_sorted_means = functions.sort_blocks_means(ch, img_means, 
                                                      valid_blocks_indices)
    
    # update the number of samples per bin
    b = functions.update_samples_per_bin(b, len(valid_blocks_indices))
    
    # compute number of bins 
    num_bins = int(round(len(valid_blocks_indices) / b))

    for Bin in range(num_bins):
        
        # list the blocks in the bin
        blocks_in_bin = functions.bin_block_list(num_bins, Bin, 
                                                 blocks_sorted_means, b)
        
        # select blocks according to their variance in low freqs
        blocks_select_LF_var = functions.select_blocks_VL(b, n, low_freq_var, 
                                                          blocks_in_bin)
        
        # sort selected blocks according to their std
        blocks_stds_sorted = functions.std_blocks(img_blocks, 
                                                  blocks_select_LF_var)
        
        if functions.bin_is_valid(b, n, m, blocks_stds_sorted):
            for k, pos in enumerate(blocks_stds_sorted):             
                V.append(pos)
                if k < int(b * n * m): 
                    S.append(pos)
                    
    return V, S

def do_one_image(f, w, W, b, n, m):
    
    # read image
    img = functions.read_image(f)
    
    # compute list of valid blocks
    valid_blocks_indices = functions.compute_valid_blocks_indices(img,w) 
    
    # compute means of blocks
    img_means = functions.all_image_means(img, w)
    
    
    V, S = [], []
    
    for ch in range(img.shape[2]):
        V_ch, S_ch = do_one_channel(ch, n, m, img, w, img_means,
                                    valid_blocks_indices, b)
        V = np.concatenate((V, V_ch))
        S = np.concatenate((S, S_ch))

    functions.compute_output(img, w, W, m, V, S)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-w")
    parser.add_argument("-W")
    parser.add_argument("-b")
    parser.add_argument("-n")
    parser.add_argument("-m")
    parser.parse_args()
    args = parser.parse_args()

    f = args.filename
    w = int(args.w)
    W = int(args.W)
    b = int(args.b)
    n = float(args.n)
    m = float(args.m)
    do_one_image(f,w, W, b, n, m)
