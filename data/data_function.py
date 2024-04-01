import os
import sys
import json
import torch
from glob import glob
import logging
from utils.file_io import read_img, read_disp


def filelines_split(line):
    splits = line.split()
    return splits

def load_left_right_gtdisp_path(splits):
    left_img, right_img = splits[:2]
    gt_disp = None if len(splits) == 2 else splits[2]
    return left_img, right_img, gt_disp

def build_sample(save_filename, data_dir, left_img, right_img, gt_disp, load_pseudo_gt):
    sample = dict()
    if save_filename:
        sample['left_name'] = left_img.split('/', 1)[1]
        #sample = {'left_name': 'image_2/000199_10.png'}
    sample['left'] = os.path.join(data_dir, left_img)
    sample['right'] = os.path.join(data_dir, right_img)
    sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None
    if load_pseudo_gt and sample['disp'] is not None:
        # KITTI 2015
        if 'disp_occ_0' in sample['disp']:
            sample['pseudo_disp'] = (sample['disp']).replace('disp_occ_0',
                                                             'disp_occ_0_pseudo_gt')
        #KITTI 2012
        elif 'disp_occ' in sample['disp']:
            sample['pseudo_disp'] = (sample['disp']).replace('disp_occ',
                                                             'disp_occ_pseudo_gt')
        else:
            raise NotImplementedError
    else:
        sample['pseudo_disp'] = None
    return sample

def path_to_array(save_filename, samples, index, dataset_name):
    
    sample = {}
    sample_path = samples[index]
    
    if  save_filename:
        sample['left_name'] = sample_path['left_name']

    sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
    sample['right'] = read_img(sample_path['right'])

    # GT disparity of subset if negative, finalpass and cleanpass is positive
    subset = True if 'subset' in dataset_name else False
    if sample_path['disp'] is not None:
        sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
    if sample_path['pseudo_disp'] is not None:
        sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

    return sample