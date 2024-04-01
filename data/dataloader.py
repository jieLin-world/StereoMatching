from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os

import data.data_function as f


class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform

        sceneflow_finalpass_dict = {
            'train': 'filenames/SceneFlow_train.txt',
            'test': 'filenames/SceneFlow_test.txt'
        }

        kitti_2012_dict = {
            'train': 'filenames/kitti12_train.txt',
            'val': 'filenames/kitti12_val.txt',
            'test': 'filenames/kitti12_test.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/kitti15_train.txt',
            'val': 'filenames/kitti15_val.txt',
            'test': 'filenames/kitti15_test.txt'
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name
        self.samples = []
        
        data_filenames = dataset_name_dict[dataset_name][mode]
        #data_filenames = kitti15_train.txt
        
        lines = f.read_text_lines(data_filenames)
        #lines = ['a.png','a.png','c.png'...]
        
        for line in lines:
            splits = line.split()
            #splits = ['a.png','a.png','c.png']
            
            left_img, right_img, gt_disp = f.load_left_right_gtdisp_path(splits)
            #left_img = training/image_2/000199_10.png
            
            sample = f.getpathlist(
                self.save_filename, 
                self.data_dir,
                left_img,
                right_img,
                gt_disp,
                load_pseudo_gt)
            
            self.samples.append(sample)
        #sample = [{'left_name':'','left':'',...},{...}...]
        
    def __getitem__(self, index):     
        sample = f.path_to_array(   self.save_filename,
                                    self.samples,
                                    index,
                                    self.dataset_name)
        
        ##########################################################################################
        sample = self.transform(sample)    
        return sample

    def __len__(self):
        return len(self.samples)
