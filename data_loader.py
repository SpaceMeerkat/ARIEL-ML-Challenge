# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:59:46 2019

@author: c1307135
"""
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   

import os
import torch
import torch.utils.data as data
from astropy.io import fits

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   

class FITSCubeDataset(data.Dataset):
    def __init__(self,data_path,return_filename=None):
        self.data_path = data_path
        self.IMG_EXTENSIONS = [".fits"]
        self._images = self.make_dataset()
        self.return_filename = return_filename
        
    def is_image_file(self,filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)
        
    def make_dataset(self):
        directory = self.data_path
        images = []
        assert os.path.isdir(directory), '%s is not a valid directory' % directory
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images
        
    def default_fits_loader(self,file_name):
        _file = fits.open(file_name)
        _data = _file[1].data.astype(float)   
        _data = torch.tensor(_data)
        _data = _data.unsqueeze(0)
        _radius = float(_file[0].header['RSTAR'])
        _label = _file[2].data.astype(float)
        _training_data = (_data, _radius, _label)
        _file.close()
        if self.return_filename is not None:
            _training_data = (_data, _radius, _label, os.path.basename(file_name)[:-5])
        return _training_data

    def __getitem__(self,index): 
        _training_data = self.default_fits_loader(self._images[index])
        return _training_data 
        
    def __len__(self):
        return len(self._images)
        
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   
