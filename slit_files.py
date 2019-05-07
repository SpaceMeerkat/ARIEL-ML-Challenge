# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:50:49 2019

@author: c1307135
"""
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

import os 
import shutil
from random import shuffle
from data_loader import FITSCubeDataset  

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

_load_path = '/home/anubis/c1307135/ARIEL/TRAIN_FLUX/'
_destination = '/home/anubis/c1307135/ARIEL/TEST_FLUX/'
_filenames = FITSCubeDataset(_load_path).make_dataset()
shuffle(_filenames)

def split(_filenames):
    _length = len(_filenames)
    _twenty = int(0.2*_length)
    _test = _filenames[:_twenty]
    return _test
    
def mover(_filenames,_destination):
    for _file in _filenames:
        shutil.move(_file,_destination+os.path.basename(_file))
    return
    
_test_files = split(_filenames)
mover(_test_files,_destination)

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#