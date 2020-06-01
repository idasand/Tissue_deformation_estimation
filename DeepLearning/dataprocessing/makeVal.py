import numpy as np
import h5py
import os
import glob
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path')
args = parser.parse_args()


fixed = np.empty(shape=(410,316,1))
moving = np.empty(shape=(410,316,1))
this_img = np.empty(shape=(410,316))


f_read1 = h5py.File('../dataprocessing/paddedcycles/J5A96806_padded_onecycle.h5', 'r')
tissueSmooth1 = np.array(f_read1['tissuepaddedSmoothOneCycle'], dtype='f')

for i in range(tissueSmooth1.shape[2]-1):
    this_img[:,:] = np.float32(tissueSmooth1[:,:,i])
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth1.shape[2]):
    this_img[:,:] = np.float32(tissueSmooth1[:,:,i])
    moving = np.dstack((moving, this_img))


f_read2 = h5py.File('../dataprocessing/paddedcycles/J5M9L6O4_padded_onecycle.h5', 'r')
tissueSmooth2 = np.array(f_read2['tissuepaddedSmoothOneCycle'], dtype='f')

for i in range(tissueSmooth2.shape[2]-1):
    this_img[:,:] = np.float32(tissueSmooth2[:,:,i])
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth2.shape[2]):
    this_img[:,:] = np.float32(tissueSmooth2[:,:,i])
    moving = np.dstack((moving, this_img))


f_read3 = h5py.File('../dataprocessing/paddedcycles/J5M86MO4_padded_onecycle.h5', 'r')
tissueSmooth3 = np.array(f_read3['tissuepaddedSmoothOneCycle'], dtype='f')

for i in range(tissueSmooth3.shape[2]-1):
    this_img[:,:] = np.float32(tissueSmooth3[:,:,i])
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth3.shape[2]):
    this_img[:,:] = np.float32(tissueSmooth3[:,:,i])
    moving = np.dstack((moving, this_img))


f_read4 = h5py.File('../dataprocessing/paddedcycles/J5D85D8C_padded_onecycle.h5', 'r')
tissueSmooth4 = np.array(f_read4['tissuepaddedSmoothOneCycle'], dtype='f')

for i in range(tissueSmooth4.shape[2]-1):
    this_img[:,:] = np.float32(tissueSmooth4[:,:,i])
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth4.shape[2]):
    this_img[:,:] = np.float32(tissueSmooth4[:,:,i])
    moving = np.dstack((moving, this_img))

f_read5 = h5py.File('../dataprocessing/paddedcycles/J5MB2404_padded_onecycle.h5', 'r')
tissueSmooth5 = np.array(f_read5['tissuepaddedSmoothOneCycle'], dtype='f')

for i in range(tissueSmooth5.shape[2]-1):
    this_img[:,:] = np.float32(tissueSmooth5[:,:,i])
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth5.shape[2]):
    this_img[:,:] = np.float32(tissueSmooth5[:,:,i])
    moving = np.dstack((moving, this_img))




fixed=fixed[:,:,1:]
moving=moving[:,:,1:]
fixed=fixed[:,:,:,np.newaxis]
moving=moving[:,:,:,np.newaxis]
fixed=np.transpose(fixed,(2,1,0,3))
moving=np.transpose(moving,(2,1,0,3))

#shuffle indexes
idx = [x for x in range(fixed.shape[0])]
random.shuffle(idx)
fixed=fixed[idx,:,:,:]
moving=moving[idx,:,:,:]



hf = h5py.File('valKidney_randomsequence.h5','w')
hf.create_dataset('fixed', data=fixed, dtype='f')
hf.create_dataset('moving',data=moving, dtype='f')
hf.close()