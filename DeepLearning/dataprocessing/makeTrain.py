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


f_read1 = h5py.File('../dataprocessing/paddedcycles/J5AB4H04_padded_onecycle.h5', 'r')
tissueSmooth1 = np.array(f_read1['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth1.shape[2]-1):
    this_img[:,:] = tissueSmooth1[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth1.shape[2]):
    this_img[:,:] = tissueSmooth1[:,:,i]
    moving = np.dstack((moving, this_img))


f_read2 = h5py.File('../dataprocessing/paddedcycles/J5ABD204_padded_onecycle.h5', 'r')
tissueSmooth2 = np.array(f_read2['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth2.shape[2]-1):
    this_img[:,:] = tissueSmooth2[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth2.shape[2]):
    this_img[:,:] = tissueSmooth2[:,:,i]
    moving = np.dstack((moving, this_img))


f_read3 = h5py.File('../dataprocessing/paddedcycles/J5AAQE84_padded_onecycle.h5', 'r')
tissueSmooth3 = np.array(f_read3['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth3.shape[2]-1):
    this_img[:,:] = tissueSmooth3[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth3.shape[2]):
    this_img[:,:] = tissueSmooth3[:,:,i]
    moving = np.dstack((moving, this_img))


f_read4 = h5py.File('../dataprocessing/paddedcycles/J5D959G4_padded_onecycle.h5', 'r')
tissueSmooth4 = np.array(f_read4['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth4.shape[2]-1):
    this_img[:,:] = tissueSmooth4[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth4.shape[2]):
    this_img[:,:] = tissueSmooth4[:,:,i]
    moving = np.dstack((moving, this_img))


f_read5 = h5py.File('../dataprocessing/paddedcycles/J5D9EDO6_padded_onecycle.h5', 'r')
tissueSmooth5 = np.array(f_read5['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth5.shape[2]-1):
    this_img[:,:] = tissueSmooth5[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth5.shape[2]):
    this_img[:,:] = tissueSmooth5[:,:,i]
    moving = np.dstack((moving, this_img))


f_read6 = h5py.File('../dataprocessing/paddedcycles/J5D9OLG6_padded_onecycle.h5', 'r')
tissueSmooth6 = np.array(f_read6['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth6.shape[2]-1):
    this_img[:,:] = tissueSmooth6[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth6.shape[2]):
    this_img[:,:] = tissueSmooth6[:,:,i]
    moving = np.dstack((moving, this_img))


f_read7 = h5py.File('../dataprocessing/paddedcycles/J5DA4KG4_padded_onecycle.h5', 'r')
tissueSmooth7 = np.array(f_read7['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth7.shape[2]-1):
    this_img[:,:] = tissueSmooth7[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth7.shape[2]):
    this_img[:,:] = tissueSmooth7[:,:,i]
    moving = np.dstack((moving, this_img))


f_read8 = h5py.File('../dataprocessing/paddedcycles/J5DAET04_padded_onecycle.h5', 'r')
tissueSmooth8 = np.array(f_read8['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth8.shape[2]-1):
    this_img[:,:] = tissueSmooth8[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth8.shape[2]):
    this_img[:,:] = tissueSmooth8[:,:,i]
    moving = np.dstack((moving, this_img))


f_read9 = h5py.File('../dataprocessing/paddedcycles/J5DAMOG4_padded_onecycle.h5', 'r')
tissueSmooth9 = np.array(f_read9['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth9.shape[2]-1):
    this_img[:,:] = tissueSmooth9[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth9.shape[2]):
    this_img[:,:] = tissueSmooth9[:,:,i]
    moving = np.dstack((moving, this_img))


f_read10 = h5py.File('../dataprocessing/paddedcycles/J5DB35O6_padded_onecycle.h5', 'r')
tissueSmooth10 = np.array(f_read10['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth10.shape[2]-1):
    this_img[:,:] = tissueSmooth10[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth10.shape[2]):
    this_img[:,:] = tissueSmooth10[:,:,i]
    moving = np.dstack((moving, this_img))


f_read11 = h5py.File('../dataprocessing/paddedcycles/J5DBGAO6_padded_onecycle.h5', 'r')
tissueSmooth11 = np.array(f_read11['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth11.shape[2]-1):
    this_img[:,:] = tissueSmooth11[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth11.shape[2]):
    this_img[:,:] = tissueSmooth11[:,:,i]
    moving = np.dstack((moving, this_img))


f_read12 = h5py.File('../dataprocessing/paddedcycles/J5AAFTG4_padded_onecycle.h5', 'r')
tissueSmooth12 = np.array(f_read12['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth12.shape[2]-1):
    this_img[:,:] = tissueSmooth12[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth12.shape[2]):
    this_img[:,:] = tissueSmooth12[:,:,i]
    moving = np.dstack((moving, this_img))


f_read13 = h5py.File('../dataprocessing/paddedcycles/J5M8GI04_padded_onecycle.h5', 'r')
tissueSmooth13 = np.array(f_read13['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth13.shape[2]-1):
    this_img[:,:] = tissueSmooth13[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth13.shape[2]):
    this_img[:,:] = tissueSmooth13[:,:,i]
    moving = np.dstack((moving, this_img))

f_read14 = h5py.File('../dataprocessing/paddedcycles/J5M9CJ04_padded_onecycle.h5', 'r')
tissueSmooth14 = np.array(f_read14['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth14.shape[2]-1):
    this_img[:,:] = tissueSmooth14[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth14.shape[2]):
    this_img[:,:] = tissueSmooth14[:,:,i]
    moving = np.dstack((moving, this_img))


f_read15 = h5py.File('../dataprocessing/paddedcycles/J5MA59O4_padded_onecycle.h5', 'r')
tissueSmooth15 = np.array(f_read15['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth15.shape[2]-1):
    this_img[:,:] = tissueSmooth15[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth15.shape[2]):
    this_img[:,:] = tissueSmooth15[:,:,i]
    moving = np.dstack((moving, this_img))


f_read16 = h5py.File('../dataprocessing/paddedcycles/J5MACN04_padded_onecycle.h5', 'r')
tissueSmooth16 = np.array(f_read16['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth16.shape[2]-1):
    this_img[:,:] = tissueSmooth16[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth16.shape[2]):
    this_img[:,:] = tissueSmooth16[:,:,i]
    moving = np.dstack((moving, this_img))


f_read17 = h5py.File('../dataprocessing/paddedcycles/J5MAOGG4_padded_onecycle.h5', 'r')
tissueSmooth17 = np.array(f_read17['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth17.shape[2]-1):
    this_img[:,:] = tissueSmooth17[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth17.shape[2]):
    this_img[:,:] = tissueSmooth17[:,:,i]
    moving = np.dstack((moving, this_img))


f_read18 = h5py.File('../dataprocessing/paddedcycles/J5MB7J84_padded_onecycle.h5', 'r')
tissueSmooth18 = np.array(f_read18['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth18.shape[2]-1):
    this_img[:,:] = tissueSmooth18[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth18.shape[2]):
    this_img[:,:] = tissueSmooth18[:,:,i]
    moving = np.dstack((moving, this_img))

f_read19 = h5py.File('../dataprocessing/paddedcycles/J5AA5806_padded_onecycle.h5', 'r')
tissueSmooth19 = np.array(f_read19['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth19.shape[2]-1):
    this_img[:,:] = tissueSmooth19[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth19.shape[2]):
    this_img[:,:] = tissueSmooth19[:,:,i]
    moving = np.dstack((moving, this_img))

f_read20 = h5py.File('../dataprocessing/paddedcycles/J5D7PGOC_padded_onecycle.h5', 'r')
tissueSmooth20 = np.array(f_read20['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth20.shape[2]-1):
    this_img[:,:] = tissueSmooth20[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth20.shape[2]):
    this_img[:,:] = tissueSmooth20[:,:,i]
    moving = np.dstack((moving, this_img))

f_read21 = h5py.File('../dataprocessing/paddedcycles/J5D8PK06_padded_onecycle.h5', 'r')
tissueSmooth21 = np.array(f_read21['tissuepaddedSmoothOneCycle'])

for i in range(tissueSmooth21.shape[2]-1):
    this_img[:,:] = tissueSmooth21[:,:,i]
    fixed = np.dstack((fixed, this_img))
for i in range(1,tissueSmooth21.shape[2]):
    this_img[:,:] = tissueSmooth21[:,:,i]
    moving = np.dstack((moving, this_img))





fixed=fixed[:,:,1:]
moving=moving[:,:,1:]
fixed=fixed[:,:,:,np.newaxis]
moving=moving[:,:,:,np.newaxis]
fixed=np.transpose(fixed,(2,1,0,3))
moving=np.transpose(moving,(2,1,0,3))
print(fixed.shape)
print(moving.shape)

#shuffle indexes
idx = [x for x in range(fixed.shape[0])]
random.shuffle(idx)
fixed=fixed[idx,:,:,:]
moving=moving[idx,:,:,:]


hf = h5py.File('trainKidney_randomsequence.h5','w')
hf.create_dataset('fixed', data=fixed, dtype='f')
hf.create_dataset('moving',data=moving, dtype = 'f')
hf.close()