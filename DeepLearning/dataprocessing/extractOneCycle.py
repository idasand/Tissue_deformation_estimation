import h5py
import numpy as np
import argparse

lengthOneCycle = 120
framestep = 3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path')
args = parser.parse_args()

f_read = h5py.File(args.data_path, 'r')
print('Keys in the h5 file parsed:')
for key in f_read.keys():
    print(key)

tissue = np.array(f_read['tissuepadded'])
tissueSmooth = np.array(f_read['tissuepaddedSmooth'])


#Padding the normal US images
imgs = np.empty(shape=(410,316,lengthOneCycle))
#print(imgs.shape)
this_img = np.empty(shape=(410,316))

for i in range(0,lengthOneCycle,framestep):
    this_img[:,:] = tissue[:,:,i]
    imgs = np.dstack((imgs, this_img))


#Padding the filtered US images
imgs = np.empty(shape=(410,316,lengthOneCycle))
for i in range(0,lengthOneCycle,framestep):
    this_img[:,:] = tissueSmooth[:,:,i]
    imgs = np.dstack((imgs, this_img))

tissue=imgs[:,:,lengthOneCycle:]
tissueSmooth=imgs[:,:,lengthOneCycle:]


hf = h5py.File('round3/J5AAFTG4_padded_onecycle.h5','w')
hf.create_dataset('tissuepaddedOneCycle', data=tissue)
hf.create_dataset('tissuepaddedSmoothOneCycle',data=tissueSmooth)
hf.close()