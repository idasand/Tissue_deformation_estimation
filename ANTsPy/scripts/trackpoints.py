import ants
import ants.core
import numpy as np
import h5py
from PIL import Image 
from matplotlib import pyplot as plt
import os
import pandas as pd
from notebookHelpers import ultraSoundAnimation
from scipy.spatial import distance




filenames = ['J49BK002_1']
path = "path_to_file"

for i in range(len(filenames)):

	hdf = h5py.File(path + filenames[i] +".h5",'r')
	data = np.array(hdf["tissue/data"])

	#Transpose data if needed - should be [width,height,frame]
	#data=np.transpose(data,(2,1,0))
	
	#Get the initial tracking points into a DataFrame (Pandas-style)
	initialpoints = np.array(hdf["tissue/trackingPoints"])
	initial_tracking_points = {'x': [initialpoints[0,0], initialpoints[0,1], initialpoints[0,2], initialpoints[0,3]], 'y': [initialpoints[1,0], initialpoints[1,1], initialpoints[1,2], initialpoints[1,3]]}
	

	for j in range(data.shape[2]-1):
		arrayfix =data[:,:,j]
		arraymov =data[:,:,j+1]

		#Stupid image conversion to get the ANTsImage
		plt.imsave('fixed' + str(j) +'.png', arrayfix, cmap='gray')
		Image.open('fixed' + str(j) +'.png').convert('L').save('fixed' + str(j) +'.jpg')
		plt.imsave('moving' + str(j) +'.png', arraymov, cmap='gray')
		Image.open('moving' + str(j) +'.png').convert('L').save('moving' + str(j) +'.jpg')
		fi = ants.image_read('fixed' + str(j) +'.jpg')
		mi = ants.image_read('moving' + str(j) +'.jpg')
		

		mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyNCC') )
		pts = pd.DataFrame(data=initial_tracking_points)
		initial_tracking_points = ants.apply_transforms_to_points(2, pts, mytx['fwdtransforms'])

		'''#If the points need to be printed
		x = d1['x']
		y = d1['y']
		v0 = [x[0],y[0]]
		v1 = [x[1],y[1]]
		v2 = [x[2],y[2]]
		v3 = [x[3],y[3]]
		v = np.stack((v0,v1,v2,v3))
		points = np.concatenate((points,v), axis=0)'''


		os.remove('fixed' + str(j) +'.png')
		os.remove('fixed' + str(j) +'.jpg')
		os.remove('moving' + str(j) +'.png')
		os.remove('moving' + str(j) +'.jpg')


	#Calculation of the distances in both segments
	left_distances = []
	right_distances = []
	for n in range(arrayimages.shape[2]-1):
		euclideandistanceleft = distance.euclidean(points[n,0,:], points[n,2,:])
		left_distances = np.append(left_distances,euclideandistanceleft)
		euclideandistanceright = distance.euclidean(points[n,1,:], points[n,3,:])
		right_distances = np.append(right_distances,euclideandistanceright)

	#Calclation of longitudinal Lagrangian strain in both basal segments
	print("Strain in "+str(filenames[i])+" original:" )
	print(str((min(left_distances) - left_distances[0]) * 100 / left_distances[0]) +"  "+ str((min(right_distances) - right_distances[0])*100/ right_distances[0]))


	#Video of the estimated points overlaid on the ultrasound video
	video = arrayimages
	video=np.transpose(video,(2,1,0))
	anim = ultraSoundAnimation(video2,points=np.around(points))
	anim.save(filenames[i] + '_Original.mp4')
	