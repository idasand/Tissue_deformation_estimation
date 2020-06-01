import ants
import ants.core
import numpy as np
import h5py
from PIL import Image 
from matplotlib import pyplot as plt
import os #for deleting images after the grids are computed
import pandas as pd
from notebookHelpers import ultraSoundAnimation
from scipy.spatial import distance
import datetime



#BATCH

#filenames = ['K2RBPKP0']
#filenames = ['K2RBOT0O']
filenames = ['K2RBPE8U']

for i in range(len(filenames)):
	#print("CURRENTLY WORKING ON FILE NR " + str(i))

	hdf = h5py.File("../../ShortAxis/" + filenames[i] +"_Bilateral.h5",'r')
	arrayimagesCorrected = np.array(hdf["tissueCorrected/data"])
	#arrayimages = np.array(hdf["tissue/data"])
	arrayimages = np.array(hdf["tissueOriginal/data"])
	#arrayimages=np.transpose(arrayimages,(2,1,0))
	#arrayimagesCorrected=np.transpose(arrayimagesCorrected,(2,1,0))
	#arrayimagesOriginal=np.transpose(arrayimagesOriginal,(2,1,0))
	initialpoints = np.array(hdf["tissue/trackingPoints"])
	print(arrayimages.shape)
	initial_tracking_points = {'x': [initialpoints[0,0], initialpoints[0,1], initialpoints[0,2], initialpoints[0,3], initialpoints[0,4], initialpoints[0,5]], 'y': [initialpoints[1,0], initialpoints[1,1], initialpoints[1,2], initialpoints[1,3], initialpoints[1,4] , initialpoints[1,5]]}
	points = np.array([[initialpoints[0,0], initialpoints[1,0]] , [initialpoints[0,1], initialpoints[1,1]] , [initialpoints[0,2], initialpoints[1,2]] , [initialpoints[0,3], initialpoints[1,3]] , [initialpoints[0,4], initialpoints[1,4]] , [initialpoints[0,5], initialpoints[1,5]]]) 
	arrayimages = arrayimages[:,:,:40]
	d1 = initial_tracking_points
	print(arrayimages.shape)
	'''
	#BILATERAL FILTERED IMAGES
	for j in range(2):#range(arrayimages.shape[2]-1):
		arrayfix =arrayimages[:,:,j]
		arraymov =arrayimages[:,:,j+1]
		arrayfix = np.transpose(arrayfix)
		arraymov = np.transpose(arraymov)
		plt.imsave('fixed' + str(j) +'.png', arrayfix, cmap='gray')
		Image.open('fixed' + str(j) +'.png').convert('L').save('fixed' + str(j) +'.jpg')
		plt.imsave('moving' + str(j) +'.png', arraymov, cmap='gray')
		Image.open('moving' + str(j) +'.png').convert('L').save('moving' + str(j) +'.jpg')
		
		fi = ants.image_read('fixed' + str(j) +'.jpg')
		mi = ants.image_read('moving' + str(j) +'.jpg')
		#print(fi)
		
		#mygr = ants.create_warped_grid(mi)
		#mygr.plot()
		mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyNCC') )
		#print(mytx['fwdtransforms'])
		#mywarpedgrid = ants.create_warped_grid( mi,
		#                        transform=mytx['fwdtransforms'], fixed_reference_image=fi )
		#mywarpedgrid.plot()


		pts = pd.DataFrame(data=d1)
		#point1 = ants.transform_index_to_physical_point(fi, (112, 120))
		#print(point1)
		d1 = ants.apply_transforms_to_points( 2, pts, mytx['fwdtransforms'])
		#d = tsw 
		print(d1)
		x = d1['x']
		y = d1['y']
		v0 = [x[0],y[0]]
		v1 = [x[1],y[1]]
		v2 = [x[2],y[2]]
		v3 = [x[3],y[3]]
		v = np.stack((v0,v1,v2,v3))

		points = np.concatenate((points,v), axis=0)
		

		os.remove('fixed' + str(j) +'.png')
		os.remove('fixed' + str(j) +'.jpg')
		os.remove('moving' + str(j) +'.png')
		os.remove('moving' + str(j) +'.jpg')


	size0 = int(points.shape[0]) / int(4)
	points = np.reshape(points,(int(size0),4,2))
	#print(np.around(points).astype(int))
	points = np.around(points).astype(int)
	#print(points)
	#print(points.shape)

	#points=np.reshape(points())
	#print(warpedgrids[0,0:10,0:10])
	#warpedgridex = warpedgrids[4,:,:]
	#warpedgridex.plot()
	#mywarpedgrids(4,:,;).plot()
	
	left_distances = []
	right_distances = []
	for n in range(arrayimages.shape[2]-1):
		euclideandistanceleft = distance.euclidean(points[n,0,:], points[n,2,:])
		left_distances = np.append(left_distances,euclideandistanceleft)
		euclideandistanceright = distance.euclidean(points[n,1,:], points[n,3,:])
		right_distances = np.append(right_distances,euclideandistanceright)

	#Calclating Lagrangian strain
	print("Left strain in "+str(filenames[i])+" bilateral:" )
	print((min(left_distances) - left_distances[0]) / left_distances[0])
	print("Right strain in "+str(filenames[i])+" bilateral:" )
	print((min(right_distances) - right_distances[0]) / right_distances[0])
	'''


	


	#NON-LINEAR CORRECTED DATA

	for k in range(arrayimages.shape[2]-1):
		arrayfixCor =arrayimagesCorrected[:,:,k]
		arraymovCor =arrayimagesCorrected[:,:,k+1]
		arrayfixCor = np.transpose(arrayfixCor)
		arraymovCor = np.transpose(arraymovCor)
		plt.imsave('fixedCor' + str(k) +'.png', arrayfixCor, cmap='gray')
		Image.open('fixedCor' + str(k) +'.png').convert('L').save('fixedCor' + str(k) +'.jpg')
		plt.imsave('movingCor' + str(k) +'.png', arraymovCor, cmap='gray')
		Image.open('movingCor' + str(k) +'.png').convert('L').save('movingCor' + str(k) +'.jpg')
		
		print("START")
		print(datetime.datetime.now().time())
		fi = ants.image_read('fixedCor' + str(k) +'.jpg')
		mi = ants.image_read('movingCor' + str(k) +'.jpg')

		mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyNCC') )
		#print(mytx['fwdtransforms'])
		#mywarpedgrid = ants.create_warped_grid( mi,
		#                        transform=mytx['fwdtransforms'], fixed_reference_image=fi )
		#mywarpedgrid.plot()


		pts = pd.DataFrame(data=d1)
		#point1 = ants.transform_index_to_physical_point(fi, (112, 120))
		#print(point1)
		d1 = ants.apply_transforms_to_points( 2, pts, mytx['fwdtransforms'])
		#d = tsw 
		#print(d1)
		x = d1['x']
		y = d1['y']
		v0 = [x[0],y[0]]
		v1 = [x[1],y[1]]
		v2 = [x[2],y[2]]
		v3 = [x[3],y[3]]
		v4 = [x[4],y[4]]
		v5 = [x[5],y[5]]
		v = np.stack((v0,v1,v2,v3,v4,v5))

		points = np.concatenate((points,v), axis=0)
		print("STOP")
		print(datetime.datetime.now().time())

		os.remove('fixedCor' + str(k) +'.png')
		os.remove('fixedCor' + str(k) +'.jpg')
		os.remove('movingCor' + str(k) +'.png')
		os.remove('movingCor' + str(k) +'.jpg')


	size0 = int(points.shape[0]) / int(6)
	points = np.reshape(points,(int(size0),6,2))
	points = np.around(points).astype(int)
	print(points)
	'''
	left_distancesCor = []
	right_distancesCor = []
	for m in range(arrayimagesCorrected.shape[2]-1):
		euclideandistanceleft = distance.euclidean(points[m,0,:], points[m,2,:])
		left_distancesCor = np.append(left_distancesCor,euclideandistanceleft)
		euclideandistanceright = distance.euclidean(points[m,1,:], points[m,3,:])
		right_distancesCor = np.append(right_distancesCor,euclideandistanceright)

	#Calclating Lagrangian strain
	print("Left strain in "+str(filenames[i])+" corrected:" )
	print((min(left_distancesCor) - left_distancesCor[0]) / left_distancesCor[0])
	print("Right strain in "+str(filenames[i])+" corrected:" )
	print((min(right_distancesCor) - right_distancesCor[0]) / right_distancesCor[0])
	'''
'''
	#VIDEO - NON-LIN ON ORIGINAL DATA
	video = arrayimagesOriginal
	video=np.transpose(video,(2,1,0))
	video = video[:,:,:,None]
	video2=np.squeeze(video, axis=3)
	tracked_points= np.around(points)
	anim = ultraSoundAnimation(video2,points=tracked_points)
	anim.save(filenames[i] + '_BMode.mp4')

	#VIDEO
	video = arrayimagesCorrected
	video=np.transpose(video,(2,1,0))
	video = video[:,:,:,None]
	video2=np.squeeze(video, axis=3)
	tracked_points= np.around(points)
	#print("tracked points shape")
	#print(tracked_points.shape)
	anim = ultraSoundAnimation(video2,points=tracked_points)
	anim.save(filenames[i] + '_Bi_nonlin.mp4')
	
	'''