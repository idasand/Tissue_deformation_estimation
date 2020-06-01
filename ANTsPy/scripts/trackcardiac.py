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



#BATCH nr 1

filenames1 = ['J49BK002_1','J49BK002_2','J49BK002_3','J49BK006_1','J49BK006_2',
'J49BK006_3','J49BK08O_1','J49BK08O_2','J49BK08O_3','J49BK08O_4','J49BK08U_1','J49BK08U_2','J49BK08U_3',
'J49BK0G2_1','J49BK0G2_2']
filenames2 = ['J49BK0G2_3','J49BK0G6_1','J49BK0G6_2','J49BK0G6_3',
'J49BK0O2_1','J49BK0O2_2','J49BK0O2_3','J49BK0O6_1','J49BK0O6_2','J49BK0O6_3',
'J49BK102_1','J49BK102_2','J49BK102_3','J49BK106_1',
'J49BK106_2','J49BK106_3','J49BK1A4_1','J49BK1A4_2','J49BK1A4_3','J49BK19S_1',
'J49BK19S_2','J49BK19S_3']

#BATCH nr 2
filenames4 = ['J49BK1H0_1','J49BK1H0_2','J49BK1H0_3','J49BK1H4_1','J49BK1H4_2','J49BK1H4_3','J49BK1PU_1','J49BK1PU_2','J49BK1PU_3','J49BK1Q2_1','J49BK1Q2_2','J49BK1Q2_3','J49BK228_1','J49BK228_2','J49BK228_3','J49BK228_4','J49BK22G_1']
filenames4 =['J49BK22G_2','J49BK22G_3','J49BK286_1','J49BK286_2','J49BK286_3','J49BK2I0_1','J49BK2I0_2','J49BK2I0_3','J49BK2I4_1','J49BK2I4_2','J49BK2I4_3','J49BK2O2_1','J49BK2O2_2','J49BK2O2_3','J49BK2O8_1','J49BK2O8_2','J49BK2O8_3','J49BK31O_1','J49BK31O_2','J49BK31O_3','J49BK31O_4','J49BK31O_5','J49BK31S_1','J49BK31S_2','J49BK31S_3','J49BK31S_4','J49BK31S_5','J49BK382_1','J49BK382_2','J49BK382_3','J49BK386_1','J49BK386_2','J49BK386_3','J49BK3H0_1','J49BK3H0_2','J49BK3H0_3','J49BK3HE_1','J49BK3HE_2','J49BK3HE_3','J49BK3HE_4','J49BK3OG_1','J49BK3OG_2','J49BK3OG_3','J49BK3OK_1','J49BK3OK_2','J49BK3OK_3','J49BK3OK_4','J49BK420_1','J49BK420_2','J49BK420_3','J49BK424_1','J49BK424_2','J49BK424_3','J49BK4LG_1','J49BK4LG_2','J49BK4LG_3','J49BK4LQ_1','J49BK4LQ_2','J49BK4LQ_3','J49BK4LQ_4','J49BK4PK_1','J49BK4PK_2','J49BK4PK_3','J49BK4PO_1','J49BK4PO_2','J49BK4PO_3','J49BK518_1','J49BK518_2']
filenames2=['J49BK518_3','J49BK51C_1','J49BK51C_2','J49BK51C_3','J49BK58O_1','J49BK58O_2','J49BK58O_3','J49BK58S_1','J49BK58S_2','J49BK58S_3','J49BK5I0_1','J49BK5I0_2','J49BK5I0_3','J49BK5I4_1','J49BK5I4_2','J49BK5I4_3','J49BK5PO_1','J49BK5PO_2','J49BK5PO_3','J49BK5PK_1','J49BK5PK_2','J49BK5PK_3']


filenames = ['J49BK22G_1']


#filenames = ['J49BK1H0_1']
filenamesx = ['J49BK58O_3','J49BK58S_1','J49BK58S_2','J49BK58S_3','J49BK5I0_1','J49BK5I0_2','J49BK5I0_3','J49BK5I4_1','J49BK5I4_2','J49BK5I4_3','J49BK5PO_1','J49BK5PO_2','J49BK5PO_3','J49BK5PK_1','J49BK5PK_2','J49BK5PK_3']
#filenames = ['J49BK1H0_1']

#filenames = ['J49BK002_1','J49BK002_2','J49BK002_3','J49BK102_1','J49BK102_2','J49BK102_3','J49BK1A4_1','J49BK1A4_2','J49BK1A4_3']


#filenames = ['J49BK00A_1']

# longaxis removed'J49BK00A_1','J49BK00A_2','J49BK00A_3' ,'J49BK092_1 'J49BK092_2','J49BK092_3','J49BK0GA_1','J49BK0GA_2','J49BK0GA_3',
#'J49BK0OA_1','J49BK0OA_2','J49BK0OA_3' 'J49BK10A_1','J49BK10A_2','J49BK10A_3' 'J49BK1AI_1','J49BK1A1_2','J49BK1AI_3'

for i in range(len(filenames)):
	#print("CURRENTLY WORKING ON FILE NR " + str(i))

	hdf = h5py.File("../../cardiac_filtered/" + filenames[i] +"_Bilateral.h5",'r')
	arrayimagesCorrected = np.array(hdf["tissueCorrected/data"])
	arrayimages = np.array(hdf["tissue/data"])
	arrayimagesOriginal = np.array(hdf["tissueOriginal/data"])
	arrayimages=np.transpose(arrayimages,(2,1,0))
	arrayimagesCorrected=np.transpose(arrayimagesCorrected,(2,1,0))
	arrayimagesOriginal=np.transpose(arrayimagesOriginal,(2,1,0))
	initialpoints = np.array(hdf["tissue/trackingPoints"])

	initial_tracking_points = {'x': [initialpoints[0,0], initialpoints[0,1], initialpoints[0,2], initialpoints[0,3]], 'y': [initialpoints[1,0], initialpoints[1,1], initialpoints[1,2], initialpoints[1,3]]}
	points = np.array([[initialpoints[0,0], initialpoints[1,0]] , [initialpoints[0,1], initialpoints[1,1]] , [initialpoints[0,2], initialpoints[1,2]] , [initialpoints[0,3], initialpoints[1,3]]]) 

	d1 = initial_tracking_points
	
	#BILATERAL FILTERED IMAGES
	for j in range(arrayimages.shape[2]-1):
		arrayfix =arrayimagesOriginal[:,:,j]
		arraymov =arrayimagesOriginal[:,:,j+1]
		arrayfix = np.transpose(arrayfix)
		arraymov = np.transpose(arraymov)
		plt.imsave('fixed' + str(j) +'.png', arrayfix, cmap='gray')
		Image.open('fixed' + str(j) +'.png').convert('L').save('fixed' + str(j) +'.jpg')
		plt.imsave('moving' + str(j) +'.png', arraymov, cmap='gray')
		Image.open('moving' + str(j) +'.png').convert('L').save('moving' + str(j) +'.jpg')
		
		fi = ants.image_read('fixed' + str(j) +'.jpg')
		mi = ants.image_read('moving' + str(j) +'.jpg')
		#print(fi)
		print("START")
		print(datetime.datetime.now().time())
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
		#print(d1)
		x = d1['x']
		y = d1['y']
		v0 = [x[0],y[0]]
		v1 = [x[1],y[1]]
		v2 = [x[2],y[2]]
		v3 = [x[3],y[3]]
		v = np.stack((v0,v1,v2,v3))

		points = np.concatenate((points,v), axis=0)
		print("STOP")
		print(datetime.datetime.now().time())

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
	print("Strain in "+str(filenames[i])+" original:" )
	print(str((min(left_distances) - left_distances[0]) * 100 / left_distances[0]) +"  "+ str((min(right_distances) - right_distances[0])*100/ right_distances[0]))




	#VIDEO
	video = arrayimages
	video=np.transpose(video,(2,1,0))
	video = video[:,:,:,None]
	video2=np.squeeze(video, axis=3)
	tracked_points= np.around(points)
	#print("tracked points shape")
	#print(tracked_points.shape)
	anim = ultraSoundAnimation(video2,points=tracked_points)
	anim.save(filenames[i] + '_Original.mp4')
	
	'''


	#NON-LINEAR CORRECTED DATA
	initial_tracking_points = {'x': [initialpoints[0,0], initialpoints[0,1], initialpoints[0,2], initialpoints[0,3]], 'y': [initialpoints[1,0], initialpoints[1,1], initialpoints[1,2], initialpoints[1,3]]}
	points = np.array([[initialpoints[0,0], initialpoints[1,0]] , [initialpoints[0,1], initialpoints[1,1]] , [initialpoints[0,2], initialpoints[1,2]] , [initialpoints[0,3], initialpoints[1,3]]]) 


	d1 = initial_tracking_points
	for k in range(arrayimages.shape[2]-1):
		arrayfixCor =arrayimagesCorrected[:,:,k]
		arraymovCor =arrayimagesCorrected[:,:,k+1]
		arrayfixCor = np.transpose(arrayfixCor)
		arraymovCor = np.transpose(arraymovCor)
		plt.imsave('fixedCor' + str(k) +'.png', arrayfixCor, cmap='gray')
		Image.open('fixedCor' + str(k) +'.png').convert('L').save('fixedCor' + str(k) +'.jpg')
		plt.imsave('movingCor' + str(k) +'.png', arraymovCor, cmap='gray')
		Image.open('movingCor' + str(k) +'.png').convert('L').save('movingCor' + str(k) +'.jpg')
		
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
		v = np.stack((v0,v1,v2,v3))

		points = np.concatenate((points,v), axis=0)
		

		os.remove('fixedCor' + str(k) +'.png')
		os.remove('fixedCor' + str(k) +'.jpg')
		os.remove('movingCor' + str(k) +'.png')
		os.remove('movingCor' + str(k) +'.jpg')


	size0 = int(points.shape[0]) / int(4)
	points = np.reshape(points,(int(size0),4,2))
	points = np.around(points).astype(int)

	left_distancesCor = []
	right_distancesCor = []
	for m in range(arrayimagesCorrected.shape[2]-1):
		euclideandistanceleft = distance.euclidean(points[m,0,:], points[m,2,:])
		left_distancesCor = np.append(left_distancesCor,euclideandistanceleft)
		euclideandistanceright = distance.euclidean(points[m,1,:], points[m,3,:])
		right_distancesCor = np.append(right_distancesCor,euclideandistanceright)

	#Calclating Lagrangian strain
	print("Strain in "+str(filenames[i])+" corrected:" )
	print(str((min(left_distancesCor) - left_distancesCor[0]) * 100 / left_distancesCor[0]) +"  "+ str((min(right_distancesCor) - right_distancesCor[0])*100/ right_distancesCor[0]))

	#VIDEO - NON-LIN ON ORIGINAL DATA
	video = arrayimagesOriginal
	video=np.transpose(video,(2,1,0))
	video = video[:,:,:,None]
	video2=np.squeeze(video, axis=3)
	tracked_points= np.around(points)
	anim = ultraSoundAnimation(video2,points=tracked_points)
	anim.save(filenames[i] + '.mp4')
'''