import tensorflow as tf
import numpy as np
from models.deformableNet import DeformableNet
from misc.notebookHelpers import ultraSoundAnimation
from misc.plots import *
import matplotlib.axes as ax
#import axes.quiver
import argparse
import os
import h5py
import glob
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment-id',
                    help='Name of the current experiment',
                    required=True)
parser.add_argument('-l', '--leakage',
                    type=float, default=0.2,
                    help='Leakage parameter in leaky relu units')
parser.add_argument('-s', '--smoothing',
                    type=float, default=0.,
                    help='Smoothing parameter for exp moving average filter')
parser.add_argument('-ds', '--downsampling-layers',
                    type=str, default='4,2,1',
                    help='Comma delimited(no spaces) list containing ' +\
                    'the number of downsampling layers for each network')
parser.add_argument('-op', '--output-path',
                    default='../output',
                    help='Path to storing/loading output, ' +
                    'such as model weights, logs etc.')
parser.add_argument('-dp', '--data-path',
                    default='../data/testfiles',#/processed/strain_point',
                    help='Path to dataset')

args = parser.parse_args()


def trackPoints(defnets, fixed, moving, points, smoothing=0.):
    num_frames = fixed.shape[0] + 1
    tracked_points = np.zeros((num_frames, points.shape[0], 2), int)

    displacements = np.zeros((fixed.shape[0],
                              fixed.shape[1],
                              fixed.shape[2], 2))

    for i, defnet in enumerate(defnets):
        moving = defnet(fixed, moving)
        displacements += defnet.interpolated_displacements.numpy()
    yy, xx = np.mgrid[:fixed.numpy().shape[1],
                      :fixed.numpy().shape[2]]

    xx = np.tile(xx[None, :, :], [fixed.shape[0], 1, 1])
    yy = np.tile(yy[None, :, :], [fixed.shape[0], 1, 1])

    grid = np.concatenate((xx[:, None, :, :], yy[:, None, :, :]),
                          axis=1)

    warped_grid = grid + np.transpose(displacements, [0, 3, 1, 2])
    displacements = np.transpose(displacements, [0, 3, 1, 2])
    for j in range(points.shape[0]):
        x_coord = np.round(points[j, 0]).astype(int)
        y_coord = np.round(points[j, 1]).astype(int)

        tracked_points[0, j, 0] = x_coord
        tracked_points[0, j, 1] = y_coord
        for frame_num in range(num_frames - 1):
            # Find points in next frame
            next_x_coord = warped_grid[frame_num, 0, y_coord, x_coord]
            next_y_coord = warped_grid[frame_num, 1, y_coord, x_coord]
            next_x_coord = np.clip(next_x_coord, 0, fixed.numpy().shape[2] - 1)
            next_y_coord = np.clip(next_y_coord, 0, fixed.numpy().shape[1] - 1)
            next_x_coord = np.round(next_x_coord).astype(int)
            next_y_coord = np.round(next_y_coord).astype(int)

            # Update current points
            x_coord = next_x_coord
            y_coord = next_y_coord
            tracked_points[frame_num + 1, j, 0] =\
                (1 - smoothing) * x_coord +\
                smoothing * tracked_points[frame_num, j, 0]
            tracked_points[frame_num + 1, j, 1] =\
                (1 - smoothing) * y_coord +\
                smoothing * tracked_points[frame_num, j, 1]

    #     Tracking with Kalman filter
    # for j in range(points.shape[0]):
    #     kalman = cv2.KalmanFilter(4, 2)
    #     kalman.measurementMatrix = np.eye(4, dtype=np.float32)
    #     kalman.transitionMatrix = np.array([[1, 0, 1, 0],
    #                                         [0, 1, 0, 1],
    #                                         [0, 0, 1, 0],
    #                                         [0, 0, 0, 1]], np.float32)
    #     kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * .005
    #     kalman.processNoiseCov = np.eye(4, dtype=np.float32) * .005

    #     mp = np.array([[np.round(points[j, 0])],
    #                    [np.round(points[j, 1])],
    #                    [0],
    #                    [-1]]).astype(np.float32)
    #     kalman.statePre = mp.copy()
    #     tracked_points[0, j, :] = mp[:2, 0].astype(int)

    #     for frame_num in range(num_frames - 1):
    #         displacement = displacements[frame_num, :,
    #                                      tracked_points[frame_num, j, 1],
    #                                      tracked_points[frame_num, j, 0]]
    #         position = tracked_points[frame_num, j, :] + displacement

    #         mp = np.array([position[0],
    #                        position[1],
    #                        displacement[0],
    #                        displacement[1]]).astype(np.float32)

    #         kalman.correct(mp)
    #         tp = np.round(kalman.predict()).astype(int)
    #         tp[0, 0] = np.clip(tp[0, 0], 0, fixed.shape[2])
    #         tp[1, 0] = np.clip(tp[1, 0], 0, fixed.shape[1])
    #         tracked_points[frame_num + 1, j, :] = tp[:2, 0]

    return tracked_points, displacements


def distances(tracked_points):
    left_dist = np.square(tracked_points[:, 0, :] - tracked_points[:, 2, :])
    left_dist = np.sqrt(left_dist[:, 0] + left_dist[:, 1])

    right_dist = np.square(tracked_points[:, 1, :] - tracked_points[:, 3, :])
    right_dist = np.sqrt(right_dist[:, 0] + right_dist[:, 1])

    return left_dist, right_dist


savedir = os.path.join(args.output_path, 'models',
                       args.experiment_id)

if not os.path.exists(os.path.join(args.output_path,
                                   'videos', args.experiment_id)):
    os.makedirs(os.path.join(args.output_path,
                             'videos', args.experiment_id))
if not os.path.exists(os.path.join(args.output_path,
                                   'results', args.experiment_id)):
    os.makedirs(os.path.join(args.output_path,
                             'results', args.experiment_id))

ds_layers = [int(num) for num in args.downsampling_layers.split(',')]
defnets = [DeformableNet(num, args.leakage) for num in ds_layers]

for i, defnet in enumerate(defnets):
    try:
        defnet.load_weights(os.path.join(savedir, str(i), args.experiment_id))
        print(f'Loaded weights from savedir')
    except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
        print('No previous weights found or weights couldn\'t be loaded')
        exit(-1)

# Nonlinear mapping for contrast enhancement
#pal_txt = open('../../pal.txt')
#line = pal_txt.readline()[:-1]
#pal = np.array([float(val) for val in line.split(',')])

# view_and_vals = pd.read_csv(
#     '../data/raw/Ultrasound_high_fps/strain_and_view_gt.csv',
#     delimiter=';')

#views = pd.read_csv('../data/raw/views.csv')
#views = dict(zip(views.file, views.view))
# views = dict(zip(view_and_vals.File, view_and_vals.View))


left_strains = {}
right_strains = {}

#for h5file in h5files:
#with h5py.File('../data/testfiles/J5A8HSO6_test.h5') as file:
f_read = h5py.File('../data/testfiles/J5M90186_test_structure.h5', 'r')
data = np.array(f_read['data'])
track_points = np.array(f_read['track_points'])
print("inside with")


video = data

print("video size before for loop")
print(video.shape)

'''video = np.array([[[[int(round(video[k, j, i]))]
                    for k in range(video.shape[0])]
                   for j in range(video.shape[1])]
                  for i in range(video.shape[2])])'''

video=np.transpose(video,(2,1,0))
video = video[:,:,:,None]

#video /= 255
#fps = 1 / (file['tissue/times'][3] - data['tissue/times'][2])

#ds_labels = data['tissue/ds_labels']
points = track_points
#es = np.argwhere(ds_labels[:] == 2.)[0][0]

print("video size")
print(video.shape)

fixed = tf.constant(video[:-1, :, :], #None på siste her
                    dtype='float32')
moving = tf.constant(video[1:, :, :],
                     dtype='float32')

print("fixed size")
print(fixed.shape)
print("moving size")
print(moving.shape)

tracked_points, displacements = trackPoints(defnets,
                                            fixed, moving,
                                            points,
                                            smoothing=0)
print("displacements")
print(displacements.shape)
print(tracked_points)

#plotVectorField(ax, displacements)
#plotGrid(ax, displacements, **kwargs)

'''
#print("tracked points")
#print(tracked_points)

video2=np.squeeze(video, axis=3)

print("video2")
print(video2.shape)
anim = ultraSoundAnimation(video2,points=tracked_points)

anim.save('../output/videos/19nov421_pas4.mp4')
file_name = '1'
file_num = 12
anim.save(os.path.join(args.output_path,
                            'videos', args.experiment_id,
                            file_name + f'_{file_num}' + '.mp4'),writer="ffmpeg")'''

#left_dist, right_dist = distances(tracked_points)


#results = pd.DataFrame(vals,
#                       columns=['file', 'cycle', 'view', 'left_strain', 'right_strain'])
#results.to_csv(os.path.join(args.output_path,
#                            'results', args.experiment_id, 'strain_ests.csv'))