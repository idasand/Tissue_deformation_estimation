import tensorflow as tf
import numpy as np
from models.deformableNet import DeformableNet
from models.losses import maskedNNCC2d, unmaskedNNCC2d
from data.dataLoader import DataSet
import matplotlib.pyplot as plt
from misc.plots import plotGrid
import argparse
import os
import io
import time

tf.enable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment-id',
                    help='Name of the current experiment',
                    required=True)
parser.add_argument('-e', '--num-epochs',
                    type=int, default=5,
                    help='Number of steps to run training')
parser.add_argument('-bs', '--batch-size',
                    type=int, default=16,
                    help='Batch size to use for training')
parser.add_argument('-lr', '--learning-rate',
                    type=float, default=1e-4,
                    help='Learning rate for optimizer')
parser.add_argument('-bp', '--penalty',
                    type=float, default=1e-6,
                    help='Impact of the bending penalty in the loss function')
parser.add_argument('-l', '--leakage',
                    type=float, default=0.2,
                    help='Leakage parameter in leaky relu units')
parser.add_argument('-ds', '--downsampling-layers',
                    type=str, default='2',
                    help='Comma delimited(no spaces) list containing ' +\
                    'the number of downsampling layers for each network')
parser.add_argument('-op', '--output-path',
                    default='../output',
                    help='Path to storing output, ' +
                    'such as model weights, logs etc.')
parser.add_argument('-mp', '--load-path',
                    default='../output',
                    help='Path to output previous training for' +
                    ' restoring weights')
parser.add_argument('-dp', '--data-path',
                    default='dataprocessing/',
                    #default='../data/processed/strain_point',
                    help='Path to dataset')

args = parser.parse_args()

savedir = os.path.join(args.output_path, 'models',
                       args.experiment_id)
loaddir = os.path.join(args.load_path, 'models', args.experiment_id)
if not os.path.exists(savedir):
    os.makedirs(savedir)
logdir = os.path.join(args.output_path, 'logs', args.experiment_id)
if not os.path.exists(logdir):
    os.makedirs(logdir)


def gridPlotTobuffer(warped_grid):
    fig, ax = plt.subplots()
    plotGrid(ax, warped_grid[0, :, :, :], color='purple')
    ax.set_title('Warp Grid')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return buf


ds_layers = [int(num) for num in args.downsampling_layers.split(',')]
defnets = [DeformableNet(num, args.leakage) for num in ds_layers]

train_writer = tf.contrib.summary.create_file_writer(logdir)
train_writer.set_as_default()

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
global_step = tf.train.get_or_create_global_step()

for i, defnet in enumerate(defnets):
    try:
        defnet.load_weights(os.path.join(loaddir, str(i), args.experiment_id))
        print('Loaded weights from savedir')
        saved_gs = int(np.loadtxt(os.path.join(loaddir, 'global_step.txt')))
        global_step.assign(saved_gs)
        print(f"Starting training at step: {saved_gs}")
    except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
        print('No previous weights found or weights couldn\'t be loaded')
        print('Starting fresh')

train_loader = DataSet(os.path.join(args.data_path, 'trainKidney_randomsequence.h5'))
val_loader = DataSet(os.path.join(args.data_path, 'valKidney_randomsequence.h5'))
for i, defnet in enumerate(defnets):
    print(f"Training stage: {i}")
    modeldir = os.path.join(savedir, str(i), args.experiment_id)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    for _ in range(args.num_epochs):
        batch_gen = train_loader.batchGenerator(args.batch_size)
        end = 0
        for fixed, moving in batch_gen:
            begin = time.time()
            print(f'{{"metric": "Load time", "value": {(begin - end)}, ' +
                  f'"step": {global_step.numpy()}}}')
            xx, yy = tf.meshgrid(
                tf.range(0., fixed.shape[2]),
                tf.range(0., fixed.shape[1]))
            warped_grid = tf.stack([xx, yy])
            if i == 0:
                with tf.GradientTape() as tape:
                    warped = defnet(fixed, moving)
                    warped_grid = tf.add(
                        warped_grid,
                        tf.transpose(defnet.interpolated_displacements,
                                     [0, 3, 1, 2]))

                    cross_corr = unmaskedNNCC2d(fixed, warped)
                    bending_pen = defnet.bending_penalty
                    loss = cross_corr + args.penalty * bending_pen
            else:
                # Run through preceeding stages
                for j in range(i):
                    moving = defnets[j](fixed, moving)
                    warped_grid = tf.add(
                        warped_grid,
                        tf.transpose(defnets[j].interpolated_displacements,
                                     [0, 3, 1, 2]))
                with tf.GradientTape() as tape:
                    warped = defnet(fixed, moving)
                    warped_grid = tf.add(
                        warped_grid,
                        tf.transpose(defnet.interpolated_displacements,
                                     [0, 3, 1, 2]))

                    cross_corr = unmaskedNNCC2d(fixed, warped)
                    bending_pen = defnet.bending_penalty
                    loss = cross_corr + args.penalty * bending_pen

            grads = tape.gradient(loss, defnet.trainable_variables)

            print(f'{{"metric": "Loss", "value": {loss}, ' +
                  f'"step": {global_step.numpy()}}}')
            print(f'{{"metric": "NCC", "value": {cross_corr}, ' +
                  f'"step": {global_step.numpy()}}}')
            print(f'{{"metric": "Bending_penalty", "value": ' +
                  f'{args.penalty * bending_pen},' +
                  f' "step": {global_step.numpy()}}}')
            # with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            #     tf.contrib.summary.scalar('loss', loss)
            #     tf.contrib.summary.scalar('NCC loss', cross_corr)
            #     tf.contrib.summary.scalar('Bending penalty',
            #                               args.penalty * bending_pen)
            #     for var in defnet.trainable_variables:
            #         tf.contrib.summary.histogram(var.name, var)
            #     for grad, var in zip(grads, defnet.trainable_variables):
            #         tf.contrib.summary.histogram(var.name + '/gradient', grad)
            with tf.contrib.summary.record_summaries_every_n_global_steps(100):
                tf.contrib.summary.image('Fixed', fixed[0:1, :, :, :],
                                         max_images=1)
                tf.contrib.summary.image('Moving', moving[0:1, :, :, :],
                                         max_images=1)
                tf.contrib.summary.image('Warped', warped[0:1, :, :, :],
                                         max_images=1)
                buf = gridPlotTobuffer(warped_grid)
                grid_img = tf.expand_dims(
                    tf.image.decode_png(buf.getvalue(), channels=4), 0)
                tf.contrib.summary.image('Warp Grid', grid_img)

            optimizer.apply_gradients(zip(grads, defnet.trainable_variables),
                                      global_step=global_step)

            end = time.time()
            if global_step.numpy() % 100 == 0:
                defnet.save_weights(modeldir)
                np.savetxt(os.path.join(savedir, 'global_step.txt'),
                           [global_step.numpy()])

                batch_gen = val_loader.batchGenerator(args.batch_size,
                                                      shuffle=False)
                cross_corrs = []
                for val_fixed, val_moving in batch_gen:
                    val_warped = defnet(val_fixed, val_moving)
                    cross_corr = unmaskedNNCC2d(val_fixed, val_warped)
                    cross_corrs.append(cross_corr)
                mean_cross_corr = np.mean(cross_corrs)
                print(f'{{"metric": "Mean Validation NCC",' +
                      f' "value": {mean_cross_corr}, ' +
                      f'"step": {global_step.numpy()}}}')

