from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)
This file is part of the OSVOS paper presented in:
	CVPR 2017

Daniel Syles Immanuel (xmandsi@gmail.com)
This file is a part of the OSVOS Paper presented in:
	Bennett University Deep Learning Internship 2019.

Please consider citing the paper if you use this code.
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset
os.chdir(root_folder)

# User defined parameters
# Change 1080p to 480p if you want to Test it with the 480p Images
# Using 480p will significantly reduce the time of computation
seq_name = "camel"
gpu_id = 0
train_model = True
result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '1080p', 'OSVOS', seq_name)

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = os.path.join('models', seq_name)
max_training_iters = 5000

# Define Dataset
test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '1080p', seq_name)))
test_imgs = [os.path.join('DAVIS', 'JPEGImages', '1080p', seq_name, frame) for frame in test_frames]
if train_model:
    train_imgs = [os.path.join('DAVIS', 'JPEGImages', '1080p', seq_name, '00000.jpg')+' '+
                  os.path.join('DAVIS', 'Annotations', '1080p', seq_name, '00000.png')]
    dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
else:
    dataset = Dataset(None, test_imgs, './')

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
        osvos.test(dataset, checkpoint_path, result_path)

# Show results
overlay_color = [255, 0, 0]
transparency = 0.6
plt.ion()
for img_p in test_frames:
    frame_num = img_p.split('.')[0]
    img = np.array(Image.open(os.path.join('DAVIS', 'JPEGImages', '1080p', seq_name, img_p)))
    mask = np.array(Image.open(os.path.join(result_path, frame_num+'.jpg')))
    mask = mask//np.max(mask)
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
    plt.imshow(im_over.astype(np.uint8))
    plt.axis('off')
    plt.show()
    plt.pause(0.01)
    plt.clf()
