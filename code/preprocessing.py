# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 21:53:15 2018

@author: swati
"""

import os, glob, gc, h5py
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave

import tensorlayer as tl
import pickle

#------------- Bounding Box (ROI) -----------------------

def bounding_box_and_normalising(gray_image, gray_mask):
    i, j = np.where(gray_mask)
    indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                          np.arange(min(j), max(j) + 1),
                          indexing='ij')
    
    sub_image = gray_image[indices]
    pixels = sub_image[sub_image > 0]
    mean = pixels.mean()
    std  = pixels.std()
    norm_img = (sub_image - mean)/std
    print(norm_img.min(), norm_img.max())
    sub_mask = gray_mask[indices]
    sub_mask = (sub_mask > 175).astype(int)
        
    x = np.multiply(norm_img, sub_mask)
    print(x.min(), x.max())
        
    desired_size = 256
    old_size = x.shape[:2]
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    image = cv2.resize(x, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    return new_im


#------------------------------------------------------------

data_dir = "S:\\ML+AI tutorials\\Projects\\Tumour_Classification\\Datasets\\"

patients_g = np.asarray([glob.glob(data_dir + '/glioma/new_tumor[0-9]*.jpg')]).reshape(1426, 1)
patients_m = np.asarray([glob.glob(data_dir + '/meningioma/new_tumor[0-9]*.jpg')]).reshape(708, 1)
patients_p = np.asarray([glob.glob(data_dir + '/pituitary_tumor/new_tumor[0-9]*.jpg')]).reshape(930, 1)

x_g = np.asarray([cv2.imread(p_g[0], 0) for p_g in patients_g])
x_m = np.asarray([cv2.imread(p_m[0], 0) for p_m in patients_m])
x_p = [cv2.imread(p_p[0], 0) for p_p in patients_p]

z = []
for p in x_p:
    if p.shape == (512, 512):
        z.append(p)
x_p = np.asarray(z)

masks_g = np.asarray([glob.glob(data_dir + '/glioma/new_tumor_mask*.jpg')]).reshape(1426, 1)
masks_m = np.asarray([glob.glob(data_dir + '/meningioma/new_tumor_mask*.jpg')]).reshape(708, 1)
masks_p = np.asarray([glob.glob(data_dir + '/pituitary_tumor/new_tumor_mask*.jpg')]).reshape(930, 1)

print(masks_p[0].shape)

mask_g = np.asarray([cv2.imread(m_g[0], 0) for m_g in masks_g])
mask_m = np.asarray([cv2.imread(m_m[0], 0) for m_m in masks_m])
mask_p = [cv2.imread(m_p[0], 0) for m_p in masks_p]

z = []
for m in mask_p:
    if m.shape == (512, 512):
        z.append(m)
mask_p = np.asarray(z)

X_g = np.empty((x_g.shape[0], 256, 256))
for glioma_imgs_i in range(x_g.shape[0]):
    X_g[glioma_imgs_i] = bounding_box_and_normalising(x_g[glioma_imgs_i], mask_g[glioma_imgs_i])
    
X_m = np.empty((x_m.shape[0], 256, 256))
for menin_imgs_i in range(x_m.shape[0]):
    X_m[menin_imgs_i] = bounding_box_and_normalising(x_m[menin_imgs_i], mask_m[menin_imgs_i])
    
X_p = np.empty((x_p.shape[0], 256, 256))
for pitui_imgs_i in range(x_p.shape[0]):
    X_p[pitui_imgs_i] = bounding_box_and_normalising(x_p[pitui_imgs_i], mask_p[pitui_imgs_i])


#-----------------Saving data-----------------------
prep_data_dir = "S:\\ML+AI tutorials\\Projects\\Tumour_Classification\\Datasets\\preprocessed_data\\"
with h5py.File(prep_data_dir + 'prep_dataset.hdf5', 'w') as hf:
    hf.create_dataset("glioma",  data=X_g)
    hf.create_dataset("meningioma",  data=X_m)
    hf.create_dataset("pituitary_tumor",  data=X_p)

#----------------------Load data----------------------------
prep_data_dir = "S:\\ML+AI tutorials\\Projects\\Tumour_Classification\\Datasets\\preprocessed_data\\"
with h5py.File(prep_data_dir + 'prep_dataset.hdf5', 'r') as hf:
#    X_g = hf['glioma'][:]
    X_m = hf['meningioma'][:]
    X_p = hf['pituitary_tumor'][:]
    
X_g = X_g[:,:,:,np.newaxis]
X_m = X_m[:,:,:,np.newaxis]
X_p = X_p[:,:,:,np.newaxis]

y_g = np.array([[1, 0, 0]] * X_g.shape[0])
y_m = np.array([[0, 1, 0]] * X_m.shape[0])
y_p = np.array([[0, 0, 1]] * X_p.shape[0])


#-------------------------Shuffle Input and Output -----------------------------
         

Y = np.concatenate([y_g, y_m, y_p], axis = 0)

X = np.concatenate([X_g, X_m, X_p], axis=0)

X = X[:, :, :, np.newaxis]

with h5py.File(prep_data_dir + 'X_data_aug_unshuffled.hdf5', 'a') as hf:
    hf.create_dataset("X",  data=X)
    hf.create_dataset("Y",  data=Y)

prep_data_dir = "S:\\ML+AI tutorials\\Projects\\Tumour_Classification\\Datasets\\preprocessed_data\\"
with h5py.File(prep_data_dir + 'X_data_aug_unshuffled.hdf5', 'r') as hf:
    print(list(hf.keys()))
    X = hf['X'][:]
    


idx = np.random.rand(X.shape[0]).argsort()
data, labels = X[idx], Y[idx]

#-----------------Saving data-----------------------
prep_data_dir = "S:\\ML+AI tutorials\\Projects\\Tumour_Classification\\Datasets\\preprocessed_data\\"
with h5py.File(prep_data_dir + 'prep_aug_shuffled_dataset.hdf5', 'w') as hf:
    hf.create_dataset("data",  data = data)
    hf.create_dataset("labels",  data = labels)
#------------------------Load data -----------------------------------
prep_data_dir = "S:\\ML+AI tutorials\\Projects\\Tumour_Classification\\Datasets\\preprocessed_data\\"
with h5py.File(prep_data_dir + 'prep_aug_shuffled_dataset.hdf5', 'r') as hf:
    data = hf['data'][:]
    labels = hf['labels'][:]

#------------------------------------------Distorting data -----------------------------------------------------------
def distort_imgs(data):
    """ data augumentation """
    
    data = tl.prepro.flip_axis_multi(data,
                            axis=1) # left right
    data = tl.prepro.elastic_transform_multi(data,
                            alpha=720, sigma=24, is_random=True)
    data = tl.prepro.rotation_multi(data, rg=25,
                            is_random=True, fill_mode='constant') # nearest, constant
    data = tl.prepro.shear_multi(data, 0.15,
                            is_random=True, fill_mode='constant')

    return data

#--------------------------------------Augmenting data---------------------------------------------------------------
    
# 1234 - left right up down / rg change

X_p1 = distort_imgs(X_p)
X_p2 = distort_imgs(X_p)

X_m2 = distort_imgs(X_m)
X_m3 = distort_imgs(X_m)
X_m4 = distort_imgs(X_m)

X = np.concatenate([X_p, X_p1, X_p2], axis=0)

#-----------------Saving data-----------------------
prep_data_dir = "S:\\ML+AI tutorials\\Projects\\Tumour_Classification\\Datasets\\preprocessed_data\\"
with h5py.File(prep_data_dir + 'prep_aug_dataset-2.hdf5', 'a') as hf:
#    hf.create_dataset("glioma",  data=X)
#    hf.create_dataset("meningioma",  data=X)
    print(list(hf.keys()))
#    hf.create_dataset("pituitary_tumor",  data=X)

#----------------------Load data----------------------------
prep_data_dir = "S:\\ML+AI tutorials\\Projects\\Tumour_Classification\\Datasets\\preprocessed_data\\"
with h5py.File(prep_data_dir + 'prep_aug_dataset.hdf5', 'r') as hf:
    X_g = hf['glioma'][:]
    X_m = hf['meningioma'][:]
    X_p = hf['pituitary_tumor'][:]


# --------------------------Extracting images for paper-------------------------
plt.imsave(img_dir+'pituitary_tumor_o.png', x_p[500,:,:], cmap='gray')
plt.imsave(img_dir+'pituitary_tumor_m.png', mask_p[500,:,:], cmap='gray')
plt.imsave(img_dir+'glioma_m.png', m_g, cmap='gray')
plt.imsave(img_dir+'meningioma_m.png', m_m, cmap='gray')

plt.imsave(img_dir+'a4.png', a4[0,:,:,0], cmap='gray')


x_g = cv2.imread(patients_g[500][0], 0)
m_g = cv2.imread(masks_g[500][0], 0)
x_g = cv2.cvtColor(x_g, cv2.COLOR_GRAY2BGR)
g, ctr, _ = cv2.findContours(m_g, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(x_g, ctr, -1, (255,0,0), 1)


x_m = cv2.imread(patients_m[500][0], 0)
m_m = cv2.imread(masks_m[500][0], 0)
x_m = cv2.cvtColor(x_m, cv2.COLOR_GRAY2BGR)
m, ctr, _ = cv2.findContours(m_m, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(x_m, ctr, -1, (255,0,0), 2)


x_pt = np.array(x_p[500,:,:])
m_pt = np.array(mask_p[500,:,:])
x_pt = cv2.cvtColor(x_pt, cv2.COLOR_GRAY2BGR)
p, ctr, _ = cv2.findContours(m_pt, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(x_pt, ctr, -1, (255,0,0), 2)