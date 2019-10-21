from skimage.measure import label, regionprops
from skimage.feature.texture import greycomatrix
from skimage.feature.texture import greycoprops
from xtract_features.glcms import *
from skimage import feature
from skimage.filters.rank import entropy
from skimage.morphology import disk
import scipy
from scipy import ndimage

import scipy.stats
import glob as glob
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def getClassAcc(cm, categories):
	num_classes = cm.shape[0]
	for i in range(num_classes):
		cm_subset = cm[i,:]
		acc = cm[i,i] / np.sum(cm[i,:])
		print(categories[i] + ' accuracy: ' + str(acc))

def findClass(label, class_label, labelint):
	inst_map = label == labelint
	inst_map = inst_map.astype('uint8')
	M = cv2.moments(inst_map)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	return int(class_label[cY, cX])

def get_bbox(mask):
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    return x1,x2,y1,y2

def nuc_stats_new(mask,intensity):

    mask_crop = mask
    intensity_crop = intensity

    intensity_instance_pixel = np.array(intensity_crop[mask_crop > 0])
    intensity_background_pixel = np.array(intensity_crop[mask_crop == 0])

    mean_instance_intensity = intensity_instance_pixel.sum() / \
                        (np.size(intensity_instance_pixel) + 1.0e-8)
    mean_background_intensity = intensity_background_pixel.sum() / \
                        (np.size(intensity_background_pixel) + 1.0e-8)
    diff = abs(mean_instance_intensity - mean_background_intensity)
    var = np.var(intensity_instance_pixel)
    skew = scipy.stats.skew(intensity_instance_pixel)
    return mean_instance_intensity, diff, var, skew

#####
def nuc_stats(mask_crop, intensity_crop, entropy_crop):
	# mask = mask
	list_intensity = []
	list_intensity2 = []
	list_entropy = []
	# x1,x2,y1,y2 = get_bbox(mask)
	# mask_crop = mask[y1:y2, x1:x2]
	# intensity_crop = intensity[y1:y2, x1:x2]
	# entropy_crop = entropy_[y1:y2, x1:x2]
	for i in range(mask_crop.shape[0]):
		for j in range(mask_crop.shape[1]):
			if mask_crop[i,j] == 1:
				list_intensity.append(intensity_crop[i,j])
				list_entropy.append(entropy_crop[i,j])
			else:
				list_intensity2.append(intensity_crop[i,j])
	list_intensity = np.asarray(list_intensity)
	list_intensity2 = np.asarray(list_intensity2)
	mean = np.mean(list_intensity)
	if np.sum(list_intensity2) > 0:
		mean_out = np.mean(list_intensity2)
	else:
		mean_out = 0
	diff = max(mean-mean_out, mean_out-mean)
	# var = np.var(list_intensity)
	skew = scipy.stats.skew(list_intensity)
	# glcm feature
	glcm = greycomatrix(intensity_crop*mask_crop, [1], [0])
	filt_glcm = glcm[1:, 1:, :, :]
	glcm_contrast = greycoprops(filt_glcm, prop='contrast')
	glcm_contrast = glcm_contrast[0,0]
	return mean_out, diff, skew, glcm_contrast

def nuc_glcm_stats(mask, intensity):
	list_intensity = []
	x1,x2,y1,y2 = get_bbox(mask)
	mask_crop = mask[y1:y2, x1:x2]
	intensity_crop = intensity[y1:y2, x1:x2]
	for i in range(mask_crop.shape[0]):
		for j in range(mask_crop.shape[1]):
			if mask_crop[i,j] == 1:
				list_intensity.append(intensity_crop[i,j])
	list_intensity = np.asarray(list_intensity)

	glcm = greycomatrix(intensity*mask, [1], [0])  # Calculate the GLCM "one pixel to the right"
	filt_glcm = glcm[1:, 1:, :, :]  # Filter out the first row and column
	glcm_contrast = greycoprops(filt_glcm, prop='contrast')
	glcm_contrast = glcm_contrast[0,0]
	glcm_dissimilarity = greycoprops(filt_glcm, prop='dissimilarity')
	glcm_dissimilarity = glcm_dissimilarity[0,0]
	glcm_homogeneity = greycoprops(filt_glcm, prop='homogeneity')
	glcm_homogeneity = glcm_homogeneity[0,0]
	glcm_energy = greycoprops(filt_glcm, prop='energy')
	glcm_energy = glcm_energy[0,0]
	glcm_ASM = greycoprops(filt_glcm, prop='ASM')
	glcm_ASM = glcm_ASM[0,0]

	return glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM

def nuc_glcm_stats_new(mask, intensity):

    mask_crop = mask
    intensity_crop = intensity

    # Calculate the GLCM "one pixel to the right"
    glcm = greycomatrix(intensity_crop * mask_crop, [1], [0])
    filt_glcm = glcm[1:, 1:, :, :]  # Filter out the first row and column
    glcm_contrast = greycoprops(filt_glcm, prop='contrast')
    glcm_contrast = glcm_contrast[0,0]
    glcm_dissimilarity = greycoprops(filt_glcm, prop='dissimilarity')
    glcm_dissimilarity = glcm_dissimilarity[0,0]
    glcm_homogeneity = greycoprops(filt_glcm, prop='homogeneity')
    glcm_homogeneity = glcm_homogeneity[0,0]
    glcm_energy = greycoprops(filt_glcm, prop='energy')
    glcm_energy = glcm_energy[0,0]
    glcm_ASM = greycoprops(filt_glcm, prop='ASM')
    glcm_ASM = glcm_ASM[0,0]

    return glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM




def get_data(path_im, path_lab):
	list_ims = glob.glob(path_im + '*')
	count = 0
	for i in range(len(list_ims)):
		image_file = list_ims[i]
		basename = os.path.basename(image_file)
		basename = basename.split('.')[0]
		label_file = path_lab + basename + '.npy'
		label_file = np.load(label_file)
		label = label_file[...,0]
		class_label = label_file[...,1]
		image = cv2.imread(image_file)
		intensity = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		entropy_ = entropy(intensity, disk(3))
		binary_lab = class_label.copy()
		binary_lab[binary_lab>0]=1
		label = label*binary_lab
		label = label.astype('int32')
		rprops = regionprops(label, intensity)
		for props in rprops:
			nuc_feats = []
			labelint = props.label
			mask = label==labelint
			nuc_class = findClass(label, class_label, labelint)
			mean_im_out, diff, skew_im = nuc_stats(mask, intensity, entropy_)

			# glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = nuc_glcm_stats(mask, intensity)
			#nuc_feats.append(mean_im)
			nuc_feats.append(mean_im_out)#
			nuc_feats.append(diff)#

			nuc_feats.append(skew_im)#

			# nuc_feats.append(glcm_contrast)#

			nuc_feats.append(props.convex_area)#
			nuc_feats.append(props.eccentricity)#
			# nuc_feats.append(props.equivalent_diameter)
			# nuc_feats.append(props.extent)
			# nuc_feats.append(props.filled_area)
			nuc_feats.append(props.major_axis_length)#
			# nuc_feats.append(props.max_intensity)
			nuc_feats.append(props.mean_intensity)#
			nuc_feats.append(props.min_intensity)#
			nuc_feats.append(props.minor_axis_length)#
			nuc_feats.append(props.perimeter)#
			# nuc_feats.append(props.solidity)
			# nuc_feats.append(props.orientation)
			# nuc_feats.append(nuc_class)
			# nuc_feats = np.asarray(nuc_feats)
			if count == 0:
				nuc_feats_all = nuc_feats
			else:
				nuc_feats_all = np.vstack((nuc_feats_all, nuc_feats))
			count +=1

	nuc_feats_all = pd.DataFrame(nuc_feats_all,
								 columns=['mean_out',
										  'diff',
										  'var',
										  'skew',
										  'mean_ent',
										  'glcm_contrast',
										  'glcm_dissimilarity',
										  'glcm_homogeneity',
										  'glcm_energy',
										  'glcm_ASM',
										  'convex_area',
										  'eccentricity',
										  'equivalent_diameter',
										  'extent',
										  'filled_area',
											'major_axis_length',
										  'max_intensity',
										  'mean_intensity',
										  'min_intensity',
										  'minor_axis_length',
										  'perimeter',
										  'solidity',
										  'orientation',
										  'class'])
	nuc_feats_all = nuc_feats_all[nuc_feats_all['class'] != 0]
	feats = nuc_feats_all.iloc[:, :-1]
	labs = nuc_feats_all.iloc[:,-1]

	return feats, labs
