from xtract_features.glcms import *
import scipy
import scipy.stats
import numpy as np
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


