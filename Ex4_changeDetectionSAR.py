import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import function as fct

def ChangeDetectionRatio(image_1, image_2, threhshold):

    ratio = np.maximum(image_1/image_2, image_2/image_1)
    change_detection_map = ratio>threhshold
    return change_detection_map


path_dir = '/scratcht/fweissge/cours_FRS/'
path_SAR = path_dir + 'Ville1.npz'
data_SAR = np.load(path_SAR, allow_pickle=True)
image_SAR = data_SAR['image']

ind_image_1 = 1
ind_image_2 = 3
image_RGB = fct.imCompareSameDynamicMax(image_SAR[:,:,ind_image_1], image_SAR[:,:,ind_image_2])
fig, ax = plt.subplots()
ax.imshow(image_RGB)

exp_1 = 0.5
exp_2 = 1
exp_3 = 2
change_detection_map_1 = ChangeDetectionRatio(image_SAR[:,:,ind_image_1], image_SAR[:,:,ind_image_2], np.power(10,exp_1))
change_detection_map_2 = ChangeDetectionRatio(image_SAR[:,:,ind_image_1], image_SAR[:,:,ind_image_2], np.power(10,exp_2))
change_detection_map_3 = ChangeDetectionRatio(image_SAR[:,:,ind_image_1], image_SAR[:,:,ind_image_2], np.power(10,exp_3))
fig, ax = plt.subplots(1,3)
ax[0].imshow(change_detection_map_1)
ax[1].imshow(change_detection_map_2)
ax[2].imshow(change_detection_map_3)

zoom = [200, 1140, 400, 1400]
fig, ax = plt.subplots(1,3)
ax[0].imshow(change_detection_map_1[zoom[0]:zoom[2], zoom[1]:zoom[3]])
ax[0].set_title('threshold 10^' + str(exp_1))
ax[1].imshow(change_detection_map_2[zoom[0]:zoom[2], zoom[1]:zoom[3]])
ax[1].set_title('threshold 10^' + str(exp_2))
ax[2].imshow(change_detection_map_3[zoom[0]:zoom[2], zoom[1]:zoom[3]])
ax[2].set_title('threshold 10^' + str(exp_3))

window_1 = [3,3]
window_2 = [7,7]
window_3 = [9,9]

fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(image_SAR[:,:,0]))
ax[1].imshow(fct.threshSAR(fct.boxcarFilter(image_SAR[:,:,0], window_1)))


change_detection_map_denoised_1 = ChangeDetectionRatio(fct.boxcarFilter(image_SAR[:,:,ind_image_1], window_1), fct.boxcarFilter(image_SAR[:,:,ind_image_2], window_1), np.power(10,exp_1))
change_detection_map_denoised_2 = ChangeDetectionRatio(fct.boxcarFilter(image_SAR[:,:,ind_image_1], window_2), fct.boxcarFilter(image_SAR[:,:,ind_image_2], window_2), np.power(10,exp_1))
change_detection_map_denoised_3 = ChangeDetectionRatio(fct.boxcarFilter(image_SAR[:,:,ind_image_1], window_3), fct.boxcarFilter(image_SAR[:,:,ind_image_2], window_3), np.power(10,exp_1))

fig, ax = plt.subplots(1,3)
ax[0].imshow(change_detection_map_denoised_1[zoom[0]:zoom[2], zoom[1]:zoom[3]])
ax[0].set_title('window ' + str(window_1))
ax[1].imshow(change_detection_map_denoised_2[zoom[0]:zoom[2], zoom[1]:zoom[3]])
ax[1].set_title('window ' + str(window_2))
ax[2].imshow(change_detection_map_denoised_3[zoom[0]:zoom[2], zoom[1]:zoom[3]])
ax[2].set_title('window ' + str(window_3))

delta_n_1 = 130
delta_n_2 = 100

vs = (2*np.pi*(data_SAR['H']/1000+6371))/(1.5)
R = data_SAR['H']*np.tan(data_SAR['theta']*np.pi/180)


v_1 = delta_n_1*data_SAR['taille_pixel_azimut']*vs/R
print('v1 (km/h)', v_1)
v_2 = delta_n_2*data_SAR['taille_pixel_azimut']*vs/R
print('v2 (km/h)', v_2)

plt.show()