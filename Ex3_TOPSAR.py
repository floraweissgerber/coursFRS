
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import function as fct
from scipy.interpolate import RegularGridInterpolator

path_dir = '/scratcht/fweissge/cours_FRS/'
path_image_IW = path_dir + 'reunion_island_IW_ramping_deramping.npz'
path_image_SM = path_dir + 'reunion_island_SM.npz'

data_IW = np.load(path_image_IW, allow_pickle=True)
image_IW = data_IW['image_IW']
image_IW_deramp = data_IW['image_IW_deramp']
image_IW_reramp = data_IW['image_IW_reramp']

data_SM = np.load(path_image_SM, allow_pickle=True)
image_SM = data_SM['image']


fig, ax = plt.subplots(1,3)
ax[0].imshow(fct.threshSAR(image_IW))
ax[1].imshow(np.angle(image_IW), cmap = 'hsv', interpolation='None')
ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(image_IW))))


fig, ax = plt.subplots(1,3)
ax[0].imshow(fct.threshSAR(image_SM))
ax[1].imshow(np.angle(image_SM), cmap = 'hsv', interpolation='None')
ax[2].imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(image_SM)))))

im_phase_ramping = image_IW_deramp*np.conj(image_IW_reramp)
fig, ax = plt.subplots()
ax.imshow(np.angle(im_phase_ramping), cmap = 'hsv', interpolation='None')

fig, ax = plt.subplots()
ax.plot(np.unwrap(np.angle(im_phase_ramping[:,0])))
ax.plot(np.unwrap(np.angle(im_phase_ramping[:,500])))
ax.plot(np.unwrap(np.angle(im_phase_ramping[:,1000])))


'''
factor_x = 0
factor_y = 2
image_SM_surech = fct.oversampling_linear(image_SM, factor_x, factor_y, 'azimuth')

fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(image_SM_surech))
ax[1].imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(image_SM_surech)))))
'''


window = [500, 1000, 550, 1050]
line_offset = 25

fig, ax = plt.subplots(3,2)

vignette = image_IW[window[1]:window[3], window[0]: window[2]]
ax[0,0].imshow(fct.threshSAR(vignette))
ax[0,1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(vignette))))

vignette = image_IW[window[1]+line_offset:window[3]+line_offset, window[0]: window[2]]
ax[1,0].imshow(fct.threshSAR(vignette))
ax[1,1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(vignette))))

vignette = image_IW[window[1]+2*line_offset:window[3]+2*line_offset, window[0]: window[2]]
ax[2,0].imshow(fct.threshSAR(vignette))
ax[2,1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(vignette))))


fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(image_IW_deramp))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(image_IW_deramp))))

fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(image_IW_reramp))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(image_IW_reramp))))


'''
factor_x = 0
factor_y = 12
image_IM_surech = fct.oversampling_linear(image_IW, factor_x, factor_y, 'azimuth')
fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(image_IM_surech))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(image_IM_surech))))


fig, ax = plt.subplots(3,2)

vignette = image_SM[window[1]:window[3], window[0]: window[2]]
ax[0,0].imshow(fct.threshSAR(vignette))
ax[0,1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(vignette))))

vignette = image_SM[window[1]+line_offset:window[3]+line_offset, window[0]: window[2]]
ax[1,0].imshow(fct.threshSAR(vignette))
ax[1,1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(vignette))))

vignette = image_SM[window[1]+2*line_offset:window[3]+2*line_offset, window[0]: window[2]]
ax[2,0].imshow(fct.threshSAR(vignette))
ax[2,1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(vignette))))
'''


plt.show()