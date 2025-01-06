import numpy as np
import matplotlib.pyplot as plt
import rasterio.windows
plt.close('all')
import function as fct
from scipy.interpolate import RegularGridInterpolator

import rasterio
from rasterio.windows import Window

import sys
sys.path.insert(1, "/d/fweissge/LABSARV42/")
from satellites import pydecoupe_Sentinel1_V42 as SENT
from satellites import pydecoupe_TSX_V42 as TSX
from sardecoupe import pydecoupe_V42 as pydec

import sarimages.sarimages.sar_display as sard
import sarimages.sarimages.read_s1 as S1
import sarimages.sarimages.projectionSAR as projSAR


path_dir = '/scratcht/fweissge/cours_FRS/'
dir_IW_1 = '/S1A_IW_SLC__1SDV_20241116T014710_20241116T014739_056573_06EFF8_2171.SAFE/'
dir_IW_2 = '/S1A_IW_SLC__1SDV_20241116T014737_20241116T014805_056573_06EFF8_A62A.SAFE/'
file_sar_1 = path_dir + dir_IW_1
file_sar_2 = path_dir + dir_IW_2

lat_ref = -21.2430556
delta_latref =0.05
lon_ref = 55.70722222222223
alt_ref =  2632
point_ref = [lon_ref, lat_ref+delta_latref,  alt_ref]

descri_SENT_1 = SENT.charger(file_sar_2, point_ref)
Nazi = 1600
Nrange = 2000
image_1 = descri_SENT_1.lirecoordcrop(point_ref, Nrange, Nazi)

fig, ax = plt.subplots()
ax.imshow(sard.threshSAR(image_1[0]))

image_surech_deramp = descri_SENT_1.sureech_avec_deramp(image_1, 1, 12)
image_surech_reramp = descri_SENT_1.sureech_ramp(image_1, 1, 12)


fig, ax = plt.subplots(1,2)
ax[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(image_surech_deramp[0]))))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(image_surech_reramp[0]))))


bbox_zoom_IW = [2660, 42, 4660,  1542]
center = [ int((bbox_zoom_IW[1]+bbox_zoom_IW[3])/2) , int((bbox_zoom_IW[0]+bbox_zoom_IW[2])/2)]

N_x = 50
N_y = 50
zoom_1 = descri_SENT_1.lirecrop(center[1], center[0], N_x, N_y)

fig, ax = plt.subplots(1,2)
ax[0].imshow(sard.threshSAR(zoom_1[0]))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(zoom_1[0]))))

file_geotiff = file_sar_2 + '/measurement/' + 's1a-iw2-slc-vv-20241116t014738-20241116t014803-056573-06eff8-005.tiff'
dataset_geotiff = rasterio.open(file_geotiff)
window = Window(int(center[1]-N_x/2), int(center[0]-N_y/2), N_x, N_y)
zoom_r = dataset_geotiff.read(1, window=window)

fig, ax = plt.subplots(1,2)
ax[0].imshow(sard.threshSAR(zoom_r))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(zoom_r))))


window_2 = Window(14600, 4750,N_x, N_y)
zoom_r2 = dataset_geotiff.read(1, window=window_2)

fig, ax = plt.subplots(1,2)
ax[0].imshow(sard.threshSAR(zoom_r2))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(zoom_r2))))


def oversampling_linear(image_in, factor_x, factor_y, direction):

    x = np.arange(image_in.shape[1])
    y = np.arange(image_in.shape[0])
    interp_real = RegularGridInterpolator((y,x), np.real(image_in[:,:]))
    interp_imag = RegularGridInterpolator((y,x), np.imag(image_in[:,:]))

    if direction == 'azimuth':

        yy = np.arange(image_in.shape[0]-1, step = 1/factor_y)
        XX,YY = np.meshgrid(x,yy)
        test_points = np.array([YY.ravel(), XX.ravel()]).T
        im_real = interp_real(test_points, method='linear').reshape(yy.shape[0], x.shape[0])
        im_imag = interp_imag(test_points, method='linear').reshape(yy.shape[0], x.shape[0])


    elif direction == 'range':

        xx = np.arange(image_in.shape[1]-1, step = 1/factor_x)
        XX,YY = np.meshgrid(xx,y)
        test_points = np.array([YY.ravel(), XX.ravel()]).T
        im_real = interp_real(test_points, method='linear').reshape(y.shape[0], xx.shape[0])
        im_imag = interp_imag(test_points, method='linear').reshape(y.shape[0], xx.shape[0])
        
    elif direction == 'both':

        xx = np.arange(image_in.shape[1]-1, step = 1/factor_x)
        yy = np.arange(image_in.shape[0]-1, step = 1/factor_y)
        XX,YY = np.meshgrid(xx,yy)
        test_points = np.array([YY.ravel(), XX.ravel()]).T
        im_real = interp_real(test_points, method='linear').reshape(yy.shape[0], xx.shape[0])
        im_imag = interp_imag(test_points, method='linear').reshape(yy.shape[0], xx.shape[0])


    image_out = im_real + 1j*im_imag

    return image_out

window_burst = Window(bbox_zoom_IW[0], bbox_zoom_IW[1], bbox_zoom_IW[2]-bbox_zoom_IW[0], bbox_zoom_IW[3]-bbox_zoom_IW[1])
burst = dataset_geotiff.read(1, window=window_burst)


fig, ax = plt.subplots(1,2)
ax[0].imshow(sard.threshSAR(burst))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(burst))))

factor_y_SM = 4
factor_y_IW = 12
factor_x = 2
direction = 'azimuth'
zoom_phase_surech = oversampling_linear(np.squeeze(burst), factor_x, factor_y_IW, direction)

fig, ax = plt.subplots(1,2)
ax[0].imshow(sard.threshSAR(zoom_phase_surech))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(zoom_phase_surech))))


np.savez('/scratcht/fweissge/cours_FRS/reunion_island_IW_ramping_deramping.npz', image_IW= image_1[0], image_IW_deramp = image_surech_deramp[0], image_IW_reramp = image_surech_reramp[0] )

plt.show()