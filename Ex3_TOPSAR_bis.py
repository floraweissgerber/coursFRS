
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import function as fct
import rasterio
import scipy 
import matplotlib.cm as cm

import sarimages.sarimages.sar_display as sard

path_dir = '/scratcht/fweissge/cours_FRS/'
dir_IW_1 = '/S1A_IW_SLC__1SDV_20241116T014710_20241116T014739_056573_06EFF8_2171.SAFE/'
dir_IW_2 = '/S1A_IW_SLC__1SDV_20241116T014737_20241116T014805_056573_06EFF8_A62A.SAFE/'

path_image_IW_S2_1 = path_dir + dir_IW_1 + '/measurement/' + 's1a-iw1-slc-vv-20241116t014710-20241116t014737-056573-06eff8-004.tiff'
path_image_IW_S2_2 = path_dir + dir_IW_2 + '/measurement/' + 's1a-iw2-slc-vv-20241116t014738-20241116t014803-056573-06eff8-005.tiff'



#bbox_zoom_IW_phase = [2400, 3450, 3000, 3850]
bbox_zoom_IW_phase = [10800, 1675, 10800, 1675]
bbox_zoom_SM_phase = [3000, 2250, 3600, 2650]
bbox_zoom_IW = [14600, 4750, 14650, 4800]
#bbox_zoom_IW = [4250, 5000, 4350, 5040]
window_x = 50
window_y = 50
offset = 15

dataset_IW_S2_1 = rasterio.open(path_image_IW_S2_2)
image_IW = dataset_IW_S2_1.read().transpose(1,2,0)

fig, ax = plt.subplots()
ax.imshow(fct.threshSAR(image_IW), cmap='gray')


#%% Image spectre 1

for ind_test in range(5):
    test = ind_test*100

    fig, ax = plt.subplots(3,3)

    zoom_1 = image_IW[bbox_zoom_IW_phase[1]+test:bbox_zoom_IW_phase[1]+window_y+test, bbox_zoom_IW_phase[0]:bbox_zoom_IW_phase[0]+window_x]
    ax[0,0].imshow(fct.threshSAR(zoom_1), cmap='gray')
    ax[0,1].imshow(np.angle(zoom_1), cmap='hsv', interpolation = 'none')
    ax[0,2].imshow(np.log10(np.abs(scipy.fft.fftshift(scipy.fft.fft2(zoom_1)))), cmap=cm.Reds)

    zoom_2 = image_IW[bbox_zoom_IW_phase[1]+offset+test:bbox_zoom_IW_phase[1]+window_y+offset+test, bbox_zoom_IW_phase[0]:bbox_zoom_IW_phase[0]+window_x]
    ax[1,0].imshow(fct.threshSAR(zoom_2), cmap='gray')
    ax[1,1].imshow(np.angle(zoom_2), cmap='hsv', interpolation = 'none')
    ax[1,2].imshow(np.log10(np.abs(scipy.fft.fftshift(scipy.fft.fft2(zoom_2)))), cmap=cm.Reds)

    zoom_3 = image_IW[bbox_zoom_IW_phase[1]+2*offset+test:bbox_zoom_IW_phase[1]+window_y+2*+offset+test, bbox_zoom_IW_phase[0]:bbox_zoom_IW_phase[0]+window_x]
    ax[2,0].imshow(fct.threshSAR(zoom_3), cmap='gray')
    ax[2,1].imshow(np.angle(zoom_3), cmap='hsv', interpolation = 'none')
    ax[2,2].imshow(np.log10(np.abs(scipy.fft.fftshift(scipy.fft.fft2(zoom_3)))), cmap=cm.Reds)



#%%% 

file_IW = '/scratcht/SITEMSA/SAR/expe_msi_33XXH/S1A_IW_SLC__1SDH_20220520T150441_20220520T150508_043296_052B9D_1EF2.SAFE/measurement/s1a-iw1-slc-hh-20220520t150443-20220520t150508-043296-052b9d-001.tiff'
sar_ds = rasterio.open(file_IW)
sar_IW = sar_ds.read(1)
bbox_SAR_tot = [ 8200, 12000, 8700, 12800]

bbox_SAR = [ 8400, 12100, 8450, 12150]
vec_offset = [0,15,30]

fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].imshow(sard.threshSAR(sar_IW[bbox_SAR_tot[0]:bbox_SAR_tot[2], bbox_SAR_tot[1]:bbox_SAR_tot[3]]), cmap = 'gray')
spectrum = np.abs(scipy.fft.fft2(sar_IW[bbox_SAR_tot[0]:bbox_SAR_tot[2], bbox_SAR_tot[1]:bbox_SAR_tot[3]]))
ax[0].set_title('image')
ax[1].imshow(scipy.fft.fftshift(np.log10(spectrum)), cmap = cm.Reds)  
ax[1].set_title('spectrum fftshift (dB)')
ax[1].set_xticks(ticks=[0, (bbox_SAR_tot[3]-bbox_SAR_tot[1])/2,(bbox_SAR_tot[3]-bbox_SAR_tot[1])], labels=[-(bbox_SAR_tot[3]-bbox_SAR_tot[1])/2, 0, (bbox_SAR_tot[3]-bbox_SAR_tot[1])/2])
ax[1].set_yticks(ticks=[0, (bbox_SAR_tot[2]-bbox_SAR_tot[0])/2,(bbox_SAR_tot[2]-bbox_SAR_tot[0])], labels=[-(bbox_SAR_tot[2]-bbox_SAR_tot[0])/2, 0, (bbox_SAR_tot[2]-bbox_SAR_tot[0])/2])
ax[0].plot([bbox_SAR[1]-bbox_SAR_tot[1], bbox_SAR[1]-bbox_SAR_tot[1], bbox_SAR[3]-bbox_SAR_tot[1], bbox_SAR[3]-bbox_SAR_tot[1], bbox_SAR[1]-bbox_SAR_tot[1]], [bbox_SAR[0]-bbox_SAR_tot[0],bbox_SAR[2]-bbox_SAR_tot[0],bbox_SAR[2]-bbox_SAR_tot[0], bbox_SAR[0]-bbox_SAR_tot[0], bbox_SAR[0]-bbox_SAR_tot[0]])
ax[0].plot([bbox_SAR[1]-bbox_SAR_tot[1], bbox_SAR[1]-bbox_SAR_tot[1], bbox_SAR[3]-bbox_SAR_tot[1], bbox_SAR[3]-bbox_SAR_tot[1], bbox_SAR[1]-bbox_SAR_tot[1]], [bbox_SAR[0]-bbox_SAR_tot[0]+vec_offset[1],bbox_SAR[2]-bbox_SAR_tot[0]+vec_offset[1],bbox_SAR[2]-bbox_SAR_tot[0]+vec_offset[1], bbox_SAR[0]-bbox_SAR_tot[0]+vec_offset[1], bbox_SAR[0]-bbox_SAR_tot[0]+vec_offset[1]])
ax[0].plot([bbox_SAR[1]-bbox_SAR_tot[1], bbox_SAR[1]-bbox_SAR_tot[1], bbox_SAR[3]-bbox_SAR_tot[1], bbox_SAR[3]-bbox_SAR_tot[1], bbox_SAR[1]-bbox_SAR_tot[1]], [bbox_SAR[0]-bbox_SAR_tot[0]+vec_offset[2],bbox_SAR[2]-bbox_SAR_tot[0]+vec_offset[2],bbox_SAR[2]-bbox_SAR_tot[0]+vec_offset[2], bbox_SAR[0]-bbox_SAR_tot[0]+vec_offset[2], bbox_SAR[0]-bbox_SAR_tot[0]+vec_offset[2]])
          
fig, ax = plt.subplots(3,2, figsize=(5,10))
ax[0,0].imshow(sard.threshSAR(sar_IW[bbox_SAR[0]:bbox_SAR[2], bbox_SAR[1]:bbox_SAR[3]]), cmap = 'gray')
spectrum = np.abs(scipy.fft.fft2(sar_IW[bbox_SAR[0]:bbox_SAR[2], bbox_SAR[1]:bbox_SAR[3]]))
ax[0,0].set_title('image')
ax[0,1].imshow(scipy.fft.fftshift(np.log10(spectrum)), cmap = cm.Reds) 
ax[0,1].set_title('spectrum fftshift (dB)')
ax[0,1].set_xticks(ticks=[0, (bbox_SAR[3]-bbox_SAR[1])/2,(bbox_SAR[3]-bbox_SAR[1])], labels=[-(bbox_SAR[3]-bbox_SAR[1])/2, 0, (bbox_SAR[3]-bbox_SAR[1])/2])
ax[0,1].set_yticks(ticks=[0, (bbox_SAR[2]-bbox_SAR[0])/2,(bbox_SAR[2]-bbox_SAR[0])], labels=[-(bbox_SAR[2]-bbox_SAR[0])/2, 0, (bbox_SAR[2]-bbox_SAR[0])/2])

offset_x = vec_offset[1]
ax[1,0].imshow(sard.threshSAR(sar_IW[offset_x+bbox_SAR[0]:offset_x+bbox_SAR[2], bbox_SAR[1]:bbox_SAR[3]]), cmap = 'gray')
spectrum = np.abs(scipy.fft.fft2(sar_IW[offset_x+bbox_SAR[0]:offset_x+bbox_SAR[2], bbox_SAR[1]:bbox_SAR[3]]))
ax[1,0].set_title('image')
ax[1,1].imshow(scipy.fft.fftshift(np.log10(spectrum)), cmap = cm.Reds) 
ax[1,1].set_title('spectrum fftshift (dB)')
ax[1,1].set_xticks(ticks=[0, (bbox_SAR[3]-bbox_SAR[1])/2,(bbox_SAR[3]-bbox_SAR[1])], labels=[-(bbox_SAR[3]-bbox_SAR[1])/2, 0, (bbox_SAR[3]-bbox_SAR[1])/2])
ax[1,1].set_yticks(ticks=[0, (bbox_SAR[2]-bbox_SAR[0])/2,(bbox_SAR[2]-bbox_SAR[0])], labels=[-(bbox_SAR[2]-bbox_SAR[0])/2, 0, (bbox_SAR[2]-bbox_SAR[0])/2])

offset_x = vec_offset[2]
ax[2,0].imshow(sard.threshSAR(sar_IW[offset_x+bbox_SAR[0]:offset_x+bbox_SAR[2], bbox_SAR[1]:bbox_SAR[3]]), cmap = 'gray')
spectrum = np.abs(scipy.fft.fft2(sar_IW[offset_x+bbox_SAR[0]:offset_x+bbox_SAR[2], bbox_SAR[1]:bbox_SAR[3]]))
ax[2,0].set_title('image')
ax[2,1].imshow(scipy.fft.fftshift(np.log10(spectrum)), cmap = cm.Reds) 
ax[2,1].set_title('spectrum fftshift (dB)')
ax[2,1].set_xticks(ticks=[0, (bbox_SAR[3]-bbox_SAR[1])/2,(bbox_SAR[3]-bbox_SAR[1])], labels=[-(bbox_SAR[3]-bbox_SAR[1])/2, 0, (bbox_SAR[3]-bbox_SAR[1])/2])
ax[2,1].set_yticks(ticks=[0, (bbox_SAR[2]-bbox_SAR[0])/2,(bbox_SAR[2]-bbox_SAR[0])], labels=[-(bbox_SAR[2]-bbox_SAR[0])/2, 0, (bbox_SAR[2]-bbox_SAR[0])/2])




plt.show()