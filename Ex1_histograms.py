import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import function as fct

path_dir = '/scratcht/fweissge/cours_FRS/'
path_optical = path_dir + 'Paris_PleiadesNeo_zoom_2.npz'
path_SAR = path_dir + 'Ville1.npz'

image_optic = np.load(path_optical, allow_pickle=True)['image']
data_SAR = np.load(path_SAR, allow_pickle=True)
image_SAR = data_SAR['image']

fig, ax = plt.subplots()
ax.imshow(image_optic)

fig, ax = plt.subplots()
ax.imshow(np.abs(image_SAR[:,:,0]), cmap = 'gray')

image_sar_thresh = fct.threshSAR(image_SAR[:,:,0])
fig, ax = plt.subplots()
ax.imshow(image_sar_thresh, cmap = 'gray')

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.angle(image_SAR[:,:,0]), cmap = 'hsv', interpolation='none')
hist_phase, bin_phase = np.histogram(np.angle(image_SAR[:,:,0]), bins=np.linspace(-np.pi, np.pi, 100))
ax[1].plot((bin_phase[:-1]+bin_phase[1:])/2, hist_phase)

#%% Pixel size
pixel_size_ground_range = data_SAR['taille_slant_range']/np.sin(data_SAR['theta']*np.pi/180)


#%% Histogram

bbox_Seine_SAR = [1260, 170, 1360, 270]
bbox_Seine_Optic = [8700, 200, 9400, 600]

bbox_trocadero_SAR = [130, 1340, 190, 1410]
bbox_trocadero_Optic = [540, 5860, 620, 5900]

fig, ax = plt.subplots()
ax.imshow(fct.threshSAR(image_SAR[:,:,0]), cmap = 'gray')
ax.plot([bbox_Seine_SAR[1], bbox_Seine_SAR[1], bbox_Seine_SAR[3], bbox_Seine_SAR[3], bbox_Seine_SAR[1]], [bbox_Seine_SAR[0], bbox_Seine_SAR[2], bbox_Seine_SAR[2], bbox_Seine_SAR[0], bbox_Seine_SAR[0]],  'b')
ax.plot([bbox_trocadero_SAR[1], bbox_trocadero_SAR[1], bbox_trocadero_SAR[3], bbox_trocadero_SAR[3], bbox_trocadero_SAR[1]], [bbox_trocadero_SAR[0], bbox_trocadero_SAR[2], bbox_trocadero_SAR[2], bbox_trocadero_SAR[0], bbox_trocadero_SAR[0]],  'r')

fig, ax = plt.subplots()
ax.imshow(image_optic)
ax.plot([bbox_Seine_Optic[1], bbox_Seine_Optic[1], bbox_Seine_Optic[3], bbox_Seine_Optic[3], bbox_Seine_Optic[1]], [bbox_Seine_Optic[0], bbox_Seine_Optic[2], bbox_Seine_Optic[2], bbox_Seine_Optic[0], bbox_Seine_Optic[0]],  'b')
ax.plot([bbox_trocadero_Optic[1], bbox_trocadero_Optic[1], bbox_trocadero_Optic[3], bbox_trocadero_Optic[3], bbox_trocadero_Optic[1]], [bbox_trocadero_Optic[0], bbox_trocadero_Optic[2], bbox_trocadero_Optic[2], bbox_trocadero_Optic[0], bbox_trocadero_Optic[0]],  'r')

# histogram SAR
hist_Seine_SAR, bin_edges_Seine_SAR = np.histogram(np.abs(image_SAR[bbox_Seine_SAR[0]:bbox_Seine_SAR[2], bbox_Seine_SAR[1]:bbox_Seine_SAR[3],0]).flatten(), bins=70)

hist_Trocadero_SAR, bin_edges_Trocadero_SAR = np.histogram(np.abs(image_SAR[bbox_trocadero_SAR[0]:bbox_trocadero_SAR[2], bbox_trocadero_SAR[1]:bbox_trocadero_SAR[3],0]).flatten(), bins=70)

# histogram Optic
hist_Seine_Optic_R, bin_edges_Seine_Optic_R = np.histogram(image_optic[bbox_Seine_Optic[0]:bbox_Seine_Optic[2], bbox_Seine_Optic[1]:bbox_Seine_Optic[3],0].flatten(), bins=np.arange(255))
hist_Seine_Optic_G, bin_edges_Seine_Optic_G = np.histogram(image_optic[bbox_Seine_Optic[0]:bbox_Seine_Optic[2], bbox_Seine_Optic[1]:bbox_Seine_Optic[3],1].flatten(), bins=bin_edges_Seine_Optic_R)
hist_Seine_Optic_B, bin_edges_Seine_Optic_B = np.histogram(image_optic[bbox_Seine_Optic[0]:bbox_Seine_Optic[2], bbox_Seine_Optic[1]:bbox_Seine_Optic[3],2].flatten(), bins=bin_edges_Seine_Optic_R)


hist_Trocadero_Optic_R, bin_edges_Trocadero_Optic_R = np.histogram(image_optic[bbox_trocadero_Optic[0]:bbox_trocadero_Optic[2], bbox_trocadero_Optic[1]:bbox_trocadero_Optic[3],0].flatten(), bins=np.arange(255))
hist_Trocadero_Optic_G, bin_edges_Trocadero_Optic_G = np.histogram(image_optic[bbox_trocadero_Optic[0]:bbox_trocadero_Optic[2], bbox_trocadero_Optic[1]:bbox_trocadero_Optic[3],1].flatten(), bins=bin_edges_Trocadero_Optic_R)
hist_Trocadero_Optic_B, bin_edges_Trocadero_Optic_B = np.histogram(image_optic[bbox_trocadero_Optic[0]:bbox_trocadero_Optic[2], bbox_trocadero_Optic[1]:bbox_trocadero_Optic[3],2].flatten(), bins=bin_edges_Trocadero_Optic_R)


fig, ax = plt.subplots(2,2)
ax[0,0].imshow(image_sar_thresh[bbox_Seine_SAR[0]:bbox_Seine_SAR[2], bbox_Seine_SAR[1]:bbox_Seine_SAR[3]], cmap='gray')
ax[0,1].plot((bin_edges_Seine_SAR[:-1]+bin_edges_Seine_SAR[1:])/2, hist_Seine_SAR)
ax[1,0].imshow(image_sar_thresh[bbox_trocadero_SAR[0]:bbox_trocadero_SAR[2], bbox_trocadero_SAR[1]:bbox_trocadero_SAR[3]], cmap='gray')
ax[1,1].plot((bin_edges_Trocadero_SAR[:-1]+bin_edges_Trocadero_SAR[1:])/2, hist_Trocadero_SAR)


fig, ax = plt.subplots(2,2)
ax[0,0].imshow(image_optic[bbox_Seine_Optic[0]:bbox_Seine_Optic[2], bbox_Seine_Optic[1]:bbox_Seine_Optic[3],:])
ax[0,1].plot((bin_edges_Seine_Optic_R[:-1]+bin_edges_Seine_Optic_R[1:])/2, hist_Seine_Optic_R, 'r')
ax[0,1].plot((bin_edges_Seine_Optic_R[:-1]+bin_edges_Seine_Optic_R[1:])/2, hist_Seine_Optic_G, 'g')
ax[0,1].plot((bin_edges_Seine_Optic_R[:-1]+bin_edges_Seine_Optic_R[1:])/2, hist_Seine_Optic_B, 'b')
ax[1,0].imshow(image_optic[bbox_trocadero_Optic[0]:bbox_trocadero_Optic[2], bbox_trocadero_Optic[1]:bbox_trocadero_Optic[3],:])
ax[1,1].plot((bin_edges_Trocadero_Optic_R[:-1]+bin_edges_Trocadero_Optic_R[1:])/2, hist_Trocadero_Optic_R, 'r')
ax[1,1].plot((bin_edges_Trocadero_Optic_R[:-1]+bin_edges_Trocadero_Optic_R[1:])/2, hist_Trocadero_Optic_G, 'g')
ax[1,1].plot((bin_edges_Trocadero_Optic_R[:-1]+bin_edges_Trocadero_Optic_R[1:])/2, hist_Trocadero_Optic_B, 'b')


#%% Histogram

detection_tresh_SAR_min = 0
detection_tresh_SAR_max = 80
image_sar_detection = (np.abs(image_SAR[:,:,0])>detection_tresh_SAR_min) & (np.abs(image_SAR[:,:,0])<detection_tresh_SAR_max)

fig, ax = plt.subplots()
ax.imshow(image_sar_detection)

detection_thresh_R_min = 70
detection_thresh_R_max = 110
detection_thresh_G_min = 75
detection_thresh_G_max = 115
detection_thresh_B_min = 50
detection_thresh_B_max = 100
image_optic_detection = ((image_optic[:,:,0]>detection_thresh_R_min) & (image_optic[:,:,0]<detection_thresh_R_max)) & ((image_optic[:,:,1]>detection_thresh_G_min) & (image_optic[:,:,1]<detection_thresh_G_max)) & ((image_optic[:,:,2]>detection_thresh_B_min) & (image_optic[:,:,2]<detection_thresh_B_max))

fig, ax = plt.subplots()
ax.imshow(image_optic_detection)



plt.show()
# %%
