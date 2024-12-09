import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import function as fct
import rasterio

path_dir = '/scratcht/fweissge/cours_FRS/'
dir_IW_1 = '/S1A_IW_SLC__1SDV_20241116T014710_20241116T014739_056573_06EFF8_2171.SAFE/'
dir_IW_2 = '/S1A_IW_SLC__1SDV_20241116T014737_20241116T014805_056573_06EFF8_A62A.SAFE/'

path_image_IW_S2_1 = path_dir + dir_IW_1 + '/measurement/' + 's1a-iw2-slc-vv-20241116t014711-20241116t014738-056573-06eff8-005.tiff'
path_image_IW_S2_2 = path_dir + dir_IW_2 + '/measurement/' + 's1a-iw2-slc-vv-20241116t014738-20241116t014803-056573-06eff8-005.tiff'

#image_SM = ''

dataset_IW_S2_1 = rasterio.open(path_image_IW_S2_1)
image_IW_S2_1 = dataset_IW_S2_1.read().transpose(1,2,0)

dataset_IW_S2_2 = rasterio.open(path_image_IW_S2_2)
image_IW_S2_2 = dataset_IW_S2_2.read().transpose(1,2,0)


fig, ax = plt.subplots()
ax.imshow(fct.threshSAR(image_IW_S2_1))

fig, ax = plt.subplots()
ax.imshow(fct.threshSAR(image_IW_S2_2))

image_tot = np.vstack((image_IW_S2_1[11990:, :], image_IW_S2_2[:3000,:]))

fig, ax = plt.subplots()
ax.imshow(fct.threshSAR(image_tot))

#np.savez('/scratcht/fweissge/cours_FRS/reunion_island_IW', image = image_tot)

plt.show()