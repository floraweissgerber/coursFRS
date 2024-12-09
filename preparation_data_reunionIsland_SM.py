import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import function as fct
import rasterio

path_dir = '/scratcht/fweissge/cours_FRS/'
dir_SM = 'S1A_S6_SLC__1SDV_20241115T145331_20241115T145355_056566_06EFB9_7330.SAFE'

path_image_SM = path_dir + dir_SM + '/measurement/' + 's1a-s6-slc-vv-20241115t145331-20241115t145355-056566-06efb9-002.tiff'
#image_SM = ''

dataset_SM= rasterio.open(path_image_SM)
image_SM = dataset_SM.read().transpose(1,2,0)

fig, ax = plt.subplots()
ax.imshow(fct.threshSAR(image_SM))


np.savez('/scratcht/fweissge/cours_FRS/reunion_island_SM', image = image_SM[6000:22500, :8000])

plt.show()