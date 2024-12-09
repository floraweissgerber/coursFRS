import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import rasterio

path_dir = '/scratcht/fweissge/cours_FRS/'
path_optical = path_dir + 'Paris_PleiadesNeo.jpg'

dataset_optic = rasterio.open(path_optical)
image_optic = dataset_optic.read().transpose(1,2,0)
x_max = 20000-5000
x_min = 5000-3000
y_max = 22500 + 2000
y_min = 10000 + 3000
image_optic_zoom = image_optic[y_min:y_max, x_min:x_max,:]

fig, ax = plt.subplots()
ax.imshow(image_optic_zoom)

np.savez(path_dir + 'Paris_PleiadesNeo_zoom_2', image = image_optic_zoom)

plt.show()