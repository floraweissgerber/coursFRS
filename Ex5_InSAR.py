import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import function as fct

path_dir = '/scratcht/fweissge/cours_FRS/'
path_SAR = path_dir + 'Ville1.npz'
data_SAR = np.load(path_SAR, allow_pickle=True)
image_SAR = data_SAR['image']


ind_ref = 0
ind_test = 2
interfero = fct.interfero(image_SAR[:,:,ind_test], image_SAR[:,:,ind_ref], [3,3])

fig, ax = plt.subplots(1,2)
h0 = ax[0].imshow(np.abs(interfero), cmap = 'gray')
fig.colorbar(h0, ax=ax[0])
ax[0].set_title('degree of coherence')
h1 =ax[1].imshow(np.angle(interfero), clim = [-np.pi, np.pi], cmap = 'hsv', interpolation = 'none')
fig.colorbar(h1, ax=ax[1])
ax[1].set_title('phase')

zoom_low = [40, 1300, 160, 1475]
zoom_high = [150, 400, 270, 575]


fig, ax = plt.subplots(2,2)
h0 = ax[0,0].imshow(np.abs(interfero[zoom_low[0]:zoom_low[2], zoom_low[1]:zoom_low[3]]), cmap = 'gray')
ax[1,0].imshow(np.abs(interfero[zoom_high[0]:zoom_high[2], zoom_high[1]:zoom_high[3]]), cmap = 'gray')
fig.colorbar(h0, ax=ax[0])
ax[0,0].set_title('degree of coherence')
h1 =ax[0,1].imshow(np.angle(interfero[zoom_low[0]:zoom_low[2], zoom_low[1]:zoom_low[3]]), clim = [-np.pi, np.pi], cmap = 'hsv', interpolation = 'none')
fig.colorbar(h1, ax=ax[1])
ax[0,1].set_title('phase')
ax[1,1].imshow(np.angle(interfero[zoom_high[0]:zoom_high[2], zoom_high[1]:zoom_high[3]]), cmap = 'hsv', interpolation = 'none')

thresh_1 = 0.1
thresh_2 = 0.3
thresh_3 = 0.6
fig, ax = plt.subplots(1,3)
ax[0].imshow(np.abs(interfero)<thresh_1)
ax[0].set_title('threshold ' + str(thresh_1))
ax[1].imshow(np.abs(interfero)<thresh_2)
ax[1].set_title('threshold ' + str(thresh_2))
ax[2].imshow(np.abs(interfero)<thresh_3)
ax[2].set_title('threshold ' + str(thresh_3))

kz= (4*np.pi*data_SAR['vec_baseline'][ind_test])/(data_SAR['l_onde']*data_SAR['H']*np.tan(data_SAR['theta']*np.pi/180))

y_mirabeau = 1634
x_min_mirabeau = 53
x_max_mirabeau = 160
unwrap_mirabeau = np.unwrap(np.angle(interfero[y_mirabeau, x_min_mirabeau:x_max_mirabeau]))

fig, ax = plt.subplots(1,2)
fig.suptitle('Mirabeau tower')
ax[0].plot(np.angle(interfero[y_mirabeau, x_min_mirabeau:x_max_mirabeau]))
ax[1].plot(unwrap_mirabeau)

dphi_mirabeau_unwrap = unwrap_mirabeau[-1]-unwrap_mirabeau[0]
h_mirabeau_unwrap = dphi_mirabeau_unwrap/kz
dphi_mirabeau_man = -3 #3*np.pi#-3*np.pi
h_mirabeau_man = dphi_mirabeau_man/kz



y_eiffel = 410
x_min_eiffel = 1288
x_max_eiffel = 1607
unwrap_eiffel = np.unwrap(np.angle(interfero[y_eiffel, x_min_eiffel:x_max_eiffel]))

fig, ax = plt.subplots(1,2)
fig.suptitle('Eiffel tower')
ax[0].plot(np.angle(interfero[y_eiffel, x_min_eiffel:x_max_eiffel]))
ax[1].plot(unwrap_mirabeau)

dphi_eiffel_unwrap = unwrap_eiffel[-1]-unwrap_eiffel[0]
h_eiffel_unwrap = dphi_eiffel_unwrap/kz
dphi_eiffel_man =  -5*np.pi #9*np.pi #-7*np.pi
h_eiffel_man = dphi_eiffel_man/kz

plt.show()