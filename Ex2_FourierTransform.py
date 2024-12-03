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

image_sar_thresh = fct.threshSAR(image_SAR[:,:,0])
fig, ax = plt.subplots()
ax.imshow(image_sar_thresh, cmap = 'gray')

spectrum_optic = np.fft.fft2(image_optic[:,:,0])
fig, ax = plt.subplots()
ax.imshow(np.log10(np.fft.fftshift(np.abs(spectrum_optic))), cmap = 'gray')

spectrum_SAR = np.fft.fft2(image_SAR[:,:,0])
fig, ax = plt.subplots()
ax.imshow(np.log10(np.fft.fftshift(np.abs(spectrum_SAR))), cmap = 'gray')

spectrum_abs_SAR = np.fft.fft2(np.abs(image_SAR[:,:,0]))
fig, ax = plt.subplots()
ax.imshow(np.log10(np.fft.fftshift(np.abs(spectrum_abs_SAR))), cmap = 'gray')

ifft_tot = np.fft.ifft2(spectrum_optic)
ifft_amplitude = np.fft.ifft2(np.abs(spectrum_optic))
ifft_phase = np.fft.ifft2(np.exp(1j*np.angle(spectrum_optic)))

display_ifft_phase = fct.applyTreshMax(np.abs(ifft_phase),(np.amax(np.abs(ifft_phase))/10))

fig, ax = plt.subplots(1,3)
ax[0].imshow(np.abs(ifft_tot), cmap='gray')
ax[1].imshow(np.log10(np.abs(ifft_amplitude)), cmap='gray')
ax[2].imshow(display_ifft_phase, cmap='gray')


shift_x = 1/5
M_x = int(shift_x*spectrum_optic.shape[0])
shift_y = 1/4
M_y = int(shift_y*spectrum_optic.shape[1])
wave_x, wave_y = np.meshgrid( np.exp(1j*2*np.pi*M_y*np.arange(spectrum_optic.shape[1])/spectrum_optic.shape[1]), np.exp(1j*2*np.pi*M_x*np.arange(spectrum_optic.shape[0])/spectrum_optic.shape[0]))

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.real(wave_x[:100,:100]), clim=[-1, 1])
ax[1].imshow(np.real(wave_y[:100,:100]), clim=[-1, 1])

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.abs(np.fft.ifft2(spectrum_optic*wave_x)), cmap='gray')
ax[1].imshow(np.abs(np.fft.ifft2(spectrum_optic*wave_y)), cmap='gray')

plt.show()