
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import function as fct
from scipy.interpolate import RegularGridInterpolator

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


def oversampling_ZP(image_in, factor, direction):

    if direction == 'azimuth':

        N_x = image_in.shape[1]
        N_y = image_in.shape[0]
        middle = int(factor*N_y/2)
        down = int(middle-N_y/2)
        print('N_y', N_y)
        print('down', down)

        image_out = np.zeros((factor*N_y, N_x)) + 1j*np.zeros((factor*N_y, N_x))
        image_out[down:down+N_y, :] = image_in

    return image_out



path_dir = '/scratcht/fweissge/cours_FRS/'
path_image_IW = path_dir + 'reunion_island_IW.npz'
path_image_SM = path_dir + 'reunion_island_SM.npz'

image_IW = np.load(path_image_IW, allow_pickle=True)['image']
image_SM = np.load(path_image_SM, allow_pickle=True)['image']

bbox_zoom_IW_phase = [2000, 3000, 4000, 4500]
bbox_zoom_SM_phase = [3000, 2250, 3600, 3650]
bbox_zoom_IW = [14600, 4750, 14650, 4900]
#bbox_zoom_IW = [4250, 5000, 4350, 5040]
window_x = 64
window_y = 64
offset = 32


#%% Image tot
fig, ax = plt.subplots()
ax.imshow(fct.threshSAR(image_IW), cmap='gray')
ax.plot([bbox_zoom_IW_phase[0], bbox_zoom_IW_phase[0], bbox_zoom_IW_phase[2], bbox_zoom_IW_phase[2], bbox_zoom_IW_phase[0]], [bbox_zoom_IW_phase[1], bbox_zoom_IW_phase[3], bbox_zoom_IW_phase[3], bbox_zoom_IW_phase[1], bbox_zoom_IW_phase[1]], 'r', linewidth = 3)
ax.plot([bbox_zoom_IW[0], bbox_zoom_IW[0], bbox_zoom_IW[2], bbox_zoom_IW[2], bbox_zoom_IW[0]], [bbox_zoom_IW[1], bbox_zoom_IW[3], bbox_zoom_IW[3], bbox_zoom_IW[1], bbox_zoom_IW[1]], 'g', linewidth = 3)

fig, ax = plt.subplots()
ax.imshow(fct.threshSAR(image_SM), cmap='gray')
ax.plot([bbox_zoom_SM_phase[0], bbox_zoom_SM_phase[0], bbox_zoom_SM_phase[2], bbox_zoom_SM_phase[2], bbox_zoom_SM_phase[0]], [bbox_zoom_SM_phase[1], bbox_zoom_SM_phase[3], bbox_zoom_SM_phase[3], bbox_zoom_SM_phase[1], bbox_zoom_SM_phase[1]], 'r', linewidth = 3)

#%% 
fig, ax = plt.subplots()
ax.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(image_SM)))))

#%% Image phase

zoom_phase_SM = image_SM[bbox_zoom_SM_phase[1]:bbox_zoom_SM_phase[3], bbox_zoom_SM_phase[0]:bbox_zoom_SM_phase[2]]
fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(zoom_phase_SM), cmap='gray')
ax[1].imshow(np.angle(zoom_phase_SM), cmap='hsv', interpolation = 'none')
fig.suptitle('StripMap')


zoom_phase = image_IW[bbox_zoom_IW_phase[1]:bbox_zoom_IW_phase[3], bbox_zoom_IW_phase[0]:bbox_zoom_IW_phase[2]]
fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(zoom_phase), cmap='gray')
ax[1].imshow(np.angle(zoom_phase), cmap='hsv', interpolation = 'none')
fig.suptitle('TOPSAR')



#%% Image spectre SM
factor_y_SM = 4
factor_y_IW = 12
factor_x = 2
direction = 'azimuth'
zoom_phase_SM_surech = oversampling_linear(np.squeeze(zoom_phase_SM),factor_x, factor_y_SM, direction)
zoom_phase_surech = oversampling_linear(np.squeeze(zoom_phase), factor_x, factor_y_IW, direction)

fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(zoom_phase_SM_surech), cmap='gray')
ax[1].imshow((np.abs(np.fft.fftshift(np.fft.fft2(zoom_phase_SM_surech)))), cmap='jet', interpolation = 'none')
fig.suptitle('StripMap surech')

fig, ax = plt.subplots(1,2)
ax[0].imshow(fct.threshSAR(zoom_phase_surech), cmap='gray')
ax[1].imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(zoom_phase_surech)))), cmap='jet', interpolation = 'none')
fig.suptitle('TOPSAR surech')


#%% Image spectre 1

fig, ax = plt.subplots(3,3)

zoom_1 = np.squeeze(image_IW[bbox_zoom_IW[1]:bbox_zoom_IW[3], bbox_zoom_IW[0]:bbox_zoom_IW[2]])
ax[0,0].imshow(fct.threshSAR(zoom_1), cmap='gray')
ax[0,1].imshow(np.angle(zoom_1), cmap='hsv', interpolation = 'none')
ax[0,2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(zoom_1))), cmap='jet', interpolation = 'none')

zoom_2 = np.squeeze(image_IW[bbox_zoom_IW[1]+offset:bbox_zoom_IW[3]+offset, bbox_zoom_IW[0]:bbox_zoom_IW[2]])
ax[1,0].imshow(fct.threshSAR(zoom_2), cmap='gray')
ax[1,1].imshow(np.angle(zoom_2), cmap='hsv', interpolation = 'none')
ax[1,2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(zoom_2))), cmap='jet', interpolation = 'none')

zoom_3 = np.squeeze(image_IW[bbox_zoom_IW[1]+2*offset:bbox_zoom_IW[3]+2*offset, bbox_zoom_IW[0]:bbox_zoom_IW[2]])
ax[2,0].imshow(fct.threshSAR(zoom_3), cmap='gray')
ax[2,1].imshow(np.angle(zoom_3), cmap='hsv', interpolation = 'none')
ax[2,2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(zoom_3))), cmap='jet', interpolation = 'none')




plt.show()