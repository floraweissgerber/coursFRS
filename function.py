import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
from scipy.interpolate import RegularGridInterpolator

def boxcarFilter(imIn,windowShape):
    '''
    Local estimation using the maximum likelyhood estimation of a rayleigh distribution :
    mu = sqrt(sum(abs(pixel)**2)/N) where N is the number of pixels used in the sum
    '''
    element = np.ones(windowShape) # filtering kernel, just a rectangle

    imConv = signal.convolve2d(np.abs(imIn)**2, element, mode = 'same') 
    #since the mode is 'same', the output image has the same size than the input

    denomIm = np.ones(imIn.shape) 
    denomConv = signal.convolve2d(denomIm, element, mode = 'same')
    #since the mode is 'same', the border of the image are average less than the middle 
    #this line count the number of pixels used in the averaging

    imFilter = np.sqrt(imConv/denomConv)

    return imFilter
    


def interfero(im1,im2, windowShape):
    
    element = np.ones(windowShape)
    cov = signal.convolve2d(im1*np.conj(im2), element, mode = 'same')
    denom1 = signal.convolve2d(im1*np.conj(im1), element, mode = 'same')
    denom2 = signal.convolve2d(im2*np.conj(im2), element, mode = 'same')
    
    interfero = cov/np.sqrt(denom1*denom2)
    
    return interfero

def imInterferoHSV_image(im_complex, im_tresh, thresh = [0, 1]):
    
    
    color = np.angle(im_complex)
    im_coherence = np.abs(im_complex)
    imRGB = imHSV(color, im_coherence, im_tresh, thresh = thresh,range_color = [-np.pi, np.pi], range_saturation = [0,1])
    
    return imRGB
        

def imHSV(color, saturation, value, thresh = [0, 1], **kwargs):
    range_color = kwargs.get('range_color', [np.min(color), np.max(color)])
    range_saturation = kwargs.get('range_saturation', [np.min(saturation), np.max(saturation)])
    
    if color.shape !=saturation.shape: 
        error_string = "Color image should have the same size than the Saturation image"
        raise ValueError(error_string)
    else:
        [Ny, Nx] = color.shape
        
    extend = thresh[1]-thresh[0]
    color = (color - range_color[0])/(range_color[1]-range_color[0])
    color[color<0] = 0
    color[color>1] = 1
    saturation = (saturation - range_saturation[0])/(range_saturation[1]-range_saturation[0])
    saturation[saturation<0] = 0
    saturation[saturation>1] = 1
    tresh_saturation = thresh[0]+extend*saturation
    
    mat_hsv = np.ones([Ny, Nx, 3])
    mat_hsv[:,:,0] = color
    mat_hsv[:,:,1] = tresh_saturation
    mat_hsv[:,:,2] = value
    im_rgb = mcolors.hsv_to_rgb(mat_hsv)
    
    return im_rgb


def threshSAR(im,thresh = 3, exp = 1):
    
    im, val_min = return2zeros(im)
    val_max  = computeTreshMax(im,thresh)
    im = applyTreshMax(im,val_max)
    im = im**exp
    
    return im


def threshSAR_findValues(im,thresh = 3, exp = 1):
    
    im, val_min = return2zeros(im)
    val_max  = computeTreshMax(im,thresh)
    im = applyTreshMax(im,val_max)
    im = im**exp
    
    return im, val_min, val_max, exp

def threshSAR_applyValues(im,val_min, val_max, exp):
    
    im = abs(im)
    im = applyTreshMin(im, val_min)-val_min
    im = applyTreshMax(im,val_max)
    im = im**exp
    
    return im

def return2zeros(im):
    
    im = abs(im)
    val_min = np.amin(im)
    im = im-val_min
    
    return im, val_min

def computeTreshMax(im,thresh):
    
    mean = np.mean(im)
    std = np.std(im)
    val_max = mean + thresh*std
    
    return val_max

def computeTreshMin(im,thresh):
    
    mean = np.mean(im)
    std = np.std(im)
    val_min = max(0,mean - thresh*std)
    
    return val_min
    
def applyTreshMax(im,val_max):
    
    mask=im<val_max
    im=im*mask+(1-mask)*val_max
    im = im/val_max
    
    return im

def applyTreshMin(im,val_min):
    
    mask=im>val_min
    im=im*mask+(1-mask)*val_min
    
    return im
    

def imCompareSameDynamicMax(im1,im2,threshMax=3,exp=1):
    
    im1, val_zero_1 = return2zeros(im1)
    im2, val_zero_2 = return2zeros(im2)
    
    val_max  = computeTreshMax(im1,threshMax)
    im1 = (applyTreshMax(im1,val_max))**exp
    im2 = (applyTreshMax(im2,val_max))**exp
    
    im_rgb = np.zeros((im1.shape[0],im1.shape[1],3))
    im_rgb[:,:,0] = im1
    im_rgb[:,:,1] = (im1+im2)/2
    im_rgb[:,:,2] = im2
    
    return im_rgb

def oversampling_linear(image_in, factor_x, factor_y, direction):

    x = np.arange(image_in.shape[1])
    y = np.arange(image_in.shape[0])
    interp_real = RegularGridInterpolator((y,x), np.real(image_in[:,:]))
    interp_imag = RegularGridInterpolator((y,x), np.imag(image_in[:,:]))

    if direction == 'azimuth':

        if factor_y > 0:

            yy = np.arange(image_in.shape[0]-1, step = 1/factor_y)
            XX,YY = np.meshgrid(x,yy)
            test_points = np.array([YY.ravel(), XX.ravel()]).T
            im_real = interp_real(test_points, method='linear').reshape(yy.shape[0], x.shape[0])
            im_imag = interp_imag(test_points, method='linear').reshape(yy.shape[0], x.shape[0])
        else: 
            raise ValueError('the factor_y should be superior to 0')


    elif direction == 'range':

        if factor_x > 0:
            xx = np.arange(image_in.shape[1]-1, step = 1/factor_x)
            XX,YY = np.meshgrid(xx,y)
            test_points = np.array([YY.ravel(), XX.ravel()]).T
            im_real = interp_real(test_points, method='linear').reshape(y.shape[0], xx.shape[0])
            im_imag = interp_imag(test_points, method='linear').reshape(y.shape[0], xx.shape[0])
        else: 
            raise ValueError('the factor_x should be superior to 0')
        
    elif direction == 'both':

        if (factor_x > 0)&(factor_y > 0):
            xx = np.arange(image_in.shape[1]-1, step = 1/factor_x)
            yy = np.arange(image_in.shape[0]-1, step = 1/factor_y)
            XX,YY = np.meshgrid(xx,yy)
            test_points = np.array([YY.ravel(), XX.ravel()]).T
            im_real = interp_real(test_points, method='linear').reshape(yy.shape[0], xx.shape[0])
            im_imag = interp_imag(test_points, method='linear').reshape(yy.shape[0], xx.shape[0])
        else: 
            raise ValueError('the factor_x and factor_y should be superior to 0')


    image_out = im_real + 1j*im_imag



    return image_out
