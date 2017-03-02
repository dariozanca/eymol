'''
Created on 31 ago 1948
@author: Dario Zanca 
@summary: Collection of functions to generate saliency map and attentive scanpaths with EYMOL
'''

import numpy as np
import cv2
from math import sin, cos, pi
from random import randint, uniform
from scipy.io import loadmat
from scipy.integrate import odeint

######################################################################################

def sm(	I, # input stimuly as np matrix

		  	eta=3.5*10**2, etap=2*10**4, lambd=10**-3, # model parameters
			theta=10**-3, # dissipation term (see: energy balance)

          	peripheral_sigmas = (151,301), # sigmas to compute peripheral vision as DoG

			apply_center=True, # if you would center bias input during preprocessing

			observers=199, # number of virtual observers 
			seconds=1, # time of observation for each of the observers
			initRay=75, # range around the center to initialize first fixation, set 0 to biggest ray

		    blurRadius=70, # blur parameter for sm optimization

			return_scanpath=False, # if scanpath is desired
			msgs=True # print messages

		  	):

    ''' '''

    # convert I to grayscale, if not already
    if len(np.shape(I)) > 2:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    I_original = np.copy(I)

    # rescale (960x540 max sizes)
    h, w = I.shape

    if w/960 > h/540:
        h, w = int(540*(w/960)), 960
    else:
        h, w = 540, int(960*(h/540))

    I = cv2.resize(I, (w,h), interpolation = cv2.INTER_CUBIC)

    # image pre-processing and compute retina boundaries
    b = compute_b(I)
    Ip = peripheral_preprocessing(I, b, peripheral_sigmas[0], peripheral_sigmas[1])
    I = image_preprocessing(I, b)


    # add center-bias
    if apply_center:
        center = loadmat('center.mat')['center']
        center = cv2.resize(center,( (w-b[2]-b[3]),(h-b[0]-b[1]) ))
        I[b[0]:-b[1],b[2]:-b[3]] *= center
        Ip[b[0]:-b[1],b[2]:-b[3]] *= center
    
    # normalize in (0,1)

    if not I.max() == 0:
        I /= I.max()

    if not Ip.max() == 0:
        Ip /= Ip.max()

    "Parameters of the system has to be set here"
    FPS = 300 # Frames per second
    FixPS = 4 # Fixations per second
    OMEGA = pi * (1.0 / FPS) * (FixPS) 
    parameters = {'eta': eta, 'etap': etap, 'lambda': lambd, 'theta': theta, 'kappa': 10**6, 'b': b, 'omega': OMEGA} 
    
    "Numerical method"
    time_step = 0.1
    times = np.arange(0,seconds*FPS,time_step)  
    I_shape = np.shape(I)  
    
    if msgs: print "Observing image..."

    i = 1
    
    while not ( i > observers ):
        
        # generate initial conditions
        if initRay == 0: initRay = int(min(h,w)/2)
        init_x1 = int(I_shape[0]/2) + randint(-initRay,initRay)
        init_x2 = int(I_shape[1]/2) + randint(-initRay,initRay)
        init_v1 = uniform(0.3,0.7) * ((-1)**randint(0,1))
        init_v2 = uniform(0.3,0.7) * ((-1)**randint(0,1))
        y0 = [init_x1, init_x2, init_v1,init_v2]

        if i == 1:
            y = odeint(myode, y0, times, args=(I, Ip, parameters), mxstep=1000, rtol=10**-2, atol=10**-2)
        else:
            y_plus = odeint(myode, y0, times, args=(I, Ip, parameters), mxstep=1000, rtol=10**-2, atol=10**-2)
            y = np.concatenate((y,y_plus))

        i += 1

    if msgs: print "Computing saliency map..."

    SM = sum_scanpath(I,np.arange(0,(i-1)*seconds*FPS,time_step),y)
    
    h, w = I_original.shape
    SM = cv2.resize(SM, (w,h), interpolation = cv2.INTER_CUBIC)

    "Saliency map optimization"
    SMo = np.copy(SM.astype(float))
    SMo = cv2.GaussianBlur(SMo, (10*blurRadius-1,10*blurRadius-1),blurRadius-1)
    if not SMo.max()-SMo.min() == 0:
        SMo = (SMo-SMo.min())/(SMo.max()-SMo.min())

    if return_scanpath:
        return SMo, y
    else:
        return SMo

######################################################################################

def derimm(I, P):
    
    "Function which compute the derivatives of the image I at the point P"
    s = np.shape(I)
    h = 1 # increment (number of pixels)
    
    x = int( P[0] )
    y = int( P[1] )
        
    if 2*h <= x < s[0]-2*h and 2*h <= y < s[1]-2*h:
        
        # first derivatives
        dIdx = ((2*h)**-1) * ( I[x+h, y] - I[x-h, y] )
        dIdy = ((2*h)**-1) * ( I[x, y+h] - I[x, y-h] )
        
        # secon derivatives
        dIdxdx = ( (4*h**2)**-1 ) * ( I[x+2*h,y] - 2*I[x,y] + I[x-2*h,y] )
        dIdydy = ( (4*h**2)**-1 ) * ( I[x,y+2*h] - 2*I[x,y] + I[x,y-2*h] )
        
        # mixed derivative
        dIdxdy = ( (4*h**2)**-1 ) * ( I[x+h, y+h] - I[x-h, y+h] - I[x+h, y-h] + I[x-h, y-h] )
        
        dI = (dIdx, dIdy, dIdxdx, dIdydy, dIdxdy)
        
    else:
        
        dI = (0., 0., 0., 0., 0.)

    dI = np.array(dI)

    
    return dI

######################################################################################

def myode(y, t, I, Ip, p):
     
    "Compute derivatives at P"
    
    size = I.shape 

    dI = derimm(I, y[0:2])
    
    dIdx = dI[0]
    dIdy = dI[1]
    dIdxdx = dI[2]
    dIdydy = dI[3]
    dIdxdy = dI[4]
    
    dIp = derimm(Ip, y[0:2])

    dIpdx = dIp[0]
    dIpdy = dIp[1]
    dIpdxdx = dIp[2]
    dIpdydy = dIp[3]
    dIpdxdy = dIp[4]

    OMEGA = p['omega']

    "System of differential equations"
    
    dy = [y[2],
          
          y[3],
          
          cos(OMEGA*t)**2 * 2 * p['eta'] * (dIdx*dIdxdx + dIdy*dIdxdy) + sin(OMEGA*t)**2 * 2 * p['etap'] * (dIpdx*dIpdxdx + dIpdy*dIpdxdy) 
          - 2 * p['lambda'] * (2*dIdx*dIdxdx + 2*dIdy*dIdxdy) * (y[2]**2 + y[3]**2) 
          - p['theta'] * (1 - 4 * p['lambda'] * ( dIdx**2 + dIdy**2) ) * y[2],
          
          cos(OMEGA*t)**2 * 2 * p['eta'] * (dIdy*dIdydy + dIdx*dIdxdy) + sin(OMEGA*t)**2 * 2 * p['etap'] * (dIpdy*dIpdydy + dIpdx*dIpdxdy) 
          - 2 * p['lambda'] * (2*dIdy*dIdydy + 2*dIdx*dIdxdy) * (y[2]**2 + y[3]**2) 
          - p['theta'] * (1 - 4 * p['lambda'] * ( dIdx**2 + dIdy**2) ) * y[3]          
          ]
    
    "Elastic barriers contributions"

    bx_up, bx_down, by_left, by_right = p['b']
         
    if y[0] > size[0] - bx_down:
        dy[2] = dy[2] - 2 * p['kappa'] * ( y[0] - size[0] + bx_down )
    else:
        if y[0] < bx_up:
            dy[2] = dy[2] - 2 * p['kappa'] * ( y[0] - bx_up )
            
    if y[1] > size[1] - by_right:
        dy[3] = dy[3] - 2 * p['kappa'] * ( y[1] - size [1] + by_right )
    else:
        if y[1] < by_left:
            dy[3] = dy[3] - 2 * p['kappa'] * ( y[1] - by_left )
            
    "Dividing by the quantity..."
    
    dy[2] = dy[2] * (1 - 4 * p['lambda'] * (dIdx**2 + dIdy**2))**(-1)
    dy[3] = dy[3] * (1 - 4 * p['lambda'] * (dIdx**2 + dIdy**2))**(-1)

    return dy

######################################################################################

def image_preprocessing(I, b):

    "Histogram equalization is applied to increase the global contrast"
    I = cv2.equalizeHist(I)
    
    I = I.astype(float)

    "Little regularization on the image"
    I[b[0]:-b[1],b[2]:-b[3]] = cv2.GaussianBlur(I[b[0]:-b[1],b[2]:-b[3]], (15,15),0) 
    
    return I

######################################################################################

def peripheral_preprocessing(I, b, s1, s2):

    "Histogram equalization is applied to increase the global contrast"
    I = cv2.equalizeHist(I)
    
    I = I.astype(float)
    img1 = np.copy(I)
    img2 = np.copy(I)

    "Peripheral input is computed as Difference of Gaussian"
    img1[b[0]:-b[1],b[2]:-b[3]] = cv2.GaussianBlur(I[b[0]:-b[1],b[2]:-b[3]], (s1,s1), 0)
    if not img1.max() == 0:
        img1 /= img1.max()
    
    img2[b[0]:-b[1],b[2]:-b[3]] = cv2.GaussianBlur(I[b[0]:-b[1],b[2]:-b[3]], (s2,s2), 0)
    if not img2.max() == 0:
        img2 /= img2.max()
    
    Ip = img2 - img1
        
    return Ip

######################################################################################

def cartesian_to_rowcol(cart, I):
    
    "This function converts cartesian to row/col coordinates"
    
    size = np.shape(I)
    
    rc = np.array([size[0] - cart[1],
                            cart[0],
                            - cart[3],
                            cart[2]
                            ])
    
    return rc

######################################################################################

def sum_scanpath(I, T, Y):
    
    "This function computes the saliency map of the exploration (T, Y) over I"

    "Initialize map"    
    s = np.shape(I)
    map = np.zeros(s)
    
    "Sum up time spent over each pixel"
    for i in np.arange(1,len(T),1):
        if not np.isnan(Y[i,:]).any():
            if 0 < int(Y[i,0]) < s[0] and 0 < int(Y[i,1]) < s[1]:
                if abs((abs(Y[i,2]) - abs(Y[i-1,2]))) + abs((abs(Y[i,3]) - abs(Y[i-1,3]))) > 10**-3: # exclude no-info
                    map[int(Y[i,0]), int(Y[i,1])] += 1
    
    "Normalization in [0 1]"
    if not map.max() == 0:
        map /= map.max()
        
    return map 

######################################################################################

def compute_b(I):

    (h,w) = np.shape(I)

    bias = 1

    bx_up = 0
    while bx_up < 0.5*h and (I[0:(bx_up+1), :] == I[0,0]).all():
        bx_up += 1
    bx_up += bias

    bx_down = 0
    while bx_down < 0.5*h and (I[(h-bx_down-1):h, :] == I[-1,0]).all():
        bx_down += 1
    bx_down += bias

    by_left = 0
    while by_left < 0.5*w and (I[:,0:(by_left+1)] == I[0,0]).all():
        by_left += 1
    by_left += bias

    by_right = 0
    while by_right < 0.5*w and (I[:,(w-by_right-1):w] == I[0,0]).all():
        by_right += 1
    by_right += bias

    return (bx_up, bx_down, by_left, by_right)
