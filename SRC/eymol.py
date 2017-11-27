'''
Created on:	29 May 2017

@author: 	Dario Zanca (dario.zanca@unifi.it, dariozanca@gmail.com)

           	Ph.D. Student in Smart Computing,
                University of Florence and University of Siena.

@summary: 	Collection of functions to generate scanpaths with EYMOL.
'''

# IMPORT EXTERNAL LIBRARIES

import numpy as np
import cv2
from math import sin, cos, pi, isnan
from random import randint, uniform
from scipy.io import loadmat
from scipy.integrate import odeint

#########################################################################################
#########################################################################################

def trajectory(VIDEO, VIDEOp,   # input stimuly as numpy 2D- or 3D-matrix
                                # (FRAME,X1,X2), grayscale already

                   times,  # time interval to simulate

                   eta_b=1.13925935 * 10 ** 3,  # curiosity (local)
                   eta_p=1.24170715 * 10 ** 5,  # curiosity (peripheral)
                   lambd=10 ** -6,  # brightness invariance

                   theta=1,  # dissipation
                   kappa=10 ** 6,  # elastic constant (boundedness)

                   #apply_center=True,  # if you would center bias input

                   y0=[],  # system initial conditions:
                   # [x1_init, x2_init, v1_init, v2_init]

                   FRAME_RATE=30,  # Frame-rate of the input source
                   FixPS=4,  # Fixations per second

                   time_scale=10,  # times focus faster then the input

                   current_time=0.0,  # since it is a real-time application,
                   # we need a variable to save value of
                   # current time; this is particulary
                   # important for the sin/cos functions
                   # in the curiosity term
               ):

    ''' Given a visual input and its peripheral as well, this function produces a
    scanpath of visual attention. The duration of the visual exploration has to be
    specified in times.

    ARGS:
        - VIDEO: 2d- or 3d-matrix of the visual input. (grayscale)
        - VIDEOp: same dimension of VIDEO, peripheral visual input. (grayscale)
        - times: A sequence of time points for which to solve for y.
        - y0 (optional): the initial value point. '''

    "If it is an image, add the temporal dimension."
    if len(VIDEO.shape) == 2:
        VIDEO = img_2d_to_3d(VIDEO)
        VIDEOp = img_2d_to_3d(VIDEOp)


    "Get video dimensions"
    _, h, w = VIDEO.shape

    "Parameters of the system has to be set here"

    parameters = {'m': 1,
                  'eta_b': eta_b,
                  'eta_p': eta_p,
                  'lambd': lambd,
                  'theta': theta,
                  'k': kappa,
                  'r': (5, h - 5, 5, w - 5),
                  'omega': pi * (1.0 / (FRAME_RATE)) * (FixPS),
                  'time_scale': time_scale}

    "Numerical method"

    # If not provided, generate random initial conditions
    if not y0: y0 = generate_initial_conditions(h,w)

    # Generate scanpath (by integrating diff. equations)
    y = odeint(myode, y0, times,
               args=(VIDEO, VIDEOp, current_time, parameters),
               mxstep=1000, rtol=10 ** -2, atol=10 ** -2
               )

    return y

#########################################################################################

def generate_initial_conditions(h,w):

    ''' This function generates initial condition for the dynamical system to be
    integrated. Numbers used here are arbitrary. Consider to motify or determine better
    numbers in future implementations. '''

    initRay = int(min(h, w) * 0.17)
    x1_init = int(h / 2) + randint(-initRay, initRay)
    x2_init = int(w / 2) + randint(-initRay, initRay)
    v1_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))
    v2_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))

    return [x1_init, x2_init, v1_init, v2_init]

#########################################################################################

def img_2d_to_3d(img):

    ''' This function simply add the temporal dimension to the 2d-frame. '''

    img3d = np.zeros((1, img.shape[0], img.shape[1]))
    img3d[0, :, :] = img

    return img3d

#########################################################################################

def derivatives(b, P,

                h=1 # increment for derivative computation
                ):

    '''	Function which compute the derivatives of the image b(TIME,X1,X2) at
        the point P=(fr,x1,x2).

        b() is a video; in the case of static images, we assume to recive a
        single frame with sizes (1, X1, X2).

        Notice, time t is considered in terms of frames fr. '''

    s = np.shape(b)

    '''If the input b is a single frame (an image instead of a video), derivatives are
    calculated in the zero time instant. This cause no time derivatives, but only 
    spatial derivatives, as it is desired in such a case.'''

    if s[0] == 1:
        P[0] = 0.0 # if it is a single frame, set time to zero

    '''Get integer value for x- and y-pixel, and time t.'''
    x1 = int(P[1])
    x2 = int(P[2])
    t = int(P[0])

    '''Derivatives values will be filled inside a dictionary'''
    b_ = {}  # this dictionary will contain derivatives

    if (2 * h <= x1 < s[1] - 2 * h and 2 * h <= x2 < s[2] - 2 * h) and (t < s[0]):

        # first derivatives

        b_['x1'] = ((2 * h) ** -1) * (b[t, x1 + h, x2] - b[t, x1 - h, x2])

        b_['x2'] = ((2 * h) ** -1) * (b[t, x1, x2 + h] - b[t, x1, x2 - h])

        if 2 * h <= t < s[0] - 2 * h:
            b_['t'] = ((2 * h) ** -1) * (b[t + h, x1, x2] - b[t - h, x1, x2])
        else:
            b_['t'] = 0.0

        # secon derivatives (same variable)

        b_['x1,x1'] = ((4 * h ** 2) ** -1) * (b[t, x1 + 2 * h, x2]
                                              - 2 * b[t, x1, x2] + b[t, x1 - 2 * h, x2] )

        b_['x2,x2'] = (	( 4 * h **2 ) **-1 ) * ( b[t, x1, x2+ 2 * h]
                                              - 2 * b[t, x1, x2] + b[t, x1, x2 - 2 * h] )

        if 2* h <= t < s[0] - 2 * h:
            b_['t,t'] = ((4 * h ** 2) ** -1) * (b[t + 2 * h, x1, x2]
                                                - 2 * b[t, x1, x2] + b[t - 2 * h, x1, x2] )
        else:
            b_['t,t'] = 0.0

        # secon derivatives (mixed)

        b_['x1,x2'] = (	( 4 * h **2 ) **-1 ) * ( b[t, x1+ h, x2 + h] - b[t, x1 - h, x2 + h]
                                              - b[t, x1 + h, x2 - h] + b[t, x1 - h, x2 - h] )

        b_['x2,x1'] = b_['x1,x2']

        if 2* h <= t < s[0] - 2 * h:

            b_['t,x1'] = ((4 * h ** 2) ** -1) * (b[t + h, x1 + h, x2]
                                                 - b[t + h, x1 - h, x2] - b[t - h, x1 + h, x2]
                                                 + b[t - h, x1 - h, x2] )
            b_['x1,t'] = b_['t,x1']

            b_['t,x2'] = (	( 4 * h **2 ) **-1 ) * ( b[ t +h, x1, x2+ h]
                                                 - b[t + h, x1, x2 - h] - b[t - h, x1, x2 + h]
                                                 + b[t - h, x1, x2 - h] )
            b_['x2,t'] = b_['t,x2']

        else:

            b_['t,x1'] = 0.0
            b_['x1,t'] = b_['t,x1']

            b_['t,x2'] = 0.0
            b_['x2,t'] = b_['t,x2']


    else:

        # first derivatives
        b_['x1'] = 0.0
        b_['x2'] = 0.0
        b_['t'] = 0.0

        # second derivatives (same variable)
        b_['x1,x1'] = 0.0
        b_['x2,x2'] = 0.0
        b_['t,t'] = 0.0

        # second derivatives (mixed)
        b_['x1,x2'] = 0.0
        b_['x2,x1'] = 0.0
        b_['t,x1'] = 0.0
        b_['x1,t'] = 0.0
        b_['t,x2'] = 0.0
        b_['x2,t'] = 0.0

    return b_

################################################################################

def myode(y, t, b, p, current_time, parameters):

    '''	This function describes the system of two second-order differential
        equations which describe visual attention.

        y: it is the vector of the variables (x1, x2, dot x1, dot x2)

        t: time (frames)

        b: size=(TIME,X1,X2) it is the visual input - in general a video.

        p: size=(TIME,X1,X2) it is the peripherical visual input

        parameters: dictionary containing all the parameters of the model '''

    # Get parameters
    m = parameters['m']
    eta_b = parameters['eta_b']
    eta_p = parameters['eta_p']
    lambd = parameters['lambd']
    theta = parameters['theta']
    k = parameters['k']
    r1_1, r1_2, r2_1, r2_2 = parameters['r']
    omega = parameters['omega']
    time_scale = parameters['time_scale']

    # Exact time in thi instant must sum current time in the real-time app
    # Remember: we simulate in a batch o 5 frames
    # | 0 | 1 | 2 | 3 | 4 |
    #         *   *
    # in the small interval depicted by the two stars. When you have the first
    # 5 frames, current_time = 5. So, to get the exact time in the small
    # interval between stars, we multiply by time_scale add the "t" of myode.
    # Note: in case of static image, it works as well.

    current_time = current_time*time_scale + t

    # Compute derivatives of b() and p(), at the actual position P
    P = np.array([ t /time_scale, y[0], y[1]])
    b_ = derivatives(b, P)
    p_ = derivatives(p, P)


    # Potentials contribution

    C_x = [	2 * eta_b * cos( omega *current_time) * b_['x1'] * b_['x1,x1']
            + 2 * eta_p * sin( omega *current_time) * p_['x1'] * p_['x1,x1'] ,

            2 * eta_b * cos( omega *current_time) * b_['x2'] * b_['x2,x2']
            + 2 * eta_p * sin( omega *current_time) * p_['x2'] * p_['x2,x2']	]

    V_x = [	2 * k * ((y[0] - r1_2) * float(y[0] > r1_2)
            + (y[0] - r1_1) * float(y[0] < r1_1)),

            2 * k * ((y[1] - r2_2) * float(y[1] > r2_2)
            + (y[1] - r2_1) * float(y[1] < r2_1))	]

    dt_dotb = (	b_['t,t'] + b_['t,x1' ] *y[2] + b_['t,x2' ] *y[3] +
                (b_['x1,t'] + b_['x1,x1' ] *y[2] + b_['x1,x2' ] *y[3]) * y[2] +
                (b_['x2,t'] + b_['x2,x1' ] *y[2] + b_['x2,x2' ] *y[3]) * y[3]	)

    BI_res_x = [	2 * lambd * dt_dotb * b_['x1'],
                    2 * lambd * dt_dotb * b_['x2']	]

    # dissipation term
    dissipation = [	theta * ( m *y[2] - 2 * (	b_['t'] + b_['x1' ] *y[2]
                                            + b_['x2' ] *y[3]) * b_['x1']),

                    theta * ( m *y[3] - 2 * (	b_['t'] + b_['x1' ] *y[2]
                                            + b_['x2' ] *y[3]) * b_['x2'])	]

    # A term
    A = ( 	+ np.array(C_x)
            - np.array(V_x)
            + np.array(BI_res_x)
            - np.array(dissipation)		)

    # Cramer's D's

    D =  np.linalg.det(
            np.array([	[m - 2 * lambd * b_['x1' ] **2,
                            - 2 * lambd * b_['x1'] * b_['x2']	],

                        [- 2 * lambd * b_['x1'] * b_['x2'],
                            m - 2 * lambd * b_['x2' ] **2			]	])
                        )

    D1 = np.linalg.det(
            np.array([	[A[0],		- 2 * lambd * b_['x1'] * b_['x2']	],
                        [A[1],	m - 2 * lambd * b_['x2' ] **2			]	])
                        )

    D2 = np.linalg.det(
            np.array([	[m - 2 * lambd * b_['x1' ] **2,		A[0]	],
                        [- 2 * lambd * b_['x1'] * b_['x2'],	A[1]	]	])
                        )

    
    "System of differential equations"

    dy = [	y[2],

            y[3],

            D1/D,

            D2/D
          ]

    return dy


################################################################################

def preprocessing_frame(frame,

                        apply_center=True,
                        apply_hist_eq=True,
                        ):

    # Convert to gray-scale
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    if apply_hist_eq: frame = cv2.equalizeHist(frame)

    # Convert to float
    frame = frame.astype(float)

    # Apply center bias
    if apply_center:
        h, w = np.shape(frame)[0], np.shape(frame)[1]
        center = loadmat('center.mat')['center']
        center = cv2.resize(center, (w,h) )
        frame = frame * center

    return frame


################################################################################

def preprocessing_local(frame, blur):

    I = cv2.GaussianBlur(frame, (blur, blur), 0)

    I = frame_normalize(I)

    return I

################################################################################

def write_red_dot(frame, (row, col), RAY=5):
    # get point coordinates
    if isnan(row) or isnan(col):
        row, col = 0, 0
    else:
        row, col = int(row), int(col)

    if (row - RAY < 0):
        row = RAY
    else:
        if (row + RAY >= np.shape(frame)[0]):
            row = np.shape(frame)[0] - RAY - 1
    if (col - RAY < 0):
        col = RAY
    else:
        if (col + RAY >= np.shape(frame)[1]):
            col = np.shape(frame)[1] - RAY - 1

    cv2.circle(frame,
               (col, row),
               5, (0, 0, 255), 3)

    return frame


################################################################################

def frame_normalize(frame):
    if not frame.max() == 0:
        frame = (frame - frame.min()).astype(float) / (frame.max() - frame.min()).astype(float)
        #frame = (frame) / (frame.max())

    return frame


################################################################################

def frame_resize(frame, max_dim=480.0):
    h, w = np.shape(frame)[0], np.shape(frame)[1]

    h, w = float(h), float(w)

    if h > w:
        w = (max_dim / h) * w
        h = max_dim
    else:
        h = (max_dim / w) * h
        w = max_dim

    h, w = int(h), int(w)

    frame_smaller = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)

    return frame_smaller
