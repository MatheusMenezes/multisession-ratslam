
# -------------------------------------------------------------------[ header ]

import numpy as np
from ._globals import *

# ------------------------------------------------------------[ VisualOdometry class ]

class VisualOdometry(object):
    def __init__(self):
        '''
        vtrans-profile: 1D numpy array normalized (or intensity profile) that represents
        the current frame and it is used to calculate the translation velocity
        
        vtrans-prev-profile: 1D numpy array normalized (or intensity profile) that
        represents the previous frame and it is used to calculate the translation velocity

        vrot-profile: 1D numpy array normalized (or intensity profile) that represents
        the current frame and it is used to calculate rotational velocity

        vrot-prev-profile: 1D numpy array normalized (or intensity profile) that
        represents the previous frame and it is used to calculate rotational velocity

        first: controls if it is the first time that this class is being used
        '''
        self.vtrans_profile      = np.zeros( VTRANS_IMAGE_X_MAX - VTRANS_IMAGE_X_MIN )
        self.vtrans_prev_profile = np.zeros( VTRANS_IMAGE_X_MAX - VTRANS_IMAGE_X_MIN )
        self.vrot_profile        = np.zeros( VROT_IMAGE_X_MAX - VROT_IMAGE_X_MIN )
        self.vrot_prev_profile   = np.zeros( VROT_IMAGE_X_MAX - VROT_IMAGE_X_MIN )
        self.first = True
        self.odometry = [0., 0., np.pi/2]

    def on_image(self, data, greyscale):
        '''
        Purpose: This routine defines the sequence of steps necessary to calculate the
        translational velocity and rotational velocity between the current frame (data)
        and the previous frame.

        Algorithm: First, transforms the current frame in 1D array normalized (template),
        then calculates the translational velocity by comparing the current profile
        (vtrans-profile) to the previous profile (vtrans-prev-profile). For rotational
        velocity, it is the same steps used for translational velocity.

        Inputs: 
            data: current frame

            greyscale: True if the current frame is on grayscale, False if the current
            frame is not on grayscale

        Outputs: translational velocity and rotational velocity
        '''
        if(self.first):
            self.vtrans_prev_profile = self.vtrans_profile[:]
            self.vrot_prev_profile   = self.vrot_profile[:]
            self.first = False

        aux = 0

        # Calculates the translational velocity
        # first: convert the image to a specific template/profile

        # Matheus: soma todos os valores das colunas e coloca em um vetor 1-D
        self.vtrans_profile = self.__convert_view_to_view_template__( self.vtrans_profile,
                                                                     data, greyscale,
                                                                     VTRANS_IMAGE_X_MIN,
                                                                     VTRANS_IMAGE_X_MAX,
                                                                     VTRANS_IMAGE_Y_MIN,
                                                                     VTRANS_IMAGE_Y_MAX )

        # second: calculates the translational velocity by comparing current profile and
        # previous profile
        self.vtrans_prev_profile, vtrans_ms, aux = self.__visual_odo__( self.vtrans_profile,
                                                                    self.vtrans_profile.size,
                                                                    self.vtrans_prev_profile )

        # Calculates the rotational velocity
        # first: convert the image to a specific template/profile
        self.vrot_profile = self.__convert_view_to_view_template__( self.vrot_profile,
                                                                data, greyscale,
                                                                VROT_IMAGE_X_MIN,
                                                                VROT_IMAGE_X_MAX,
                                                                VROT_IMAGE_Y_MIN,
                                                                VROT_IMAGE_Y_MAX )
        # second: calculates the rotational velocity by comparing current profile and
        # previous profile
        self.vrot_prev_profile, aux, vrot_rads = self.__visual_odo__( self.vrot_profile,
                                                                  self.vrot_profile.size,
                                                                  self.vrot_prev_profile)

        self.odometry[2] += vrot_rads 
        self.odometry[0] += vtrans_ms*np.cos(self.odometry[2])
        self.odometry[1] += vtrans_ms*np.sin(self.odometry[2])

        return vtrans_ms, vrot_rads

    def __visual_odo__(self, data, width, olddata):
        '''
        Purpose: This routine calculates the translational and rotational velocity
        between the current frame and the previous frame.

        Algorithm: Compares data to olddata, first shifting only olddata by an offset
        value, then shifting only data by an offset value. The minimum difference is
        stored in mindiff and the minimum offset is stored in minoffset. Then, olddata
        is updated, receving data value. Finally, the rotational velocity and translational
        velocity are calculated.

        Inputs: 
            data: 1D numpy array normalized (or intensity profile) that represents the
            current frame

            width: size of data

            olddata: 1D numpy array normalized (or intensity profile) that represents
            the previous frame 
            
        Outputs: olddata updated, translational velocity and rotational velocity
        '''
        mindiff   = 1e10 #1e6
        minoffset = 0
        cwl  = width       # length of the intensity profile to actually compare,
                            # and must be less than image width minus 1 x slen
        slen = OFFSET_MAX  # range of offset in pixels to consider i.e. slen = 0 considers only the no offset case

        for offset in range( slen ):
            
            # => Sera que o problema eh a normalizacao? <=

            ''' np.subtract
                
                x1 = np.arange(9.0).reshape((3, 3))
                array([[ 0.,  1.,  2.],
                       [ 3.,  4.,  5.],
                       [ 6.,  7.,  8.]])

                x2 = np.arange(3.0)
                array([ 0.,  1.,  2.])
                
                np.subtract(x1, x2)
                array([[ 0.,  0.,  0.],
                       [ 3.,  3.,  3.],
                       [ 6.,  6.,  6.]])
            '''
            cdiff = np.sum( np.abs( np.subtract( data[:cwl - offset], olddata[offset:cwl] ) ) )
            cdiff /= ( 1.0 * ( cwl - offset ) ) 
            if ( cdiff < mindiff ):
                mindiff   = cdiff
                minoffset = -offset
            cdiff = np.sum( np.abs( np.subtract( data[offset:cwl], olddata[:cwl - offset] ) ) )
            cdiff /= ( 1.0 * ( cwl - offset ) )
            if ( cdiff < mindiff ):
                mindiff   = cdiff
                minoffset = offset
            
            '''
            e = (cwl-offset)

            cdiff = np.abs(data[offset:cwl] - olddata[:cwl - offset])
            cdiff = np.sum(cdiff)/e

            if cdiff < mindiff:
                mindiff = cdiff
                minoffset = offset

            cdiff = np.abs(data[:cwl - offset] - olddata[offset:cwl])
            cdiff = np.sum(cdiff)/e

            if cdiff < mindiff:
                mindiff = cdiff
                minoffset = -offset
            '''
                
        '''
        print "OUT LOOP:"
        print "mindiff: " + str(mindiff) 
        print "minoffset: " + str(minoffset) 
        print "========"
        '''
        # Updating olddata (vtrans_prev_profile or vrot_prev_profile)
        olddata   = data[:]
        # print "minoffset: " + str(minoffset)  
        # Calculating rotational velocity and translational velocity
        # vrot_rads = minoffset*(50./IMAGE_WIDTH)*np.pi/180
        vrot_rads = (minoffset * (CAMERA_FOV_DEG / IMAGE_WIDTH) * CAMERA_HZ) * np.pi / 180.0 # vrot is in radians
        vtrans_ms = min( mindiff * VTRANS_SCALING, VTRANS_MAX ) # vtrans should not be higher then VTRANS_MAX
        #print "vtrans_ms: " + str(mindiff * VTRANS_SCALING) 
        return olddata, vtrans_ms, vrot_rads

    def __convert_view_to_view_template__(self, current_view, view_rgb, grayscale, X_RANGE_MIN, X_RANGE_MAX, Y_RANGE_MIN, Y_RANGE_MAX):
        '''
        Purpose: This routine transforms the current frame in a 1D array normalized
        that can be used as a profile.
        
        Algorithm: It first defines the ranges of the current frame (view-rgb) that
        are going to be used, then sums all its values in the y-axis. Finally, the
        resulting array is divided by the range used for the y-axis (Y-RANGE-MAX -
        Y-RANGE-MIN) and by 255, if it is a grayscale frame. In case it is used an
        RGB frame, the algorithm is the same, however, the current view is also
        divided by 3.

        Inputs: 
            current-view: numpy array that is going to store the 1D array normalized,
            or template produced

            view-rgb: current frame captured by the robot

            grayscale: True if the current frame is on grayscale, False if the current
            frame is not on grayscale
            
            X-RANGE-MAX and X-RANGE-MIN: ranges in x-axis of the current frame that
            will be used

            Y-RANGE-MAX and Y-RANGE-MIN: ranges in y-axis of the current frame that
            will be used           
        
        Outputs: numpy array of 1D (X-RANGE-MAX - X-RANGE-MIN) normalized
        '''
        TEMPLATE_Y_SIZE = 1
        TEMPLATE_X_SIZE = X_RANGE_MAX - X_RANGE_MIN
        
        sub_range_x = X_RANGE_MAX - X_RANGE_MIN
        sub_range_y = Y_RANGE_MAX - Y_RANGE_MIN
        
        x_block_size = sub_range_x / TEMPLATE_X_SIZE
        y_block_size = sub_range_y / TEMPLATE_Y_SIZE
        '''
        print "x_block_size: " + str(x_block_size)
        print "y_block_size: " + str(y_block_size)
        '''
        # select the part of the frame that is going to be used
        current_view = view_rgb[Y_RANGE_MIN : Y_RANGE_MAX, X_RANGE_MIN : X_RANGE_MAX]

        if grayscale:
            current_view = np.sum( current_view ,0 ) / ( 255.0 * x_block_size * y_block_size)
        else:
            current_view = np.sum( np.sum( current_view.reshape( sub_range_y, 3 * sub_range_x ),0 ).reshape( sub_range_x, 3 ),1 )\
                           / ( 3.0 * 255.0 * x_block_size * y_block_size )
        return current_view

    def __call__(self, data, greyscale):
        return self.on_image( data, greyscale )
