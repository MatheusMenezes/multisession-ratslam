import numpy as np
import math as m

def clip_rad_360(angle):
    '''
    Purpose: This routine returns the corresponding angle in the range between
    0 and 2 * pi.

    Algorithm: Calculates the corresponding angle in the range between 0 and
    2 * pi.

    Inputs:
        angle: any angle in radians

    Outputs: an angle between 0 and 2 * pi
    '''
    angle = angle % (2.0 * np.pi)
    return angle

def clip_rad_180(angle):
    '''
    Purpose: This routine returns the corresponding angle in the range between
    - pi and pi.

    Algorithm: Calculates the corresponding angle in the range between - pi and
    pi.

    Inputs:
        angle: any angle in radians

    Outputs: an angle between - pi and pi
    '''
    while ( angle > np.pi ):
        angle -= 2.0 * np.pi
    while ( angle <= -np.pi ):
        angle += 2.0 * np.pi

    return angle

def get_signed_delta_rad(angle1, angle2):
    '''
    Purpose: This routine calculates the delta between two angles in radians in
    the range between 0 and 2 * pi.

    Algorithm: First, calculates if the delta is positive or negative. Then,
    calculates the smallest delta between the two angles in the range between
    0 and 2 * pi.

    Inputs:
        angle1: any angle in radians
        angle2: any angle in radians

    Outputs: the delta between two angles in radians in the range between 0 and
    2 * pi
    '''
    dir         = clip_rad_180( angle2 - angle1 )
    delta_angle = np.abs( clip_rad_360( angle1 ) - clip_rad_360( angle2 ) )
    if delta_angle < ( 2.0 * np.pi - delta_angle ):
        if dir > 0:
            return delta_angle
        else:
            return - delta_angle
    else:
        if dir > 0:
            return ( 2.0 * np.pi - delta_angle )
        else:
            return -( 2.0 * np.pi - delta_angle )

