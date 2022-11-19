# ======================================
# OXFORD NEWCOLLEGE FILE ===============

import numpy as np
import itertools

IMAGE_WIDTH  = 640
IMAGE_HEIGHT = 480
''' ------------------------------------------------------------[ Visual Odometry ] '''
# These four parameters allow the specification of the cropping region for translational velocity
VTRANS_IMAGE_X_MIN = 195
VTRANS_IMAGE_X_MAX = 475
VTRANS_IMAGE_Y_MIN = 270
VTRANS_IMAGE_Y_MAX = 430
# These four parameters allow the specification of the cropping region for rotational velocity
VROT_IMAGE_X_MIN   = 180+15
VROT_IMAGE_X_MAX   = 460+15
VROT_IMAGE_Y_MIN   = 75
VROT_IMAGE_Y_MAX   = 240

# The horizontal camera field of view which is used to scale the rotational velocity
CAMERA_FOV_DEG     = 360.
# The camera frame rate which is used to scale the velocities by accounting for
# the time between frames
CAMERA_HZ          = 1.
# This parameter directly
#  scales the translation velocity into meters per second
VTRANS_SCALING     = 1000.
# This parameter limits the maximum translation velocity to handle large changes in illumination
VTRANS_MAX         = 20.
# range of offset in pixels to consider i.e. slen = 0 considers only the no offset case
OFFSET_MAX         = 100

''' ------------------------------------------------------------[ LocalViewMatch ] '''

# This effectively increases the local contrast of patch regions across the
# current view to handle local changing light conditions and bring out more
# image details. The parameter sets the size of the patch in pixels from its centre
VT_PATCH_NORMALISATION = 0
VT_MIN_PATCH_NORMALISATION_STD = 0
# All templates are normalized by scaling their mean to this parameter. This
# addresses global changes in illumination. Note that values are clipped between
# zero and one
VT_NORMALISATION = 0.4
# The range (in pixel units) of horizontal offsets over which the current image is
# compared to all learnt image templates. Unused in panoramic mode
VT_SHIFT_MATCH = 5
# The number of pixels to increment the shift match offset
VT_STEP_MATCH = 1
# Set this to 1 if the images are panoramic
VT_PANORAMIC = 0
# The sensitivity parameter that determines the boundary between the current visual
# scene being considered novel and being matched to an already learnt visual template
VT_MATCH_THRESHOLD = 0.035

# The horizontal and vertical size in pixels of the 'subsampled' template that
# represents the camera view. For a single intensity profile set template_y_size to 1
TEMPLATE_X_SIZE = 80
TEMPLATE_Y_SIZE = 30
# These four parameters allow a cropping region of the original camera image to
# be specified. Cropping is a useful tool for specifying image regions that are
# salient for place localization. For example, carpet or road can be removed from
# the image. Note these are defined from the top left of the image
IMAGE_VT_X_RANGE_MIN = 40
IMAGE_VT_X_RANGE_MAX = 600
IMAGE_VT_Y_RANGE_MIN = 150
IMAGE_VT_Y_RANGE_MAX = 300
# value of maximum representable finite floating-point (double) number
DBL_MAX = np.finfo(float).max
# Total size of the 'subsampled' templated that represents the camera view
TEMPLATE_SIZE = TEMPLATE_X_SIZE * TEMPLATE_Y_SIZE

''' ------------------------------------------------------------[ PosecellNetwork ] '''
# The side length of the square (x, y) plane of the pose cell network. The larger
# the network size, the greater the computation, but the lower the likelihood of
# a hash collision in the pose cell network and local view cells resulting in a
# false positive loop closure in the experience map
PC_DIM_XY = 30
# The side length of th plane #
PC_DIM_TH = 36

PC_GLOBAL_INHIB = 0.00002
# A local view cell saturation mechanism uses this parameter to rapidly attenuate
# the amount of activity that is injected by repeated exposure to the same visual
# scene. This mechanism ensures the robot is less likely to perform false positive
# loop closures while stationary. The higher the value, the shorter period of time
# over which a single local view cell will inject activity into the pose cell network,
# and the longer the sequence of different visual matches required to perform loop closure
VT_ACTIVE_DECAY = 1.0
# Determines the amount of energy that is injected into the pose cell network
# when a familiar visual scene is recognized. Setting this to a very high value
# ensures one shot localization but makes the system brittle to false positive visual
# matches. Conversely, setting this parameter low means the system is very robust to
# false positive matches but may require long sequences of familiar visual input in
# order to perform loop closure
PC_VT_INJECT_ENERGY = 0.25
# A scaling factor that can be adjusted to suit the translational velocity range
# of the robot or sensor platform. For efficiency reasons the pose cell dynamics
# do not scale ad infinitem and as such this parameter can be adjusted to ensure
# the pose cell network is within its normal operating range. The normal operating
# range for this implementation of OpenRatSLAM is to limit the movement of the energy
# in the network to one cell per iteration
PC_CELL_X_SIZE = 2.0
#  The radius within the pose cell network which can be associated with a single
# experience, if the centroid of the pose cell activity packet moves more than this
# distance a new experience is/ generated, regardless of whether the visual scene has changed
EXP_DELTA_PC_THRESHOLD = 1.0

# Determines the rate at which a local view cell is restored to its original
# state after being attenuated due to repeated activations
PC_VT_RESTORE = 0.05

PC_C_SIZE_TH    = ( 2.0 * np.pi ) / PC_DIM_TH

# parameters used to build the excitatory and inhibitory matrices
PC_W_E_DIM = 7
PC_W_I_DIM = 5
PC_W_E_VAR = 1
PC_W_I_VAR = 2

PC_W_E_DIM_HALF = int(np.floor(PC_W_E_DIM/2.))
PC_W_I_DIM_HALF = int(np.floor(PC_W_I_DIM/2.))

# wrap connections in all axis for excitatory and inhibitory processes / # Change Python 3 - Paulo    
PC_E_XY_WRAP = list(range(PC_DIM_XY - PC_W_E_DIM_HALF, PC_DIM_XY))+ list(range(PC_DIM_XY)) + list(range(PC_W_E_DIM_HALF))
PC_E_TH_WRAP = list(range(PC_DIM_TH - PC_W_E_DIM_HALF, PC_DIM_TH))+ list(range(PC_DIM_TH)) + list(range(PC_W_E_DIM_HALF))
PC_I_XY_WRAP = list(range(PC_DIM_XY - PC_W_I_DIM_HALF, PC_DIM_XY))+ list(range(PC_DIM_XY)) + list(range(PC_W_I_DIM_HALF))
PC_I_TH_WRAP = list(range(PC_DIM_TH - PC_W_I_DIM_HALF, PC_DIM_TH))+ list(range(PC_DIM_TH)) + list(range(PC_W_I_DIM_HALF))

# parameters used to find the centre of activity packet / # Change Python 3 - Paulo    
PC_CELLS_TO_AVG = 5

PC_AVG_XY_WRAP = list(range(PC_DIM_XY - PC_CELLS_TO_AVG, PC_DIM_XY)) + list(range(PC_DIM_XY)) + list(range(PC_CELLS_TO_AVG))
PC_AVG_TH_WRAP = list(range(PC_DIM_TH - PC_CELLS_TO_AVG, PC_DIM_TH)) + list(range(PC_DIM_TH)) + list(range(PC_CELLS_TO_AVG))

PC_XY_SUM_SIN_LOOKUP = np.sin(np.multiply(range(1, PC_DIM_XY+1), (2*np.pi)/PC_DIM_XY))
PC_XY_SUM_COS_LOOKUP = np.cos(np.multiply(range(1, PC_DIM_XY+1), (2*np.pi)/PC_DIM_XY))
PC_TH_SUM_SIN_LOOKUP = np.sin(np.multiply(range(1, PC_DIM_TH+1), (2*np.pi)/PC_DIM_TH))
PC_TH_SUM_COS_LOOKUP = np.cos(np.multiply(range(1, PC_DIM_TH+1), (2*np.pi)/PC_DIM_TH))

# utilize func for constants below
def norm2d(dim, var):
   dim_centre = int(np.floor(dim / 2.0))

   weight = np.zeros([dim, dim, dim])
   for x, y, z in itertools.product(range(dim), range(dim), range(dim)): # Change Python 3 - Paulo    
       dx = (x - dim_centre) ** 2
       dy = (y - dim_centre) ** 2
       dz = (z - dim_centre) ** 2
       weight[z, y, x] = 1.0 / (var * np.sqrt(2 * np.pi)) * np.exp((-dx - dy - dz) / (2. * (var ** 2)))

   weight = weight / np.sum(weight)
   return weight

# Excitatory and inhibitory matrices
PC_W_EXCITE = norm2d(PC_W_E_DIM, PC_W_E_VAR)
PC_W_INHIB  = norm2d(PC_W_I_DIM, PC_W_I_VAR)


# Actions for the experience map's topological graph. The possible actions are:
# no action, create a new node (which implicitly includes creating an edge from the
# previous node), create an edge between two existing nodes or set the location to
# an existing node
NO_ACTION   = 0
CREATE_NODE = 1
CREATE_EDGE = 2
SET_NODE    = 3

''' ------------------------------------------------------------[ ExperienceMap ]'''
# The number of complete experience map graph relaxation cycles to perform per system iteration
EXP_LOOPS          = 250
# Correction parameter to correct e0 and e1 (x,y) in the loop closure
# A 0.5 correction parameter means that e0 and e1 will be fully
# corrected based on e0's link information
EXP_CORRECTION     = 0.5
# Initial facing angle
EXP_INITIAL_EM_DEG = 140
# not used -- maximum number of goals
MAX_GOALS          = 10