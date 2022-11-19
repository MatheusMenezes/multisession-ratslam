
# -------------------------------------------------------------------[ header ]

import numpy as np
from ._globals import *

# ------------------------------------------------------------[ PosecellVisualTemplate class ]

class PosecellVisualTemplate(object):
    def __init__(self, id, pc_x, pc_y, pc_th, decay):
        '''
        id: id of the visual template
        pc_x: x coordinate of the associated pose cell
        pc_y: y coordinate of the associated pose cell
        pc_th: th coordinate of the associated pose cell
        decay: controls the injected energy avoiding potentially incorrect
        re-localizations when the robot is motionless for long periods of time
        exps: collection of experiences that are associated to this visual template
        '''
        self.id = id
        self.pc_x = pc_x
        self.pc_y = pc_y
        self.pc_th = pc_th
        self.decay = decay
        self.exps = np.array([])

# ------------------------------------------------------------[ PosecellExperience class ]

class PosecellExperience(object):
    def __init__(self, x_pc, y_pc, th_pc, vt_id):
        '''
        x_pc: x coordinate of the associated pose cell
        y_pc: y coordinate of the associated pose cell
        th_pc: th coordinate of the associated pose cell
        vt_id: id of the visual template
        '''
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.vt_id = vt_id

# ------------------------------------------------------------[ PosecellNetwork class ]

class PoseCellNetwork(object):

    def __init__(self, transform_function):
        '''
        best_x: x coordinate of the associated pose cell
        best_y: y coordinate of the associated pose cell
        best_th: th coordinate of the associated pose cell
        vt_delta_pc_th: relative angle difference between closest visual
            template and current visual template
        odo_update: controls if the odometry was analyzed
        vt_update: controls if influence of the visual template was analyzed

        posecells: pose cell network activity
        visual_templates: collection of pose cell visual templates
        experiences: collection of pose cell experiences

        current_vt: id of current visual template
        prev_vt: id of previous visual template
        current_exp: id of current pose cell experience
        prev_exp: id of previous pose cell experience
        '''

        #DUVIDA: best th, y, x quando treinar novos templates deve ser inicializado onde?
        self.best_th, self.best_y, self.best_x = transform_function
        self.vt_delta_pc_th = 0
        self.odo_update = False
        self.vt_update = False

        self.posecells = np.zeros([PC_DIM_TH, PC_DIM_XY, PC_DIM_XY])
        self.posecells[int(self.best_th),int(self.best_y),int(self.best_x)] = 1 # Change Python 3 - Paulo    
        self.visual_templates = np.array([])
        self.experiences = np.array([])

        self.current_vt = 0
        self.prev_vt = 0
        self.current_exp = 0
        self.prev_exp = 0

    def __inject__(self, act_x, act_y, act_th, energy):
        '''
        Purpose: This routine injects energy into a specific point in the network.

        Algorithm: Check if the point is a valid point in the pose cell network,
        then add the energy at this specific point.

        Inputs:
            act_x: x coordinate the point that will receive the energy
            act_y: y coordinate the point that will receive the energy
            act_z: z coordinate the point that will receive the energy
            energy: the value that should be injected in the pose cell network

        Outputs: True, if energy is injected otherwise, False.
        '''
        if 0 <= act_x < PC_DIM_XY and 0 <= act_y < PC_DIM_XY and 0 <= act_th < PC_DIM_TH:
            self.posecells[act_th][act_y][act_x] += energy
            return True
        else:
            return False

    def __excite__(self):
        '''
        Purpose: This routine locally excites points in the pose cell network, spreading energy
        through the network.

        Algorithm: Find which cells in the pose cell have energy. Then spread the energy locally,
        which the range of cells affected are defined by PC_W_E_DIM matrix and the weight of
        excitatory connections are stored in PC_W_EXCITE.

        Inputs: -

        Outputs: -
        '''
        pca_new = np.zeros( [ PC_DIM_TH, PC_DIM_XY, PC_DIM_XY ] )

        # first: find which cells in the posecell have energy
        index = np.nonzero( self.posecells )

        # second: generates all the positions that will receive energy and add energy to these positions
        # PC_E_XY_WRAP: is used to maintain the wrap connection through this network
        # PC_W_E_DIM: means how far does the energy of and cell spreads
        # PC_W_EXCITE: excitatory matrix
        for k,j,i in zip( *index ):
            pca_new[ np.ix_( PC_E_TH_WRAP[ k : k + PC_W_E_DIM ] ,
                            PC_E_XY_WRAP[ j : j + PC_W_E_DIM ],
                            PC_E_XY_WRAP[ i : i + PC_W_E_DIM] ) ] += \
                self.posecells[ k,j,i ] * PC_W_EXCITE

        self.posecells = pca_new

    def __inhibit__(self):
        '''
        Purpose: This routine locally inhibits points in the pose cell network, compressing
        through the network.

        Algorithm: Find which cells in the pose cell have energy. Then compress the energy locally,
        which the range of cells affected are defined by PC_W_I_DIM matrix and the weight of
        inhibitory connections are stored in PC_W_INHIB.

        Inputs: -

        Outputs: -

        '''
        pca_new = np.zeros([ PC_DIM_TH, PC_DIM_XY, PC_DIM_XY ])

        # first: find which cells in the posecell have energy
        index = np.nonzero( self.posecells )

        # second: generates all the positions that will have their energy compressed and
        # calculate the energy that should be compressed
        # PC_I_XY_WRAP: is used to maintain the wrap connection through this network
        # PC_W_I_DIM: means how far does the energy of and cell will be compressed
        # PC_W_INHIB: inhibitory matrix
        for k,j,i in zip( *index ):
            pca_new[ np.ix_( PC_I_TH_WRAP[ k : k + PC_W_I_DIM ],
                            PC_I_XY_WRAP[ j : j + PC_W_I_DIM ],
                            PC_I_XY_WRAP[ i : i + PC_W_I_DIM ] ) ] += \
                self.posecells[ k, j, i ] * PC_W_INHIB
        # third: subtract the inhibition process
        self.posecells -= pca_new

    def __global_inhibit__(self):
        '''
        Purpose: This routine is responsible for the global inhibition process.

        Algorithm: For all cells that have more energy than PC_GLOBAL_INHIB threshold,
        this value will be subtracted. For the rest, the energy will be set to 0.

        Inputs: -

        Outputs: -
        '''
        self.posecells[ self.posecells < PC_GLOBAL_INHIB ]   = 0
        self.posecells[ self.posecells >= PC_GLOBAL_INHIB ] -= PC_GLOBAL_INHIB

    def __normalise__(self):
        '''
        Purpose: This routine normalizes all the energy in the system.

        Algorithm: Divide all values by the total energy in the system.

        Inputs: -

        Outputs: -
        '''
        total = np.sum( self.posecells )
        if total > 0:
            self.posecells = self.posecells / total

    def __path_integration__(self, vtrans, vrot):
        '''
        Purpose: This routine shifts the energy in the system by a translational
        and rotational velocity.

        Algorithm: First, scale the translational velocity. Then, shift the pose cell
        network in each th plane given by the th. Rotate the pose cell network instead
        of implementing for four quadrants. Extend the pc.Posecells one unit in each
        direction work out the weight contribution to the NE cell from the SW, NW,
        SE cells given vtrans and the direction think in terms of NE divided into
        4 rectangles with the sides given by vtrans and the angle. Circular shift
        and multiple by the contributing weight copy those shifted elements for
        the wrap around. Unrotate the pose cell xy layer. Finally, shift the pose
        cells +/- theta given by vrot mod to work out the partial shift amount.

        Inputs:
            vtrans: translational velocity
            vrot: rotational velocity

        Outputs: -
        '''
        angle_to_add = 0
        # first: scale the translational velocity
        vtrans /= PC_CELL_X_SIZE

        if vtrans < 0:
            vtrans = -vtrans
            angle_to_add = np.pi

        # shift in each th given by the th
        for dir_pc in range( PC_DIM_TH ):
            dir   = np.float( dir_pc ) * PC_C_SIZE_TH + angle_to_add

            # rotate the posecell network instead of implementing for four quadrants
            pca90 = np.rot90( self.posecells[ dir_pc, : , : ], ( -1 ) * int( np.floor( dir * 2.0 / np.pi ) ) )
            dir90 = dir - np.floor( dir * 2 / np.pi ) * np.pi / 2

            # extend the pc.Posecells one unit in each direction
            # work out the weight contribution to the NE cell from the SW, NW, SE cells
            # given vtrans and the direction
            # think in terms of NE divided into 4 rectangles with the sides
            # given by vtrans and the angle
            pca_new                    = np.zeros( [ PC_DIM_XY + 2, PC_DIM_XY + 2 ] )
            pca_new[ 1 : -1 , 1 : -1 ] = pca90

            weight_sw = ( vtrans**2 ) * np.cos( dir90 ) * np.sin( dir90 )

            weight_se = vtrans * np.sin( dir90 ) * ( 1.0 - vtrans * np.cos( dir90 ) )

            weight_nw = vtrans * np.cos( dir90 ) * ( 1.0 - vtrans * np.sin( dir90 ) )

            weight_ne = 1.0 - weight_sw - weight_se - weight_nw

            # circular shift and multiple by the contributing weight
            # copy those shifted elements for the wrap around
            pca_new = pca_new * weight_ne +\
                      np.roll( pca_new, 1, 1 ) * weight_se + \
                      np.roll( pca_new, 1, 0 ) * weight_nw + \
                      np.roll( np.roll( pca_new, 1, 0 ), 1, 1 ) * weight_sw
            pca90        = pca_new[ 1 : -1, 1 : -1 ]
            pca90[1:,0] += pca_new[ 2 : -1, -1]
            pca90[0,1:] += pca_new[ -1, 2 : -1 ]
            pca90[0,0]  += pca_new[ -1, -1 ]

            # unrotate the pose cell xy layer
            self.posecells[ dir_pc , : , : ] = np.rot90( pca90, ( -1 ) * ( 4 - int( np.floor( dir * 2.0 /np.pi ) ) ) )

        # Shift the pose cells +/- theta given by vrot mod to work out the partial shift amount
        if vrot != 0:
            weight = ( np.abs( vrot ) / PC_C_SIZE_TH )%1
            if weight == 0:
                weight = 1.0
            shift1 = int( np.sign( vrot ) * int( np.floor( np.abs( vrot ) / PC_C_SIZE_TH ) ) )
            shift2 = int( np.sign( vrot ) * int( np.ceil( np.abs( vrot ) / PC_C_SIZE_TH ) ) )
            self.posecells = np.roll( self.posecells, shift1, 0 ) * ( 1.0 - weight ) \
                             + np.roll( self.posecells, shift2, 0 ) * weight

    def __find_best__(self):
        '''
        Purpose: This routine finds an approximation of the center of the energy
        packet.

        Algorithm: First, find the max activated cell. Second, locate de cells
        that are in the area of PC_CELLS_TO_AVG distance. Third, get the sums
        for each axis. Then, find the (x, y, th) using population vector decoding
        to handle the wrap around.

        Inputs: -

        Outputs: position of the centre of the energy packet in the pose cell network
        '''
        # first: find the max activated cell
        z, y, x = np.unravel_index( np.argmax( self.posecells ), self.posecells.shape )

        z_posecells = np.zeros( [ PC_DIM_TH, PC_DIM_XY, PC_DIM_XY ] )

        # second: locate de cells that are in the area of PC_CELLS_TO_AVG distance
        zval = self.posecells[ np.ix_(
            PC_AVG_TH_WRAP[ z : z + PC_CELLS_TO_AVG * 2 + 1],
            PC_AVG_XY_WRAP[ y : y + PC_CELLS_TO_AVG * 2 + 1],
            PC_AVG_XY_WRAP[ x : x + PC_CELLS_TO_AVG * 2 + 1]
        )]

        z_posecells[np.ix_(
            PC_AVG_TH_WRAP[ z : z + PC_CELLS_TO_AVG * 2 + 1],
            PC_AVG_XY_WRAP[ y : y + PC_CELLS_TO_AVG * 2 + 1],
            PC_AVG_XY_WRAP[ x : x + PC_CELLS_TO_AVG * 2 + 1]
        )] = zval

        # third: get the sums for each axis
        x_sums = np.sum( np.sum( z_posecells, 1 ), 0 )
        y_sums = np.sum( np.sum( z_posecells, 2 ), 0 )
        th_sums = np.sum( np.sum( z_posecells, 2 ), 1 )
        th_sums = th_sums[ : ]

        # print x_sums, y_sums, th_sums
        # print '\n\n\n'
        # x_sums = np.random.randint(0,20)
        # y_sums = np.random.randint(0,20)
        # th_sums = np.random.randint(0,36)
        # then: find the (x, y, th) using population vector
        # decoding to handle the wrap around
        x = ( np.arctan2( np.sum( PC_XY_SUM_SIN_LOOKUP * x_sums ),
                        np.sum( PC_XY_SUM_COS_LOOKUP * x_sums ) ) * \
             PC_DIM_XY / ( 2 * np.pi ) -1) % ( PC_DIM_XY )

        y = ( np.arctan2( np.sum( PC_XY_SUM_SIN_LOOKUP * y_sums ),
                        np.sum( PC_XY_SUM_COS_LOOKUP * y_sums ) ) * \
             PC_DIM_XY / ( 2 * np.pi ) -1) % ( PC_DIM_XY )

        th = ( np.arctan2( np.sum( PC_TH_SUM_SIN_LOOKUP * th_sums ),
                         np.sum( PC_TH_SUM_COS_LOOKUP * th_sums ) ) * \
              PC_DIM_TH / ( 2 * np.pi ) -1) % ( PC_DIM_TH )
        
        # x = np.random.randint(0,20)%PC_DIM_XY
        # y = np.random.randint(0,20)%PC_DIM_XY
        # th = np.random.randint(0,60)%PC_DIM_XY
        # print x, y, th

        # if th == PC_DIM_TH:
        #     th = 0.0
        
        self.best_x  = x
        self.best_y  = y
        self.best_th = th
        #maximo       = self.posecells[th][y][x]
        # print th, y, x
        maximo       = self.posecells[int(th)][int(y)][int(x)]

        return ( x, y, th )

    def __get_min_delta__(self, d1, d2, maximo):
        '''
        Purpose: This routine finds the smallest distance between two specific
        points in the pose cell network in one of the axis, respecting the wrap
        connections.

        Algorithm: Calculates the difference between the first two inputs. Then,
        calculates the other distance between these two cells due to the wrap
        connections (maximo - absval). Finally, evaluate which one is the smaller.

        Inputs:
            d1: first coordinate
            d2: second coordinate
            maximo: posecell dimension

        Outputs: smaller distance between two specific points in the pose cell network
        in one of the axis
        '''
        absval = abs( d1 - d2 )
        return min( absval, maximo - absval )

    def __get_delta_pc__(self, x, y, th):
        '''
        Purpose: This routine calculates the distance between a specific position
        and the pose cell with the highest energy.

        Algorithm: Adjusts the orientation of the robot, subtracting the value of the
        heading direction variation between the current visual template and the previous
        one. Then, calculates the distance between the between a specific position
        and the pose cell with the highest energy.

        Inputs:
            x: x coordinate of a specific position
            y: y coordinate of a specific position
            th: th coordinate of a specific position

        Outputs: distance between a specific position and the pose cell with the
        highest energy
        '''
        # first: adjust the orientation of the robot, subtracting the value of the heading
        # direction variation between the current visual template and the previous one
        pc_th_corrected = self.best_th - self.vt_delta_pc_th
        if pc_th_corrected < 0:
            pc_th_corrected += PC_DIM_TH
        if pc_th_corrected >= PC_DIM_TH:
            pc_th_corrected -= PC_DIM_TH

        # calculate the distance between the etween a specific position
        # and the posecell with highest energy
        return np.sqrt( pow( self.__get_min_delta__( self.best_x, x, PC_DIM_XY ), 2 ) + \
                       pow( self.__get_min_delta__( self.best_y, y, PC_DIM_XY ), 2 ) + \
                       pow( self.__get_min_delta__( pc_th_corrected, th, PC_DIM_TH ), 2 ) )

    def __create_experience__(self):
        '''
        Purpose: This routine creates a new PosecellExperience object and add this
        to the collection.

        Algorithm: Find the current PosecellVisualTemplate, update current_exp
        with the id of the new PosecellExperience and create a new PosecellExperience.
        Then, add the new PosecellExperience to the collection and add its id to
        the collection of the current PosecellVisualTemplate.

        Inputs: -

        Outputs: id of the new PosecellExperience object
        '''
        # first: find the current PosecellVisualTemplate
        pcvt = self.visual_templates[ self.current_vt ]

        # second: update current_exp with the id of the new PosecellExperience
        self.current_exp = self.experiences.size

        # third: create a new PosecellExperience
        
        exp = PosecellExperience( self.best_x, self.best_y, self.best_th, # posecell position of this experience
                                  self.current_vt ) # visual template related to this experience

        # finally: add the new PosecellExperience to the collection and
        # add its id to the collection of the current PosecellVisualTemplate
        self.experiences = np.append( self.experiences, exp )
        pcvt.exps = np.append( pcvt.exps, self.current_exp )

        return self.current_exp



    def get_action(self):
        '''
        Purpose: This routine determines an action for the experience map's
        topological graph.

        Algorithm: First, check if odometry and visual template inputs were processed.
        Then, go through all the experiences associated with the current view and find
        the one closest to the current center of activity packet in the pose cell
        network. If an experience is closer than the threshold, creates a link.
        If there is an experience matching the current and exceeds the threshold,
        then the current experience should set to the previous one. Otherwise, creates
        a new experience.

        Inputs: -

        Outputs: action and matched experience
        '''
        action         = NO_ACTION
        matched_exp_id = -1

        min_delta    = DBL_MAX
        delta_pc = DBL_MAX

        # first: check if odometry and visual template inputs were processed
        if( self.odo_update and self.vt_update ):
            self.odo_update = False
            self.vt_update  = False
        else:
            return action, matched_exp_id

        if self.visual_templates.size == 0:
            #print "1"
            action = NO_ACTION
            return action, matched_exp_id

        if self.experiences.size == 0:
            #print "2"
            matched_exp_id = self.__create_experience__()
            action         = CREATE_NODE
        else:
            #print "3"
            experience = self.experiences[ int(self.current_exp) ]
            # distance between current experience position in the pose cell network
            # to the centre of activity packet
            delta_pc   = self.__get_delta_pc__( experience.x_pc, experience.y_pc, experience.th_pc )
            pcvt       = self.visual_templates[ self.current_vt ]

            if pcvt.exps.size == 0:
                #print "4"
                matched_exp_id = self.__create_experience__()
                action         = CREATE_NODE
            else:
                #print "5"
                #print self.current_vt
                if ( delta_pc > EXP_DELTA_PC_THRESHOLD or self.current_vt != self.prev_vt ):
                    #print "6"
                    # go through all the exps associated with the current view
                    # and find the one with the closest delta_pc (distance)
                    min_delta_id = -1
                    min_delta    = DBL_MAX

                    # find the closest experience in pose cell space
                    for index in pcvt.exps:
                        # make sure we aren't comparing to the current experience
                        if self.current_exp == index:
                            continue
                        experience = self.experiences[ int(index) ]
                        delta_pc   = self.__get_delta_pc__( experience.x_pc, experience.y_pc, experience.th_pc )

                        if delta_pc < min_delta:
                            min_delta    = delta_pc
                            min_delta_id = index
                    # if an experience is closer than the threshold, create a link
                    if min_delta < EXP_DELTA_PC_THRESHOLD:
                        #print "7"
                        matched_exp_id = min_delta_id
                        action         = CREATE_EDGE

                    # Matheus: 14/10/2018 
                    #           use this condition to create experiences

                    if self.current_exp != matched_exp_id:
                        #print "8"
                        if matched_exp_id == -1:
                            #print "9"
                            matched_exp_id = self.__create_experience__()
                            action         = CREATE_NODE
                        else:
                            #print "10"
                            self.current_exp = matched_exp_id
                            if action == NO_ACTION:
                                #print "11"
                                action = SET_NODE
                    else:
                        #print "12"
                        if self.current_vt == self.prev_vt:
                            #print "13"
                            matched_exp_id = self.__create_experience__()
                            action         = CREATE_NODE

        return action, matched_exp_id

    def on_odo(self, vtrans, vrot, time_diff_s):
        '''
        Purpose: This routine process the odometry information and start the
        dynamic in the pose cell network ( excitation, inhibition, global inhibition
        and path integration processes ).

        Algorithm: First, the pose cell network is locally excited, where energy
        is added around each active pose cell. Second, the pose cell network is
        locally inhibited, where energy is removed around each active pose cell.
        These first two steps ensure the stabilization of the energy packets. Third,
        global inhibition process happens, where energy is removed from all active
        pose cells but not below zero. Then, network energy normalization occurs to
        ensure the total energy in the system is equal to one. This stage ensures
        the stability of the global pose cell system. Then, path integration occurs, by
        shifting the pose cell energy. Finally, the centroid of the dominant activity
        packet in the network is identified.

        Inputs:
            vtrans: translational velocity
            vrot: rotational velocity
            time_diffs_s: time difference between the previous position and current
                position

        Outputs: -
        '''
        self.__excite__()
        self.__inhibit__()
        self.__global_inhibit__()
        self.__normalise__()
        self.__path_integration__( vtrans * time_diff_s, vrot * time_diff_s )
        self.best_x, self.best_y, self.best_th = self.__find_best__()
        self.odo_update = True

    def __create_view_template__(self):
        '''
        Purpose: This routine creates a new PosecellVisualTemplate object and add this
        to the collection.

        Algorithm: Create a new PosecellVisualTemplate object and then add this to the
        collection.

        Inputs: -

        Outputs: -
        '''
        pcvt = PosecellVisualTemplate( self.visual_templates.size, # id
                                       self.best_x, self.best_y, self.best_th, # posecell position of this visual template
                                       VT_ACTIVE_DECAY )
        self.visual_templates = np.append( self.visual_templates, pcvt )

    def on_view_template(self, vt, vt_rad):
        '''
        Purpose: This routine decides which action on a view template will be taken;
        inject energy or associate the current peak of activity to the view template.

        Algorithm: The action on a view template input depends on whether this
        is a new or existing view template. For new view templates, the id is associated
        with the centroid of the current peak activity packet in the pose cell network.
        For existing view templates, activity is injected into the previously associated
        location in the pose cells. The injected activity for consecutive matches of the
        same view template decays rapidly but is gradually restored over time.
        Because RatSLAM has no explicit motion model, this decay process is necessary
        to avoid potentially incorrect re-localizations when the robot is motionless
        for long periods of time.

        Inputs:
            vt: id of the current visual template
            vt_rad: relative angle between closest visual template and current visual
                template

        Outputs: -
        '''
        # first: check if it is a new visual template
        if vt >= self.visual_templates.size:
            self.__create_view_template__()
        else:
            pcvt = self.visual_templates[vt]
            if vt < ( self.visual_templates.size - 10 ):
                if vt != self.current_vt:
                    pass
                else:
                    pcvt.decay += VT_ACTIVE_DECAY
                # second: calculate the energy that will be injected in the pose cell network
                energy = PC_VT_INJECT_ENERGY * 1.0 / 30.0 * ( 30.0 - np.exp( 1.2 * pcvt.decay ) )

                # if this is higher then 0, should be injected in the posecell network
                if energy > 0:
                    self.vt_delta_pc_th = vt_rad / ( 2.0 * np.pi ) * PC_DIM_TH
                    pc_th_corrected     = pcvt.pc_th + vt_rad / ( 2.0 * np.pi ) * PC_DIM_TH
                    if pc_th_corrected < 0:
                        pc_th_corrected += PC_DIM_TH
                    if pc_th_corrected >= PC_DIM_TH:
                        pc_th_corrected -= PC_DIM_TH
                    self.__inject__( int( pcvt.pc_x ), int( pcvt.pc_y ), int( pc_th_corrected ), energy )

        # then: restore the energy of the visual template
        for visual_template in self.visual_templates:
            visual_template.decay -= PC_VT_RESTORE
            if visual_template.decay < VT_ACTIVE_DECAY:
                visual_template.decay = VT_ACTIVE_DECAY

        # finally: update the value of current_vt and previous vt and vt_update
        self.prev_vt    = self.current_vt
        self.current_vt = vt
        self.vt_update  = True

    def get_relative_rad(self):
        '''
        Purpose: This routine returns the relative angle between closest visual
        template and current visual template in radians.

        Algorithm: Calculates the relative angle between the closest visual
        template and current visual template in radians.

        Inputs: -

        Outputs: the relative angle in radians
        '''
        return self.vt_delta_pc_th * 2.0 * np.pi / PC_DIM_TH

    def __call__(self, lv_current_vt, lv_rad, vtrans, vrot, time_diff_s):
        self.on_view_template( lv_current_vt, lv_rad )
        self.on_odo( vtrans, vrot, time_diff_s )
        action, matched_exp_id = self.get_action( )
        return action, matched_exp_id

    def save(self, prefix):
        '''
        Purpose: This routine saves all the pose cell visual templates stored in the
        collection and all the pose cell experiences stored in the collection.

        Algorithm: Create a file to store information about all pose cell visual templates
        ( id, position in the pose cell network, decay ), then create a file to store
        information about all pose cell experiences ( position in the pose cell network,
        the id of the visual template associated )

        Inputs: -

        Outputs: -
        '''
        with open(  str(prefix) +'/posecellvisual_templates.txt', 'w') as file:
            for vt in self.visual_templates:
                file.writelines(str(vt.id) + '\n')
                file.writelines(str(vt.pc_x) + '\n')
                file.writelines(str(vt.pc_y) + '\n')
                file.writelines(str(vt.pc_th) + '\n')
                file.writelines(str(vt.decay) + '\n')
                np.savetxt(file, vt.exps, newline=" ")
                file.writelines('\n')
            file.writelines("-\n")
            file.writelines(str(self.current_vt) + '\n')
            file.writelines(str(self.prev_vt) + '\n')
        with open(  str(prefix) +'/posecellexperiences.txt', 'w') as file:
            for exp in self.experiences:
                file.writelines(str(exp.x_pc) + '\n')
                file.writelines(str(exp.y_pc) + '\n')
                file.writelines(str(exp.th_pc) + '\n')
                file.writelines(str(exp.vt_id) + '\n')
            file.writelines("-\n")
            file.writelines(str(self.current_exp) + '\n')
            file.writelines(str(self.prev_exp) + '\n')

    def load(self, prefix):
        '''
        Purpose: This routine loads all the visual templates saved and add them to
        the collection

        Algorithm: Open a file with all the pose cell visual template stored and for each
        pose cell visual template, create a new PosecellVisualTemplate object and store it
        in the collection. Open a file with all the pose cell experiences stored and for each
        pose cell experience, create a new PosecellExperience object and store it
        in the collection.

        Inputs: -

        Outputs: -
        '''
        with open(  str(prefix) +'/posecellvisual_templates.txt', 'r') as file:
            line = file.readline()
            while line != "-\n":
                id = float(line)
                line = file.readline()
                pc_x = float(line)
                line = file.readline()
                pc_y = float(line)
                line = file.readline()
                pc_th = float(line)
                line = file.readline()
                decay = float(line)
                line = file.readline()
                expsstr = np.array(line.split(" "))
                exps = np.array([])
                for exp in expsstr:
                    if exp != '\n':
                        exp = float(exp)
                        exps = np.append(exps, np.array(exp))
                line = file.readline()
                #print line
                pcvt = PosecellVisualTemplate(id, pc_x, pc_y, pc_th,decay)
                pcvt.exps = exps
                self.visual_templates = np.append(self.visual_templates, pcvt)
            line = file.readline()
            #print '->' + line
            self.current_vt = float(line)
            line = file.readline()
            self.prev_vt = file.readline()

        with open(  str(prefix) +'/posecellexperiences.txt', 'r') as file:
            line = file.readline()
            while line != "-\n":
                x_pc = float(line)
                line = file.readline()
                y_pc = float(line)
                line = file.readline()
                th_pc = float(line)
                line = file.readline()
                vt_id = float(line)
                line = file.readline()
                pcvt = self.visual_templates[int(vt_id)]
                exp = PosecellExperience(x_pc, y_pc, th_pc, vt_id)
                self.experiences = np.append(self.experiences, exp)
            line = file.readline()
            self.current_exp = float(line)
            line = file.readline()
            self.prev_exp = float(line)

    #MAURO -> FUNCAO DE MERGE DOS POSECELLS NETWORK

    def merge(self, visual_templates, experiences, id1, id2, pcvt_T, pcexp_T, pcvt_size, vt_size, pcexp_size):

        '''
        Purpose: This routine insert new posecells visual templates and posecells experiences into
        a existing estructure

        Algorithm: Insert into posecellsVisualTemplates and posecellsExperiences arrays the experiences.

        Inputs: pose_visual, pose_exp

        Outputs: -
        '''

        for pcvt in visual_templates:
            
            # print 'pcvt.id: ' + str(pcvt.id)
            pc_x       = (pcvt.pc_x  + pcvt_T[0]) % PC_DIM_XY
            
            pc_y       = (pcvt.pc_y  + pcvt_T[1]) % PC_DIM_XY
            
            pc_th      = (pcvt.pc_th + pcvt_T[2]) % PC_DIM_TH
            
            
            decay      = pcvt.decay
            exps       = pcvt.exps

            # debug

            # print 'id do pcvt: ' + str(pcvt.id)
            # print 'exps do novo pcvt: ' + str(pcvt.exps)
            
            for j in range (len(exps)):
                
                # print 'exp anterior: ' +str(exps[j])
                if exps[j] == id2:
                    exps[j] = id1

                else:
                    exps[j] = exps[j] + pcexp_size
            
            new_pcvt = PosecellVisualTemplate(pcvt.id + vt_size, pc_x, pc_y, pc_th, decay)
            new_pcvt.exps = exps

            # debug
            # print '\n'
            # print 'id do novo pcvt: ' + str(new_pcvt.id)
            # print 'exps do novo pcvt: ' + str(new_pcvt.exps)
            
            self.visual_templates = np.append(self.visual_templates, new_pcvt)
            # print 'id: ' + str(new_pcvt.id)
            
        # size_exp_fix = self.experiences.size * 1.0 

        for pcexp in experiences:

            x_pc  = (pcexp.x_pc  + pcexp_T[0]) % PC_DIM_XY
            y_pc  = (pcexp.y_pc  + pcexp_T[1]) % PC_DIM_XY
            th_pc = (pcexp.th_pc + pcexp_T[2]) % PC_DIM_TH
            # vt_id = self.experiences.size

            # atente para vt_id < id
            
            vt_id = vt_size + pcexp.vt_id * 1.0
            # print 'new pcexp.vt_id: ' + str(vt_id)

            new_exps = PosecellExperience(x_pc, y_pc, th_pc, vt_id)
            self.experiences = np.append(self.experiences, new_exps)
        
                
    # ------------------------------------------------------------[ Pruning ]

    def prune(self, exps):
                        deleted_exps = np.setdiff1d(exps[0], exps[1])
                        deleted_exps = np.sort(-deleted_exps)
                        deleted_exps = -deleted_exps
                        for vt in self.visual_templates:
                            vt.exps = np.setdiff1d(vt.exps, deleted_exps)
                            vt.exps = exps[2, vt.exps.astype('int64')]
                        for id in deleted_exps:
                            if id == self.experiences.size - 1:
                                self.experiences = self.experiences[:-1]
                            else:
                                self.experiences = np.append(self.experiences[:id], self.experiences[id + 1:])
