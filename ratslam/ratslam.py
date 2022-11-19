
# -------------------------------------------------------------------[ header ]

from .local_view_match import LocalViewMatch
from .posecellnetwork import PoseCellNetwork
from .visual_odometry import VisualOdometry
from .experience_map import ExperienceMap
from ._globals import *

# ------------------------------------------------------------[ RatSLAM class ]


class Ratslam(object):
    def __init__(self):
        '''
        visual-odometry: object of VisualOdometry class that controls odometry
        information
        visual-templates: object of LocalViewMatch class that controls the
        matching scenes
        network: object of PoseCellNetwork class that controls activity in the
        pose cell network and the actions in the experience map
        map: object of ExperienceMap class
        vtrans: translational velocity of the robot
        vrot: rotational velocity of the robot
        prev-vt: id of the previous visual template
        time-diff: the time difference between frames

        # not used
        frame_count: the number of frames that have been
        time_s: the time difference between frames
        '''
        self.visual_odometry  = VisualOdometry()
        self.visual_templates = LocalViewMatch()
        self.network          = PoseCellNetwork([PC_DIM_TH / 2,  PC_DIM_XY / 2, PC_DIM_XY / 2])
        self.map              = ExperienceMap()

        self.acum_vtrans = 0.0
        self.acum_vrot   = 0.0
        self.acum_time_s = 0
        self.prev_vt     = 0
        self.frame_count = 0
        self.time_s      = 0
        self.time_diff   = 0. 
        self.current_vt  = 0

        
        # Matheus 12/05/2020 - correcting first activation on PCN
        self.crct_pcvt   = False 

        x, y, th = self.visual_odometry.odometry
        self.odometry = [[x], [y], [th]]



    # def __call__(self, img, vtrans, vrot, greyscale):
    def __call__(self, img, p_odom, pvtrans, pvrot, ptime_diff, greyscale):
        '''
        Purpose: This routine updates the current position of the experience
        map since the last experience.

        Algorithm: First, calculates the translational velocity and rotational
        velocity between the current frame (data) and the previous frame.
        Second, finds the matching template and the angle between the current
        and previous template. Third, determines an action for the experience
        map's topological graph. Finally, updates the current position of the
        experience map since the last experience, performs the action from pose
        cell network and graph relaxation.

        Inputs:
            img: current frame
            v_odom: False if VisualOdometry() will be used, True if the
            odometry will be passed by external
            pvtrans: passed by external translational velocity of the robot
            pvrot: passed by external rotational velocity of the robot
            ptime-diff: passed by external the time difference between frames
            greyscale: True if the current frame is on grayscale, False if
            the current frame is not on grayscale

        Outputs: -
        '''

        # Matheus 14/10/2018  =>  REVISADO

        # first: calculate the translational velocity and rotational velocity
        # between the current frame (data) and the previous frame
        if not p_odom:
            vtrans, vrot       = self.visual_odometry( img, greyscale )
            # print('here')
        else: 
            vtrans             = pvtrans
            vrot               = pvrot
        
        
        self.time_diff         = ptime_diff
        self.acum_vtrans      += vtrans
        self.acum_vrot        += vrot
        self.acum_time_s      += self.time_diff

        # Matheus 14/10/2018  =>  REVISADO

        # visual_odometry.on_image() changes the odometry value. This odometry
        # is calculated by vtrans and vrot, and it's not used on experience map 
        # and posecell network

        if not p_odom :
            
            x, y, th = self.visual_odometry.odometry
            self.odometry[0].append(x)
            self.odometry[1].append(y)
            self.odometry[2].append(th)

        # Matheus 14/10/2018   =>  REVISADO

        # second: find the matching template and the angle between the current
        # and previous template

        current_vt, vt_rad      = self.visual_templates( img, greyscale )
        
        self.current_vt         = current_vt

        # third: determine an action for the experience map's topological graph
        action, matched_exp_id = self.network( current_vt, vt_rad, vtrans, vrot, self.time_diff )
        

        # finally: update the current position of the experience map since the
        # last experience, perform the action from pose cell network and graph
        # relaxation
        self.map.on_odo( vtrans, vrot, self.time_diff )
        

        if action == CREATE_NODE: # create new experience node
            self.map.on_create_experience( matched_exp_id, current_vt )
            self.map.on_set_experience( matched_exp_id, 0)

        if action == CREATE_EDGE: # link together previous exp node w/ current one
            self.map.on_create_link( self.map.current_exp_id, matched_exp_id, self.network.get_relative_rad() )
            self.map.on_set_experience( matched_exp_id, self.network.get_relative_rad())
            
        if action == SET_NODE: # matched exp is the current exp; no link & no new exp node
            self.map.on_set_experience( matched_exp_id, self.network.get_relative_rad() )
        
        self.map.iterate()

        if self.current_vt == 1 and self.crct_pcvt == False:
            self.network.visual_templates[0].pc_th = self.network.visual_templates[self.current_vt].pc_th - EXP_DELTA_PC_THRESHOLD/2 + 0.2
            self.crct_pcvt = True

        # self.prune()

        """
        # interface
        # get where the robot is and where it can go
        Later:
        current_exp, exps_goals = self.map.get_status()
        id = raw_input('digite qual exp')
        self.map.get_distance_and_direction(int(id))
        """

        """
        Later:
        self.map.calculate_path_to_goal(self.time_s)
        self.map.get_goal_waypoint()
        """

    def save(self, prefix):
        '''
        Purpose: This routine saves all the visual template, pose cell and
        experience map information..

        Algorithm: Creates files for each object and stores all the information.

        Inputs: -

        Outputs: -
        '''
        self.network.save(prefix)
        self.visual_templates.save(prefix)
        self.map.save(prefix)
    
    def load(self, prefix):
        '''
        Purpose: This routine loads all the visual template, pose cell and
        experience map information

        Algorithm: Opens files and loads all the information.

        Inputs: 
            - prefix: the prefix of directory name to find the files

        Outputs: -
        '''
        self.visual_templates.load(prefix)
        self.network.load(prefix)
        self.map.load(prefix)

    def prune(self):
        #TODO
        deleted_exps = self.map.prune(self.network.visual_templates.size)
        if deleted_exps.size > 1:
            self.network.prune(deleted_exps)
