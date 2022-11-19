
# -------------------------------------------------------------------[ header ]

import numpy as np
from ._globals import *
from .utils import *
import math

# This def rotates a map in relation with the [0,0] origin and angle in degrees
def rotMap(origin, point, angle):
    
    # angle = np.radians(angle)

    origen_x, origen_y = origin
    point_x, point_y = point

    rot_point_x = origen_x + math.cos(angle) * (point_x - origen_x) - math.sin(angle) * (point_y - origen_y)
    rot_point_y = origen_y + math.sin(angle) * (point_x - origen_x) + math.cos(angle) * (point_y - origen_y)
    return rot_point_x, rot_point_y

# def rotPath(p_origin, p_trans, angle):

#     xt, yt = p_trans
#     x, y   = p_origin

#     xr = (x * math.cos(angle)) - (y * math.sin(angle)) + xt
#     yr = (x * math.sin(angle)) + (y * math.cos(angle)) + yt

#     return xr, yr
# ------------------------------------------------------------[ Link class ]

class Link(object):
    def __init__(self):
        '''
        d: distance between two experiences
        heading_rad: heading direction/ orientation angle between the two experiences
        facing_rad: facing angle between the two experiences
        exp_to_id: the link ends at this experience
        exp_from_id: the link starts at this experience
        delta_time_s: time interval between two experiences
        '''
        self.d            = 0
        self.heading_rad  = 0
        self.facing_rad   = 0
        self.exp_to_id    = 0
        self.exp_from_id  = 0
        self.delta_time_s = 0

# ------------------------------------------------------------[ Experience class ]

class Experience(object):
    def __init__(self, id):
        '''
        id: id of the experience
        x_m: x coordinate of the position
        y_m: y coordinate of the position
        th_rad: heading direction/ orientation
        vt_id: id of the visual template associated to this experience
        links_from: collection of id of the links that starts at this experience
        links_to: collection of id of the links that ends at this experience

        # not used -- Later
        time_from_current_s
        goal_to_current
        current_to_goal
        '''
        self.id     = id
        self.x_m    = 0
        self.y_m    = 0
        self.th_rad = 0
        self.vt_id  = 0
        self.links_from = np.array([])
        self.links_to = np.array([])
        self.time_from_current_s = 0
        self.goal_to_current = 0
        self.current_to_goal = 0

# ------------------------------------------------------------[ ExperienceMap class ]

class ExperienceMap(object):
    def __init__(self):
        '''
        experiences: collection of Experience objects
        links: collection of Link objects

        current-exp-id: id of the current experience
        prev-exp-id: id of the previous experience

        accum-delta-facing: orientation movement due to the rotation
        accum-delta-x: space traveled in x axis due to translation
        accum-delta-y: space traveled in y axis due to translation
        accum-delta-time-s: time interval from the last position to the current

        relative-rad: relative angle difference between closest visual template
        and current visual template

        # not used -- Later
        goal_list: list of goals
        goal_path_final_exp_id
        waypoint_exp_id
        goal_timeout_s
        goal_sucess: controls if teh robot achieves the goal
        '''
        self.experiences            = np.array([])
        self.links                  = np.array([])

        self.current_exp_id         = 0
        self.prev_exp_id            = 0
        self.waypoint_exp_id        = -1
        self.goal_timeout_s         = 0
        self.goal_success           = False

        self.accum_delta_facing     = EXP_INITIAL_EM_DEG * np.pi / 180
        self.accum_delta_x          = 0
        self.accum_delta_y          = 0
        self.accum_delta_time_s     = 0

        self.relative_rad           = 0

        self.goal_list              = np.array([])
        self.goal_path_final_exp_id = 0

        # edited by matheus
        self.translation            = [0,0]

    def on_create_experience(self, id, vt_id):
        '''
        Purpose: This routine creates a new experience in the experience map.
        This experience will be associated with a visual template with vt-id as
        id.

        Algorithm: Creates a new experience, sets the position of this experience
        due to the translational movement, adds the new experience to the collection,
        then creates a link between the current experience and the previous one.

        Inputs:
            id: id of the new experience

            vt-id: id of the visual template associated with this experience

        Outputs: the id of the new experience
        '''
        # first: create a new experience
        new_exp = Experience( id )

        # second: set the position of this experience due to the translational movement
        if self.experiences.size == 0:
            new_exp.x_m    = 0
            new_exp.y_m    = 0
            new_exp.th_rad = 0
        else:
            new_exp.x_m    = self.experiences[ int(self.current_exp_id) ].x_m + self.accum_delta_x # dx due to translation
            new_exp.y_m    = self.experiences[ int(self.current_exp_id) ].y_m + self.accum_delta_y # dy due to translation
            new_exp.th_rad = clip_rad_180( self.accum_delta_facing )

        new_exp.vt_id           =  vt_id # setting the visual template id associated

        new_exp.goal_to_current = -1
        new_exp.current_to_goal = -1

        # third: add the new experience to the collection
        self.experiences = np.append(self.experiences, new_exp)

        # finally: create a link between the current experience and the previous one
        if (self.experiences.size != 1):
            self.on_create_link(self.current_exp_id, id, 0)

        return self.experiences.size - 1

    # MAURO -> FUNCAO DE MERGE DAS EXPERIENCIAS
    # MAURO -> FUNCAO DE MERGE DAS EXPERIENCIAS
    def on_create_experience_merge(self, exps, links, exp_T, id1, id2, map_size, link_size, vt_size):
        '''
        Purpose: This routine merges experiences from partial map in the experience map of loaded map.

        Algorithm: 

        Inputs:
            -
            -

        Outputs: -
        '''
        
        # taking the correct experience 
        for exp1 in self.experiences:
            if exp1.vt_id == id1:
                break

        # adding the new experiences according to their new ids
        for exp in exps[:-1]:

            # first: create a new experience
            new_exp = Experience( map_size + exp.id )
            
            # second: transfor this experience to the corrected coordinate
            new_exp.x_m, new_exp.y_m = rotMap(exp_T[3], [exp.x_m ,exp.y_m], exp_T[2])
            new_exp.x_m += exp_T[0]
            new_exp.y_m += exp_T[1]
            new_exp.th_rad = clip_rad_180(exp.th_rad + exp_T[2])

            # third: associate with the correct vt id
            new_exp.vt_id = vt_size + exp.vt_id  # setting the visual template id associated

            # fourth: correct the link information for the new experiences ids
            new_exp.links_from = exp.links_from
            # if new_exp.links_from != []:
            for i in range(len(new_exp.links_from)):
                new_exp.links_from[i] += link_size

            new_exp.links_to = exp.links_to
            
            for i in range(len(new_exp.links_to)):
                new_exp.links_to[i] += link_size
    
            new_exp.goal_to_current = -1
            new_exp.current_to_goal = -1

            # finally: add the new experience to the collection
            # self.current_exp_id = new_exp.id
            self.experiences = np.append(self.experiences, new_exp)
           
        # lastly, the link information between 
        for link in links:
            
            from_id = link.exp_from_id
            if int(from_id) == int(exps[-1].id):
                link.exp_from_id = exp1.id
            else:
                link.exp_from_id += map_size        
            
            to_id = link.exp_to_id
            if int(to_id) == int(exps[-1].id):
                link.exp_to_id = exp1.id
            else:
                link.exp_to_id += map_size
            
            link.heading_rad  = clip_rad_180(link.heading_rad + exp_T[4][0]) # + some delta threshold
            # # facing angle of the robot
            link.facing_rad   = clip_rad_180(link.facing_rad + exp_T[4][1]) # 
        
            self.links = np.append(self.links, link)

        for exp in self.experiences:
            print(exp.id)
            print(exp.links_from)
            print(exp.links_to)

        self.current_exp_id = exp1.id
        
        self.on_set_experience(self.current_exp_id, 0)

        return self.current_exp_id
            
        

    def on_odo(self, vtrans, vrot, time_diff_s):
        '''
        Purpose: This routine updates the current position of the experience map
        since the last experience.

        Algorithm: Calculates the traveled distance due to translational movement
        and the rotation. Then, updates the variables that control the position
        traveled in the experience map.

        Inputs:
            vtrans: translational velocity

            vrot: rotational velocity
            time-diff-s: time interval between the current experience and the
            previous

        Outputs: -
        '''
        # first: calculates the travelled distance due to translational movement
        # and the rotation
        vtrans *= time_diff_s
        vrot   *= time_diff_s
        '''
        print "####### DEBUG #######"
        #print "Time_Diff: " + str(time_diff_s)
        print "vtrans: " + str(vtrans)
        print "vrot: " + str(vrot)
        print "#####################"
        '''

        # second: updates the variables that controles the position travelled
        # in the experience map
        self.accum_delta_facing  = clip_rad_180(self.accum_delta_facing + vrot)
        self.accum_delta_x      += vtrans * np.cos(self.accum_delta_facing)
        self.accum_delta_y      += vtrans * np.sin(self.accum_delta_facing)
        self.accum_delta_time_s += time_diff_s
        '''
        print "####### DEBUG #######"
        print "accum_delta_facing: " + str(self.accum_delta_facing)
        print "accum_delta_x: " + str(self.accum_delta_x)
        print "accum_delta_y: " + str(self.accum_delta_y)
        print "#####################"
        '''
    
    def iterate(self):
        '''
        Purpose: This routine iterates the experience map. Perform a graph
        relaxing algorithm to allow the map to partially converge.

        Algorithm: Corrects the position of all experiences in the experience
        map based on the stored link information. A 0.5 correction parameter
        means that e0 and e1 will be fully corrected based on e0's link
        information.

        Inputs: -

        Outputs: -
        '''
        for i in range(EXP_LOOPS):
            for exp_from in self.experiences:
                links_from = exp_from.links_from[:]
                
                for indexlink in links_from:
                    #link = self.links[ indexlink ]
                    link = self.links[ int(indexlink) ]
                    exp_to = self.experiences[ int(link.exp_to_id) ]

                    # work out where e0 thinks e1 (x,y) should be based on the stored
                    # link information
                    lx = exp_from.x_m + link.d * np.cos( exp_from.th_rad + link.heading_rad )
                    ly = exp_from.y_m + link.d * np.sin( exp_from.th_rad + link.heading_rad )

                    # correct e0 and e1 (x,y) by equal but opposite amounts
                    # a 0.5 correction parameter means that e0 and e1 will be fully
                    # corrected based on e0's link information
                    exp_from.x_m += ((exp_to.x_m - lx) * EXP_CORRECTION)
                    exp_from.y_m += ((exp_to.y_m - ly) * EXP_CORRECTION)

                    exp_to.x_m -= ((exp_to.x_m - lx) * EXP_CORRECTION)
                    exp_to.y_m -= ((exp_to.y_m - ly) * EXP_CORRECTION)

                    # determine the angle between where e0 thinks e1's facing
                    # should be based on the link information
                    df = get_signed_delta_rad( exp_from.th_rad + link.facing_rad, exp_to.th_rad )

                    # correct e0 and e1 facing by equal but opposite amounts
                    # a 0.5 correction parameter means that e0 and e1 will be fully
                    # corrected based on e0's link information
                    exp_from.th_rad = clip_rad_180( exp_from.th_rad + df * EXP_CORRECTION )
                    exp_to.th_rad   = clip_rad_180( exp_to.th_rad - df * EXP_CORRECTION )

    def on_create_link(self, exp_id_from, exp_id_to, rel_rad):
        '''
        Purpose: This routine creates a new experience in the experience map.
        This experience will be associated with a visual template with vt-id as
        id.

        Algorithm: Checks if the current link already exists, then creates the
        new link and add this to the collection. Finally, adds the id of the
        new link to the collection of the experiences involved.

        Inputs:
            exp-id-from: the link starts at this experience

            exp-id-to: the link ends at this experience

            rel-rad: relative angle between these two experiencess

        Outputs: -
        '''
        # first: check if the current link already exists
        current_exp = self.experiences[ int(exp_id_from) ]
        total_links = current_exp.links_from
        
        total_links = np.append( total_links, current_exp.links_to )
        # print 'total link: ' + str(total_links)
        
        length = self.experiences.size
        #index  = np.array([ [ self.links[ link ].exp_to_id,
                              #self.links[ link ].exp_from_id ] for link in total_links ] )
        index  = np.array([ [ self.links[ int(link) ].exp_to_id,
                              self.links[ int(link) ].exp_from_id ] for link in total_links ] )
        index  = index.reshape( 1, index.size )
        if exp_id_to in index:
            return False

        # second: create the new Link
        new_link              = Link( )
        new_link.exp_to_id    = exp_id_to
        new_link.exp_from_id  = exp_id_from
        # d is the distance between experiences
        new_link.d            = np.sqrt( self.accum_delta_x ** 2 + self.accum_delta_y ** 2 )
        # heading direction angle
        new_link.heading_rad  = get_signed_delta_rad( current_exp.th_rad,
                                                    np.arctan2( self.accum_delta_y, self.accum_delta_x ) )
        # facing angle of the robot
        new_link.facing_rad   = get_signed_delta_rad( current_exp.th_rad,
                                                      clip_rad_180( self.accum_delta_facing + rel_rad ) )
        new_link.delta_time_s = self.accum_delta_time_s

        # third: add the new link to the collection
        self.links = np.append( self.links, new_link )

        # finally: add the id of the new link to the collection of the experiences involved
        self.experiences[ int(exp_id_from) ].links_from = np.append( self.experiences[int(exp_id_from)].links_from,
                                                             self.links.size - 1 )
        self.experiences[ int(exp_id_to) ].links_to = np.append( self.experiences[int(exp_id_to)].links_to,
                                                          self.links.size - 1 )

        # print self.experiences[ int(exp_id_from) ].links_from
        # print self.experiences[ int(exp_id_from) ].links_to

    #MAURO -> FUNCAO DE MERGE DOS LINKS
    def on_create_link_merge(self, link, exp_id_from, exp_id_to, rel_rad, time_fix, head_face_th):
        '''
        Purpose: This routine creates a new link with the values associates with a already 
                created link from slam2

        Algorithm: 

        Inputs:
            exp-id-from: the link starts at this experience

            exp-id-to: the link ends at this experience

            rel-rad: relative angle between these two experiencess

        Outputs: -
        '''
        
        new_link              = Link( )
        new_link.exp_to_id    = exp_id_to 
        new_link.exp_from_id  = exp_id_from
        # d is the distance between experiences
        new_link.d            = link.d
        # heading direction angle
        new_link.heading_rad  = get_signed_delta_rad(link.heading_rad , head_face_th[0])
        # facing angle of the robot
        new_link.facing_rad   = get_signed_delta_rad(link.facing_rad , head_face_th[1])

        new_link.heading_rad  = link.heading_rad
        
        new_link.facing_rad   = link.facing_rad

        self.accum_delta_time_s = link.delta_time_s + time_fix + 1
        new_link.delta_time_s = self.accum_delta_time_s

        # third: add the new link to the collection
        self.links = np.append( self.links, new_link )

        # # # finally: add the id of the new link to the collection of the experiences involved
        self.experiences[ int(exp_id_from) ].links_from = np.append( self.experiences[int(exp_id_from)].links_from,
                                                             self.links.size - 1 )
        self.experiences[ int(exp_id_to) ].links_to = np.append( self.experiences[int(exp_id_to)].links_to,
                                                          self.links.size - 1 )
        
        # print self.experiences[ int(exp_id_from) ].links_from
        # print self.experiences[ int(exp_id_from) ].links_to

    def on_set_experience(self, new_exp_id, rel_rad):
        '''
        Purpose: This routine changes the current experience.

        Algorithm: Check if it is necessary to make this change or if it is
        possible, then, update the previous and current experiences id. Finally,
        update the parameters of this class.

        Inputs:
            new-exp-id: id of the current experience

            rel-rad: relative head direction between the two experiences

        Outputs: 0, if the new experience was not changed to new-exp-id, 1, otherwise
        '''
        # first: check if it is necessary make this change or if it
        # is possible
        if new_exp_id > (self.experiences.size - 1):
            return 0
        if new_exp_id == self.current_exp_id:
            return 1

        # second: update the parameters
        self.prev_exp_id        = self.current_exp_id
        self.current_exp_id     = new_exp_id
        self.accum_delta_x      = 0 
        self.accum_delta_y      = 0 
        self.accum_delta_facing = clip_rad_180( self.experiences[ int(self.current_exp_id) ].th_rad + rel_rad )
        self.relative_rad       = rel_rad 
        return 1

    def get_status(self):
        '''
        Purpose: This routine returns the experiences that the robot can achieve
        from the current experience.

        Algorithm: First, check if there is more than one experience. Second,
        find all the links from the current experience and identify the experiences
        where these links end. Finally, find all the links that end at the current
        experience and identify the experiences where these links start.

        Inputs: -

        Outputs: id of the current experience and a list of experiences that
        the robot can achieve
        '''
        exp = self.experiences[self.current_exp_id]
        # first: check if there is more than one experience
        if self.experiences.size <= 1:
            return self.current_exp_id, np.array([])
        # second: find all the links from the current experience and identify
        # the experiences where these links end
        exps_goal = np.array([])
        links     = exp.links_from
        exps_goal = np.array([ self.links[link].exp_to_id for link in links ])
        # Finally, find all the links that ends at the current experience and
        # identify the experiences where these links start
        links     = exp.links_to
        exps_goal = np.append(exps_goal,np.array([ self.links[link].exp_from_id for link in links ]))

        return self.current_exp_id, exps_goal

    def get_distance_and_direction(self, exp_id):
        '''
        Purpose: This routine returns distance and direction to achieve the
        exp-id from current experience.

        Algorithm: First, find the links that the two experiences have in common.
        Finally, find the link with the smallest distance.

        Inputs:
            exp-id: id of the experience that the robot wants to achieve

        Outputs: distance and direction to achieve the exp-id from the current position
        '''
        exp_from = self.experiences[ self.current_exp_id ]
        exp_to   = self.experiences[ exp_id ]

        links_from    = np.intersect1d( exp_from.links_from, exp_to.links_to )
        links_to      = np.intersect1d( exp_from.links_to, exp_to.links_from )
        links         = np.append( links_from, links_to )

        d           = DBL_MAX
        index       = 0
        heading_rad = 0
        for indexlink in links:
            link = self.links[ indexlink ]
            if ( link.d < d ):
                d           = link.d
                index       = indexlink
                heading_rad = link.heading_rad

        return d, heading_rad

    def save(self, prefix):
        '''
        Purpose: This routine saves all the experiences, links and goals stored
        in the collection.

        Algorithm: Create a file to store information about all experiences
        ( id, position in the experience map, id of the visual template associated,
        collection of links to, collection of links from, time from current,
        goal to current, and current to goal), then create a file to store
        information about all links ( distance, heading direction, facing direction,
        experience to, experience from, time between these two experiences ),
        finally create a file to save the list of goals.

        Inputs: -

        Outputs: -
        '''
        with open(str(prefix) +'/experiences.txt', 'w') as file:
            for exp in self.experiences:
                file.writelines(str(exp.id) + '\n')
                file.writelines(str(exp.x_m) + '\n')
                file.writelines(str(exp.y_m) + '\n')
                file.writelines(str(exp.th_rad) + '\n')
                file.writelines(str(exp.vt_id) + '\n')
                np.savetxt(file, exp.links_from, newline=" ")
                file.writelines('\n')
                np.savetxt(file, exp.links_to, newline=" ")
                file.writelines('\n')
                file.writelines(str(exp.time_from_current_s) + '\n')
                file.writelines(str(exp.goal_to_current) + '\n')
                file.writelines(str(exp.current_to_goal) + '\n')
            file.writelines("-\n")
            file.writelines(str(self.current_exp_id) + '\n')
            file.writelines(str(self.prev_exp_id) + '\n')
        with open( str(prefix) +'/links.txt', 'w') as file:
            for link in self.links:
                file.writelines(str(link.d) + '\n')
                file.writelines(str(link.heading_rad) + '\n')
                file.writelines(str(link.facing_rad) + '\n')
                file.writelines(str(link.exp_to_id) + '\n')
                file.writelines(str(link.exp_from_id) + '\n')
                file.writelines(str(link.delta_time_s) + '\n')
        with open( str(prefix) +'/goal_list.txt', 'w') as file:
            np.savetxt(file, self.goal_list, newline=" ")

    def load(self, prefix):
        '''
        Purpose: This routine loads all the experiences, links and goals saved
        and add them to the collection.

        Algorithm: Opens a file with information about all experiences ( id,
        position in the experience map, id of the visual template associated,
        collection of links to, collection of links from, time from current,
        goal to current, and current to goal) and, for each experience, create
        an Experience object and add this to the collection.  It does the same
        for links. For goals, adds each goal stored to a list of goals.

        Inputs: -

        Outputs: -
        '''
        with open( str(prefix) +'/experiences.txt', 'r') as file:
            line = file.readline()
            while line != "-\n":
                id = float(line)
                line = file.readline()
                x_m = float(line)
                line = file.readline()
                y_m = float(line)
                line = file.readline()
                th_rad = float(line)
                line = file.readline()
                vt_id = float(line)
                line = file.readline()
                links = np.array(line.split(" "))
                links_from = np.array([])
                for link in links:
                    if link != '\n':
                        link = float(link)
                        links_from = np.append(links_from, np.array(link))
                line = file.readline()
                links = np.array(line.split(" "))
                links_to = np.array([])
                for link in links:
                    if link != '\n':
                        link = float(link)
                        links_to = np.append(links_to, np.array(link))
                line = file.readline()
                time_from_current_s = float(line)
                line = file.readline()
                goal_to_current = float(line)
                line = file.readline()
                current_to_goal = float(line)
                line = file.readline()
                new_exp = Experience(id)
                new_exp.vt_id = vt_id
                new_exp.x_m = x_m
                new_exp.y_m = y_m
                new_exp.th_rad = th_rad
                new_exp.vt_id = vt_id
                new_exp.links_from = links_from
                new_exp.links_to = links_to
                new_exp.time_from_current_s = time_from_current_s
                new_exp.goal_to_current = goal_to_current
                new_exp.current_to_goal = current_to_goal

                self.experiences = np.append(self.experiences, new_exp)
            line = file.readline()
            self.current_exp_id = float(line)
            line = file.readline()
            self.prev_exp_id = float(line)
        with open( str(prefix) +'/links.txt', 'r') as file:
            line = file.readline()
            while line != '':
                d = float(line)
                line = file.readline()
                heading_rad = float(line)
                line = file.readline()
                facing_rad = float(line)
                line = file.readline()
                exp_to_id = float(line)
                line = file.readline()
                exp_from_id = float(line)
                line = file.readline()
                delta_time_s = float(line)
                line = file.readline()
                new_link = Link()
                new_link.d = d
                new_link.heading_rad = heading_rad
                new_link.facing_rad = facing_rad
                new_link.exp_to_id = exp_to_id
                new_link.exp_from_id = exp_from_id
                new_link.delta_time_s = delta_time_s
                self.links = np.append(self.links, new_link)
        with open(str(prefix) +'/goal_list.txt', 'r') as file:
            line = file.readline()
            goals = np.array(line.split(" "))
            goal_list = np.array([])
            for goal in goal_list:
                if goal != '\n':
                    goal = float(goal)
                    goal_list = np.append(goal_list, np.array(goal))
            self.goal_list = goal_list


    # ------------------------------------------------------------[ Pruning ]

    def prune(self, vt_size):
            final_exps = np.array([np.arange(self.experiences.size)] * 3)

            for vt in range(vt_size):
                d = np.array([[]])
                exps = np.array([])
                for exp in self.experiences:
                    if exp.vt_id == vt:
                        exps = np.append(exps, exp.id)
                        if d.size == 0:
                            d = np.array([[exp.x_m, exp.y_m]])
                        else:
                            d = np.concatenate((d, [[exp.x_m, exp.y_m]]), axis=0)
                if exps.size > 1:
                    d = d[:, :] - d[0, :]
                    d = np.sqrt(np.sum(np.power(d, 2), 1))
                    merge_exps = np.array(exps[d < EXP_DELTA_PC_THRESHOLD])
                    if merge_exps.size > 1:
                        final_exps[1, merge_exps.astype('int64')] = merge_exps[0]
                        final_exps[2, merge_exps.astype('int64')] = -1
                        final_exps[2, merge_exps[0]] = merge_exps[0]
                        intersect = np.intersect1d(final_exps[0, :], final_exps[1, :])
                        final_exps[2, intersect] = np.arange(intersect.size)
                        self.merge_experiences(merge_exps)
            delexps = np.setdiff1d(final_exps[0, :], final_exps[1, :])
            delexps = np.sort(-delexps)
            delexps = -delexps
            print (delexps)
            for i in delexps:
                self.delete_experience(i)
            self.update_experience(final_exps)
            return final_exps

    def merge_experiences(self, exps):
            final_exp = self.experiences[exps[0]]
            final_exp_id = exps[0]
            exps = exps[1:]
            exps = np.sort(-exps)
            exps = -exps
            linksfrom = np.array([])
            linksto = np.array([])
            if self.current_exp_id in exps:
                self.current_exp_id = final_exp_id
            for exp in exps:
                print (exp, self.experiences.size)
                for index in self.experiences[exp].links_from:
                    link = self.links[index]
                    link.exp_from_id = final_exp_id
                    linksfrom = np.append(linksfrom, index)
                for index in self.experiences[exp].links_to:
                    link = self.links[index]
                    link.exp_to_id = final_exp_id
                    linksto = np.append(linksto, index)
            final_exp.links_from = np.append(final_exp.links_from, linksfrom)
            final_exp.links_to = np.append(final_exp.links_to, linksto)

    def delete_experience(self, id):
            if id == self.experiences.size - 1:
                self.experiences = self.experiences[:-1]
            else:
                self.experiences = np.append(self.experiences[:id], self.experiences[id + 1:])

    def update_experience(self, final_exps):
            for link in self.links:
                link.exp_to_id = final_exps[2, link.exp_to_id]
                link.exp_from_id = final_exps[2, link.exp_from_id]
            for exp in self.experiences:
                exp.id = final_exps[2, exp.id]
            self.current_exp_id = final_exps[2, self.current_exp_id]
    
    
            
    # ------------------------------------------------------------[ Not used ]

    # def compare(self, exp1, exp2):
    #     return exp1.time_from_current_s > exp2.time_from_current_s

    # def exp_euclidean_m(self, exp1, exp2):
    #     return np.sqrt((exp1.x_m - exp2.x_m) ** 2 + (exp1.y_m - exp2.y_m) ** 2)

    # def add_goal(self, x_m, y_m):
    #     min_id = -1
    #     min_dist = DBL_MAX
    #     if MAX_GOALS != 0 & self.goal_list.size >= MAX_GOALS:
    #         return
    #     length = self.experiences.size
    #     dist = np.array(
    #         [np.sqrt((self.experiences[index].x_m - x_m) ** 2 + (self.experiences[index].y_m - y_m) ** 2) for index in
    #          range(length)])

    #     min_id_val = np.argmin(dist)
    #     dist_value = dist[min_id_val]

    #     if dist_value < min_dist:
    #         min_dist = dist_value
    #         min_id = min_id_val

    #     if min_dist < 0.1:
    #         self.add_goal_to_list(min_id)

    # def add_goal_to_list(self, idt):
    #     self.goal_list = np.append(self.goal_list, idt)

    # def calculate_path_to_goal(self, time_s):
    #     self.waypoint_exp_id = -1
    #     if self.goal_list.size == 0:
    #         return False
    #     if (self.exp_euclidean_m(self.experiences[self.current_exp_id], self.experiences[self.goal_list[0]]) < 0.1 or ((self.goal_timeout_s != 0) and time_s > self.goal_timeout_s)):
    #         if (self.goal_timeout_s != 0 and time_s > self.goal_timeout_s):
    #             self.goal_success = False
    #         if (self.exp_euclidean_m(self.experiences[self.current_exp_id], self.experiences[self.goal_list[0]]) < 0.1):
    #             self.goal_success = True
    #         np.delete(self.goal_list, 0)
    #         self.goal_timeout_s = 0
    #         for exp in self.experiences:
    #             exp.time_from_current_s = DBL_MAX

    #     if (self.goal_list.size == 0):
    #         return False

    #     if (self.goal_timeout_s == 0):
    #         for exp in self.experiences:
    #             exp.time_from_current_s = DBL_MAX
    #         self.experiences[self.current_exp_id].time_from_current_s = 0
    #         self.goal_path_final_exp_id = self.current_exp_id
    #         exp_heap = np.array([(exp.idt, exp.time_from_current_s) for exp in self.experiences], dtype='float,float')
    #         exp_heap = np.sort(exp_heap, order='f1')

    #         while exp_heap.size != 0:
    #             exp_actual = self.experiences[exp_heap[0][0]]
    #             if (exp_actual.time_from_current_s == DBL_MAX):
    #                 np.delete(self.goal_list, 0)
    #                 return False
    #             np.delete(exp_heap, 0)
    #             for indexlink in exp_actual.links_to:
    #                 link = self.links[indexlink]
    #                 link_time_s = exp_actual.time_from_current_s + link.delta_time_s
    #                 if link_time_s < self.experiences[link.exp_from_id].time_from_current_s:
    #                     self.experiences[link.exp_from_id].time_from_current_s = link_time_s
    #                     self.experiences[link.exp_from_id].goal_to_current = exp_actual.id

    #             for indexlink in exp_actual.links_from:
    #                 link = self.links[indexlink]
    #                 link_time_s = exp_actual.time_from_current_s + link.delta_time_s
    #                 if link_time_s < self.experiences[link.exp_to_id].time_from_current_s:
    #                     self.experiences[link.exp_to_id].time_from_current_s = link_time_s
    #                     self.experiences[link.exp_to_id].goal_to_current = exp_actual.id
    #             if exp_heap.size != 0:
    #                 exp_heap = np.sort(exp_heap, order='f1')

    #         trace_exp_id = self.goal_list[0]
    #         while trace_exp_id != self.current_exp_id:
    #             self.experiences[self.experiences[trace_exp_id].goal_to_current].current_to_goal = trace_exp_id
    #             trace_exp_id = self.experiences[trace_exp_id].goatl_to_current

    #         if self.goal_timeout_s == 0:
    #             self.goal_timeout_s = time_s + self.experiences[self.goal_list[0]].time_from_current_s
    #     return True

    # def get_goal_waypoint(self):
    #     if self.goal_list.size == 0:
    #         return False
    #     self.waypoint_exp_id = -1
    #     trace_exp_id = self.goal_list[0]
    #     robot_exp = self.experiences[self.current_exp_id]

    #     while (trace_exp_id != self.goal_path_final_exp_id):
    #         dist = self.exp_euclidean_m(self.experiences[trace_exp_id], robot_exp)
    #         self.waypoint_exp_id = self.experiences[trace_exp_id].id
    #         if dist < 0.2:
    #             break
    #         trace_exp_id = self.experiences[trace_exp_id].goal_to_current

    #     if self.waypoint_exp_id == -1:
    #         self.waypoint_exp_id = self.current_exp_id
    #     return True

    # def get_subgoal_m(self):
    #     if self.waypoint_exp_id == -1:
    #         return 0
    #     else:
    #         return np.sqrt(
    #             pow((self.experiences[self.waypoint_exp_id].x_m - self.experiences[self.current_exp_id].x_m),
    #                    2) + pow(
    #                 (self.experiences[self.waypoint_exp_id].y_m - self.experiences[self.current_exp_id].y_m), 2))

    # def get_subgoal_rad(self):
    #     if self.waypoint_exp_id == -1:
    #         return 0
    #     else:
    #         curr_goal_rad = np.arctan2(
    #             (self.experiences[self.waypoint_exp_id].y_m - self.experiences[self.current_exp_id].y_m),
    #             (self.experiences[self.waypoint_exp_id].x_m - self.experiences[self.current_exp_id].x_m))
    #         return get_signed_delta_rad(self.experiences[self.current_exp_id].th_rad, curr_goal_rad)

    # def clear_goal_list(self):
    #     self.goal_list = np.array([])

    # def get_current_goal_id(self):
    #     if self.goal_list.size == 0:
    #         return -1
    #     else:
    #         return self.goal_list[0]

    # def delete_current_goal(self):
    #     self.goal_list = np.delete(self.goal_list, 0)

    # def dijkstra_distance_between_experiences(self, id1, id2):
    #     for exp in self.experiences:
    #         exp.time_from_current_s = DBL_MAX
    #     self.experiences[id1].time_from_current_s = 0
    #     self.goal_path_final_exp_id = self.current_exp.id
    #     exp_heap = np.array([(exp.idt, exp.time_from_current_s) for exp in self.experiences], dtype='float,float')
    #     exp_heap = np.sort(exp_heap, order='f1')

    #     while exp_heap.size != 0:
    #         exp_actual = self.experiences[exp_heap[0][0]]
    #         if (exp_actual.time_from_current_s == DBL_MAX):
    #             return DBL_MAX
    #         np.delete(exp_heap, 0)
    #         for indexlink in exp_actual.links_to:
    #             link = self.links[indexlink]
    #             link_time_s = exp_actual.time_from_current_s + link.delta_time_s
    #             if link_time_s < self.experiences[link.exp_from_id].time_from_current_s:
    #                 self.experiences[link.exp_from_id].time_from_current_s = link_time_s
    #                 self.experiences[link.exp_from_id].goal_to_current = exp_actual.id

    #         for indexlink in exp_actual.links_from:
    #             link = self.links[indexlink]
    #             link_time_s = exp_actual.time_from_current_s + link.delta_time_s
    #             if link_time_s < self.experiences[link.exp_to_id].time_from_current_s:
    #                 self.experiences[link.exp_to_id].time_from_current_s = link_time_s
    #                 self.experiences[link.exp_to_id].goal_to_current = exp_actual.id

    #         if exp_actual.id == id2:
    #             return exp_actual.time_from_current_s
    #     return DBL_MAX
