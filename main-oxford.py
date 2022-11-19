'''

RatSLAM example for mapa merging.

'''
from functools import partial, partialmethod
from gettext import find
from operator import invert
import cv2
import numpy as np
import itertools
import sys
import os
import csv

# Importing ratslam modules 
from ratslam import ratslam
# from ratslam import modratslam
from ratslam import _globals as gb

sys.path.insert(0, '/home/matheus/multisession-ratslam/ratslam/')
from merge import *
from utils import *

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
matplotlib.use('TkAgg')
from matplotlib import pyplot as plot
import matplotlib.lines as mlines


import math


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


# For load ground truth if exists 
def loadMap(filename):
    # ==========================================================
    # LOAD A GROUND TRUTH IF EXISTS
    map = []
    
    with open(filename, 'r') as file_map:

        line = file_map.readlines()

    px = []
    py = []
    pth = []
    for i in range(len(line)):

        x_value = [x.strip() for x in line[i].split(',')]
        px.append(float(x_value[0]))
        py.append(float(x_value[1]))
        pth.append(float(x_value[2]))

    for i in range(0, len(px)):
        point = [px[i], py[i], pth[i]]
        # point[1] = float(y_value[i])
        map.append(point)
    
    return px, py, pth


# Fast plot
def plotResult(slam, c_color='navy', c_color_edge='navy'):

    plot.clf()

    plot.title('Experience Map')
    
    xs = []
    ys = []
    
    for exp in slam.map.experiences:

        xs.append(exp.x_m)
        ys.append(exp.y_m)


    plot.scatter(xs, ys, color=c_color, marker='.')
    plot.plot(xs, ys, color=c_color, ls='-')
    end_m     = plot.scatter(slam.map.experiences[int(slam.map.current_exp_id)].x_m, slam.map.experiences[int(slam.map.current_exp_id)].y_m, s=200, c=c_color,  edgecolors=c_color_edge, linewidths=1, marker='D')
    
    start_m   = plot.scatter(xs[0], ys[0], s=220, c=c_color, edgecolors=c_color_edge, linewidths=1, marker='X')
    # end_m     = plot.scatter(xs[-1], ys[-1], s=200, c=c_color,  edgecolors=c_color_edge, linewidths=1, marker='D')

    # legend1 = plot.legend([start_m, end_m], ['Start of L-map','End of L-map'], loc='upper right')
    
    # plot.gca().add_artist(legend1)

    plot.xlabel("x(m)")
    plot.ylabel("y(m)")

    plot.tight_layout()

    plot.pause(0.01)


def plotSaveResult(slam_l, slam_p, name, c_color, c_color_edge, c_color2, c_color_edge2, save=False):

    plot.clf()

    plot.title('Experience Map')
    
    xs_slam1 = []
    ys_slam1 = []
    
    xs_slam2 = []
    ys_slam2 = []
    

    for exp in slam_l.map.experiences:

        xs_slam1.append(exp.x_m + 100)
        ys_slam1.append(exp.y_m + 100)

    # print('\n')
    # for link in slam_l.map.links:

    #     print("link id exp_from: " + str(link.exp_from_id) )
    #     print("link id exp_to: " + str(link.exp_to_id) )
    #     print("link distance: " + str(link.d) )
    #     print("link heading rad: " + str(link.heading_rad) )
    #     print("link facing rad: " + str(link.facing_rad) )
    
    if slam_p != None:
        for exp in slam_p.map.experiences:
            xs_slam2.append(exp.x_m)
            ys_slam2.append(exp.y_m)

    
    plot.scatter(xs_slam1[1:-1], ys_slam1[1:-1], c=c_color, edgecolors=c_color_edge, linewidths=1, marker='.')
    plot.scatter(xs_slam1[0], ys_slam1[0], s=220, c=c_color, edgecolors=c_color_edge, linewidths=1, marker='X')
    plot.scatter(xs_slam1[-1], ys_slam1[-1], s=200, c=c_color,  edgecolors=c_color_edge, linewidths=1, marker='D')
    # plot.plot(xs_slam1[1:-1], ys_slam1[1:-1], c=c_color, ls='-')
    
    if slam_p != None:
        plot.scatter(xs_slam2[1:-1], ys_slam2[1:-1], c=c_color2, edgecolors=c_color_edge2, linewidths=1, marker='.')
        plot.scatter(xs_slam2[0], ys_slam2[0], s=220, c=c_color2, edgecolors=c_color_edge2, linewidths=1, marker='X')
        plot.scatter(xs_slam2[-1], ys_slam2[-1], s=200, c=c_color2,  edgecolors=c_color_edge2, linewidths=1, marker='D')
        # plot.plot(xs_slam2[1:-1], ys_slam2[1:-1], c=c_color2, ls='-')
    
    
    
    # legend1 = plot.legend([start_l, end_l], ['Start of L-map','End of L-map'], loc='upper left')
    # legend2 = plot.legend([start_p, end_p], ['Start of P-map','End of P-map'], loc='upper right')
    # # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
    # plot.gca().add_artist(legend1)
    # plot.gca().add_artist(legend2)
    
    
    ### complete ####
    # plot.xlim(-1.2, 2.25)
    # plot.ylim(-0.5, 2.75)

    ### loaded ####
    # plot.xlim(-11, 11)
    # plot.ylim(-11, 11)

    ### partial ####
    # plot.xlim(-0.3, 1.3)
    # plot.ylim(-0.25, 1.6)

    plot.xlabel("x(m)")
    plot.ylabel("y(m)")

    # -----------------------------


    # Save directories
    if save:
        plot.savefig(r'/home/matheus/multisession-ratslam/em_'+str(name)+'.pdf', format='pdf', dpi=400)
        plot.savefig(r'/home/matheus/multisession-ratslam/em_'+str(name)+'.png', format='png') 

    ################################################### END #########################################################################

    plot.tight_layout()
    plot.pause(0.1)


def plotSaveResult_merge(slam, name, c_color1, c_color_edge1, c_color2, c_color_edge2, size_slam1=0):
    # =========================================================
    # PLOT THE CURRENT RESULTS ================================
    plot.clf()

    plot.title('Experience Map')


    xs_slam1 = []
    ys_slam1 = []

    xs_slam2 = []
    ys_slam2 = []

    if size_slam1 !=0:

        for exp in slam.map.experiences:

            if len(xs_slam1) < size_slam1:
                xs_slam1.append(exp.x_m)
                ys_slam1.append(exp.y_m)
            
            else:
                xs_slam2.append(exp.x_m)
                ys_slam2.append(exp.y_m)
        

        #SLAM1
        
        # plot.plot(xs_slam1, ys_slam1, color='navy', ls='-')
        
        plot.scatter(xs_slam1[1:-1], ys_slam1[1:-1], c=c_color1, marker='.')
        start_m1   = plot.scatter(xs_slam1[0], ys_slam1[0], s=220, c=c_color1, marker='X')
        end_m1     = plot.scatter(xs_slam1[-1], ys_slam1[-1], s=220, c=c_color1, marker='D')
        
        # legend1 = plot.legend([start_m1, end_m1], ['Start of L-map','End of L-map'], loc='upper left')
        # plot.gca().add_artist(legend1)
        #SLAM2

        # plot.plot(xs_slam2, ys_slam2, color=c_color2, ls='-')

        if xs_slam2 != [] and xs_slam2 != []:
            plot.scatter(xs_slam2[1:-1], ys_slam2[1:-1], c=c_color2, s=80, edgecolors=c_color_edge2, linewidths=1, marker='.')
            start_m2   = plot.scatter(xs_slam2[0], ys_slam2[0], s=220, c=c_color2, edgecolors=c_color_edge2, linewidths=1, marker='X')
            end_m2     = plot.scatter(xs_slam2[-1], ys_slam2[-1], s=220, c=c_color2,  edgecolors=c_color_edge2, linewidths=1, marker='D')
            # legend2 = plot.legend([start_m2, end_m2], ['Start of P-map','End of P-map'], loc='upper right')
            # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
            
            # plot.gca().add_artist(legend2)
    
    else:
        

        # for exp in slam.map.experiences:

        #     xs_slam1.append(exp.x_m)
        #     ys_slam1.append(exp.y_m)

        for exp in slam.map.experiences:
            
            rot_x, rot_y = rotate([0, 0], [exp.x_m, exp.y_m], 0.5)

            xs_slam1.append(rot_x)
            ys_slam1.append(rot_y)
            
        
        start_l   = plot.scatter(xs_slam1[0], ys_slam1[0], s=220, c=c_color1, edgecolors=c_color_edge1, linewidths=1, marker='X')
        plot.scatter(xs_slam1[1:-1], ys_slam1[1:-1], c=c_color1, edgecolors=c_color_edge1, linewidths=1, marker='.')
        end_l     = plot.scatter(xs_slam1[-1], ys_slam1[-1], s=200, c=c_color1,  edgecolors=c_color_edge1, linewidths=1, marker='D')
        
    
        plot.legend([start_l, end_l], ['Start of map','End of map'], loc="upper left")
        

    plot.xlabel("x(m)")
    plot.ylabel("y(m)")

    # plot.ylim(0, 120)
    # plot.ylim(-2.75, -0.2)
    
    # -----------------------------
    plot.savefig(r'/home/matheus/multisession-ratslam/figures/em_'+str(name)+'.pdf', format='pdf', dpi=400) 
    plot.savefig(r'/home/matheus/multisession-ratslam/figures/em_'+str(name)+'.png', format='png') 


    plot.pause(0.1)


def load_and_plot():

    slam = ratslam.Ratslam()
    partial1 = ratslam.Ratslam()
    partial2 = ratslam.Ratslam()

    partial1.load('/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_partial1_test')
    partial2.load('/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_partial2')
    
    slam.load("/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_final")

    # plotSaveResult_Merge(slam, "oxford_final", 'navy', 'navy', 'orange', 'r', partial1.map.experiences.size + partial2.map.experiences.size)

def RatSlam_by_load(mergeSlam , time_diff):
    
    '''
        RatSLAM difinition to recovery frames and odom data already processed 
    '''
    slam = ratslam.Ratslam()
    partial1 = ratslam.Ratslam()
    partial2 = ratslam.Ratslam()
    

    # =================================================
    # FOR MERGE SLAM WITH LOADED IMAGES AND ODOM ======    
    
    if mergeSlam:

        frames_repo   = '/home/matheus/data-ratslam/oxford/frames/'
        odom_file   = '/home/matheus/data-ratslam/oxford/oxford_odom.csv'
        frame_file = '/home/matheus/data-ratslam/oxford/oxford_frames.csv'

        ## get files 
        # list to store files
        res = []
        # Iterate directory
        for path in os.listdir(frames_repo):
            # check if current path is a file
            if os.path.isfile(os.path.join(frames_repo, path)):
                res.append(path)
        res.sort()

        ## get and list odometry values with stamps
        with open(odom_file, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

            stamp_sec_odom = []
            stamp_nsec_odom = []
            linear = []
            angular = []
            
            for i in range(1,len(data)):
                stamp_sec_odom.append(float(data[i][2]))
                stamp_nsec_odom.append(float(data[i][3]))
                linear.append(float(data[i][4]))
                angular.append(float(data[i][5]))

        ## get and list odometry values with stamps
        with open(frame_file, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

            stamp_sec_frame = []
            stamp_nsec_frame = []

            for i in range(1,len(data)):
                stamp_sec_frame.append(float(data[i][2]))
                stamp_nsec_frame.append(float(data[i][3]))

        
        merged1       = ratslam.Ratslam()
        merged2       = ratslam.Ratslam()

        partial1.load('/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_partial1_test')
        partial2.load('/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_partial2_test')

        # plotSaveResult_Merge(partial1, "oxford_loaded1", 'navy', 'navy', 'navy', 'navy', partial1.map.experiences.size)
        # plotSaveResult_Merge(partial2, "oxford_loaded2", 'navy', 'navy', 'navy', 'navy', partial2.map.experiences.size)

        # merged1 = partial1

        find_merge = False
        count = 0

        # part1 = list(range(2200,2960))
        # part2 = list(range(6440, len(res)))
        # it = part1 + part2
        count_success = 0 # for correctly find a match in the second map

        prev_i = 1941
        prev_sec = stamp_sec_frame[prev_i]

        count_part_2 = 0
        # for index in range(prev_i,len(res),3):
        for index in range(prev_i,4510,3):

        # for index in it:
            
            # get frame
            frame_prefx = frames_repo + res[index]
            frame = cv2.imread(frame_prefx)
            img_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            img_gray = np.array( img_gray )

            # get odom
            sec_stamp_frame = stamp_sec_frame[index]
            nsec_stamp_frame = stamp_nsec_frame[index]

            indexes_odom =  [i for i,x in enumerate(stamp_sec_odom) if x==sec_stamp_frame]
            # print(indexes_odom)
            best_i = 0
            diff = 10000000000
            for i in indexes_odom:
                if (abs(nsec_stamp_frame-stamp_nsec_odom[i]) <= diff):
                    diff = abs(nsec_stamp_frame-stamp_nsec_odom[i])
                    best_i = i
            # print(best_i)
            time_diff = sec_stamp_frame - prev_sec

            if find_merge == False and count==0:
                
                gb.EXP_CORRECTION = 0.5

                # WITH FRAMESKIP
                # slam(img_gray, True, np.mean(linear[prev_i:best_i]), np.mean(angular[prev_i:best_i]), time_diff, True)
                #WITHOUT FRAMESKIP
                slam(img_gray, True, linear[best_i], angular[best_i], time_diff, True)

                plotResult(slam)

                isTrue, vt_id = partial1.visual_templates.on_image_merge(slam.visual_templates.current_view, slam.visual_templates.current_mean)
                vt_relative_rad = partial1.visual_templates.vt_relative_rad

                # slam.save("/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_loaded")
                
                if isTrue == True and slam.current_vt != 0 and vt_id!=0:
                    
                    count_success = 0
                    print ('Encontrou match id: ' + str(vt_id))
                    print ('Actual slam id: ' + str(slam.current_vt))

                    size_partial1 = partial1.map.experiences.size
                    merged1 = merge(partial1, slam, vt_id, slam.current_vt)
                    # merged1.merge = True
                    # gb.EXP_CORRECTION = 0.0
                    # merged1.map.iterate()

                    # merged1.save("/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_merged1")

                    print(f"merged1 = {merged1.map.experiences.size}, partial1 = {partial1.map.experiences.size}")
                    # plotSaveResult_Merge(merged1, "merged1", 'navy', 'navy', 'orange', 'red', size_partial1)
                    

                    # plotSaveResult_merge(merged1, "merge_loop1-"+str(index), "navy", "navy", "orange", "orange", partial1.map.experiences.size)
                    count+=1
                    continue

                # if isTrue == True:
                #     count_success += 1

            if find_merge == False and count==1:

                # merged1(img_gray, True, np.mean(linear[prev_i:best_i]) * - 1 , np.mean(angular[prev_i:best_i]), time_diff, True)
                
                merged1(img_gray, True, linear[best_i], angular[best_i], time_diff, True)
                
                # plotSaveResult_Merge(merged1, "merged1", 'navy', 'navy', 'navy', 'navy', merged1.map.experiences.size)

                isTrue, vt_id = partial2.visual_templates.on_image_merge(merged1.visual_templates.current_view, merged1.visual_templates.current_mean)

                if isTrue == True and merged1.current_vt != 0 and vt_id!=0:

                    print ('Encontrou match id: ' + str(vt_id))
                    print ('Actual slam id: ' + str(merged1.current_vt))
               
                    size_partial1_and_2 = size_partial1 + partial2.map.experiences.size
                    
                    merged2 = merge(partial2, merged1, vt_id, merged1.current_vt)
                    
                    # plotSaveResult_Merge(merged2, "merged2", 'navy', 'navy', 'orange', 'red', size_partial1_and_2)

                    find_merge = True
                    merged2.merge = True

                    count+=1
                    continue
                
                # if isTrue:
                #     count_success+=1
                #     print(f"count part 2: {count_part_2}")

            if find_merge == True:
                
                
                merged2(img_gray, True, linear[best_i], angular[best_i], time_diff, True)

                # plotSaveResult_Merge(merged2, "merged2", 'navy', 'navy', 'orange', 'red', size_partial1_and_2)


            prev_i = best_i
            prev_sec = sec_stamp_frame

        # =================================================
        # SLAM AND IMAGE SAVE =============================  
        # plotSaveResult(merged2, None, "final", 'g', 'g', 'r', 'r', True)
        
        
        merged2.save("/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_final_test")
        
        
        # plotSaveResult_merge(prev_slam, "merge_loop4-"+str(index), "navy", "navy", "orange", "red", prev_slam_size)
        # plotResult_2slam(pos_merge, prev_slam_size, slam2_size, "paper_irat_1-0")
        # plotSaveResult(pos_merge, "paper_irat_4-0", 'b')  
        # pos_merge.save("/home/matheus/multisession-ratslam/ratslam/Saves/paper_irat_2")  
            
    # =================================================
    # FOR ONLY SLAM WITH LOADED IMAGES AND ODOM =======

    else: #only slam

        # 2973 - 5700
        # Data format
        frames_repo   = '/home/matheus/data-ratslam/oxford/frames/'
        odom_file   = '/home/matheus/data-ratslam/oxford/oxford_odom.csv'
        frame_file = '/home/matheus/data-ratslam/oxford/oxford_frames.csv'

        ## get files 
        # list to store files
        res = []
        # Iterate directory
        for path in os.listdir(frames_repo):
            # check if current path is a file
            if os.path.isfile(os.path.join(frames_repo, path)):
                res.append(path)
        res.sort()

        ## get and list odometry values with stamps
        with open(odom_file, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

            stamp_sec_odom = []
            stamp_nsec_odom = []
            linear = []
            angular = []
            
            for i in range(1,len(data)):
                stamp_sec_odom.append(float(data[i][2]))
                stamp_nsec_odom.append(float(data[i][3]))
                linear.append(float(data[i][4]))
                angular.append(float(data[i][5]))

        ## get and list odometry values with stamps
        with open(frame_file, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

            stamp_sec_frame = []
            stamp_nsec_frame = []

            for i in range(1,len(data)):
                stamp_sec_frame.append(float(data[i][2]))
                stamp_nsec_frame.append(float(data[i][3]))
        
        prev_i = 102
        prev_sec = stamp_sec_frame[prev_i]
        for index in range(prev_i, 1094, 3):
        # while True:
            print('\n')
            print (index)
            
            # get frame
            frame_prefx = frames_repo + res[index]
            frame = cv2.imread(frame_prefx)
            img_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            img_gray = np.array( img_gray )

            # get odom
            sec_stamp_frame = stamp_sec_frame[index]
            nsec_stamp_frame = stamp_nsec_frame[index]

            indexes_odom =  [i for i,x in enumerate(stamp_sec_odom) if x==sec_stamp_frame]
            print(indexes_odom)
            best_i = 0
            diff = 10000000000
            for i in indexes_odom:
                if (abs(nsec_stamp_frame-stamp_nsec_odom[i]) <= diff):
                    diff = abs(nsec_stamp_frame-stamp_nsec_odom[i])
                    best_i = i
            print(best_i)
            time_diff = sec_stamp_frame - prev_sec
            # WITH FRAMESKIP
            slam(img_gray, True, np.mean(linear[prev_i:best_i]), np.mean(angular[prev_i:best_i]), time_diff, True)
            #WITHOUT FRAMESKIP
            # slam(img_gray, True, linear[best_i], angular[best_i], time_diff, True)
            prev_i = best_i
            print(sec_stamp_frame-prev_sec)
            print("R: " + str(slam.visual_templates.templates.size/slam.map.experiences.size))
            prev_sec = sec_stamp_frame
            # if index%18 == 0:
                # plotSaveResult_Merge(slam, "oxford_loaded1", 'navy', 'navy', 'navy', 'navy', slam.map.experiences.size)

        # plotSaveResult_Merge(slam, "oxford_loaded2", 'navy', 'navy', 'navy', 'navy', slam.map.experiences.size)
        slam.save("/home/matheus/multisession-ratslam/ratslam/Saves/paper_oxford_partial1_test")  

    
if __name__ == '__main__':

    font = {'family': 'normal',
           'weight': 'bold',
           'size': 25}

    matplotlib.rc('font', **font)

    plot.figure(figsize=(10,8))
    
    time_diff   = 4.0  # oxford
    # time_diff   = 0.05      # irat
    # time_diff   = 1.0       # lacmor
    # time_diff   = 1.0       # circle
    # time_diff   = 1.0       # dir
    # time_diff   = 0.1       # stlucia


    RatSlam_by_load(True, time_diff)
    # load_and_plot()



'''
OXFORD KEY FRAMES -> LAST CHANCE AGAIN!
100 - 2900    ->  LOADED1
2973 - 5700   ->  LOADED2
5702 - 7852   ->  START MAPPING
6000 - 7852   ->  START MAPPING
'''


'''
OXFORD KEY FRAMES -> LAST CHANCE!
100 - 2900    ->  LOADED1
2973 - 5700   ->  LOADED2
5702 - 7852   ->  START MAPPING
6000 - 7852   ->  START MAPPING
'''

'''
OXFORD KEY FRAMES 
100 - 1200    ->  LOADED1
2973 - 5700   ->  LOADED2
1920 - 7852   ->  START MAPPING
1240 - 7852   ->  START MAPPING -> NOT WORKING
'''


'''
STLUCIA KEY FRAMES 
1 - 4435      ->  FIRST  PART UNTIL LOOP CLOSURE
4450 - 11560  ->  SECOND PART UNTIL LOOP CLOSURE
11560 - 12950 ->  REPEATITION OF FIRST PART
12996 - SO ON ->  CONTINUE SLAM TO OTHER PLACES -> OpenRatSLAM youtube time 0:40
'''

'''
IRAT KEY FRAMES
   0 - 2000     ->  MAJOR LOOP
1650 - 2750     ->  INTERNAL CIRCLE 1
2860 - 3450     ->  INTERNAL CIRCLE 2
3590 - 4120     ->  INTERNAL CIRCLE 3
4360 - 5100     ->  INTERNAL CIRCLE 4
'''

'''
IRAT KEY FRAMES
   0 - 1930     ->  MAJOR LOOP
2012 - 2308     ->  INTERNAL CIRCLE 1
2996 - 3177     ->  INTERNAL CIRCLE 2
3680 - 3910     ->  INTERNAL CIRCLE 3
4526 - 4764     ->  INTERNAL CIRCLE 4
'''