'''

RatSLAM example for mapa merging.

'''
from gettext import find
import cv2
import numpy as np
import itertools
import sys

# Importing ratslam modules 
from MAP_RatSlamModule import ratslam
# from MAP_RatSlamModule import modratslam
from MAP_RatSlamModule import _globals as gb

sys.path.insert(0, '/home/matheus/merge_ratslammodule/MAP_RatSlamModule/')
from utils import *
from merge import *

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
    # plot.plot(xs, ys, color=c_color, ls='-')
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

        xs_slam1.append(exp.x_m + 2)
        ys_slam1.append(exp.y_m + 2)

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
        plot.savefig(r'/home/matheus/merge_ratslammodule/em_'+str(name)+'.pdf', format='pdf', dpi=400)
        plot.savefig(r'/home/matheus/merge_ratslammodule/em_'+str(name)+'.png', format='png') 

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

        #SLAM2

        # plot.plot(xs_slam2, ys_slam2, color=c_color2, ls='-')

        
        plot.scatter(xs_slam2[1:-1], ys_slam2[1:-1], c=c_color2, s=80, edgecolors=c_color_edge2, linewidths=1, marker='.')
        start_m2   = plot.scatter(xs_slam2[0], ys_slam2[0], s=220, c=c_color2, edgecolors=c_color_edge2, linewidths=1, marker='X')
        end_m2     = plot.scatter(xs_slam2[-1], ys_slam2[-1], s=220, c=c_color2,  edgecolors=c_color_edge2, linewidths=1, marker='D')

        # legend1 = plot.legend([start_m1, end_m1], ['Start of L-map','End of L-map'], loc='upper left')
        # legend2 = plot.legend([start_m2, end_m2], ['Start of P-map','End of P-map'], loc='upper right')
        # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
        # plot.gca().add_artist(legend1)
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
        
    
        # plot.legend([start_l, end_l], ['Start of map','End of map'], loc="upper left")
        

    plot.xlabel("x(m)")
    plot.ylabel("y(m)")

    # plot.xlim(-2, 1.1)
    # plot.ylim(-2.75, -0.2)
    
    # -----------------------------
    plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.pdf', format='pdf', dpi=400) 
    plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.png', format='png') 


    plot.pause(0.1)


def load_and_plot():

    slam = ratslam.Ratslam()
    slam.load("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/irat-mod/paper_irat_4/")
    plotSaveResult_merge(slam, 'irat_4', 'navy', 'navy', 'orange', 'red', 0 )

def RatSlam_by_load(mergeSlam , time_diff):
    
    '''
        RatSLAM difinition to recovery frames and odom data already processed 
    '''

    slam        = ratslam.Ratslam()
    
    # =================================================
    # FOR MERGE SLAM WITH LOADED IMAGES AND ODOM ======    
   
    # Data format
    name_repo_1   = '/home/matheus/data-ratslam/irat/all-jpg/'
    odom_file_1   = '/home/matheus/data-ratslam/irat/irat_odom_data.txt'
    
    if mergeSlam:

        loaded1 = ratslam.Ratslam()
        loaded2 = ratslam.Ratslam()
        loaded3 = ratslam.Ratslam()
        loaded4 = ratslam.Ratslam()

        loaded1.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/irat-mod/paper_irat_1')
        loaded2.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/irat-mod/paper_irat_2')
        loaded3.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/irat-mod/paper_irat_3')
        loaded4.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/irat-mod/paper_irat_4')

        vtrans_2  = []
        vrot_2    = []

        try:
            with open(odom_file_1, 'r') as file:

                print ('ODOM_FILE FOUNDED')

                line = file.readlines()
            
                for i in range(len(line)):

                    v_value = [x.strip() for x in line[i].split('\t')]
                    vtrans_2.append(float(v_value[0]))
                    vrot_2.append(float(v_value[1]))
        except:
            print ('NO ODOM_FILE FOUND')
        

        index = 0
        prev_index = index
        find_merge = False
        slam_size = 0
        for i in range(0,4):

            merge_f = ratslam.Ratslam()

            if i == 0:
                merge_f = loaded1
            if i == 1:
                merge_f = loaded2
            if i == 2:
                merge_f = loaded3
            if i == 3:
                merge_f = loaded4

            while True:
                
                index += 1
                frame_prefx  = name_repo_1 +  format(index, '05d') + '-image' + '.jpg'
                frame = cv2.imread(frame_prefx)

                # print (index)
                if frame is None or index == 1600:
                    break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = np.array( img )

                if find_merge == False:
                    
                    slam(img, True, np.mean(vtrans_2[prev_index:index]), np.mean(vrot_2[prev_index:index]), time_diff, True) # with odom file
                    
                    if (index%20==0):
                        print("index " + str(index))
                        plotSaveResult_merge(slam, "irat-loop-"+str(i), "navy", "navy", "orange", "red", slam_size)

                    isTrue, vt_id = merge_f.visual_templates.on_image_merge(slam.visual_templates.current_view, slam.visual_templates.current_mean)
                    
                    if isTrue == True and slam.current_vt != 0 and vt_id > 0:

                        # plotSaveResult(slam, merge_f, 'irat-partial-' + str(i), 'navy', 'navy', 'orange', 'red', True)
                        slam_size = merge_f.map.experiences.size
                        # plotSaveResult_merge(slam, "merge_loop4-"+str(index), "navy", "navy", "orange", "red", slam_size)

                        print ('Encontrou match id: ' + str(vt_id))
                        print ('Actual slam id: ' + str(slam.current_vt))

                        # slam = merge(merge_f, slam, vt_id, slam.current_vt)
                        slam = merge(merge_f, slam, vt_id, slam.current_vt)
                        plotSaveResult_merge(slam, "irat-loop-"+str(i), "navy", "navy", "orange", "red", slam_size)


                        # plotSaveResult(merge_f, slam, 'irat-merged-' + str(i), 'navy', 'navy', 'orange', 'red', True)
                        # plotSaveResult(slam, merge_f, 'irat-partial-' + str(i), 'navy', 'navy', 'orange', 'red', True)
                        # slam_size = slam.map.experiences.size
                        # plotSaveResult_merge(slam, "merge_loop4-"+str(index), "navy", "navy", "orange", "red", slam_size)

                        if i == 3:
                            find_merge = True

                        if i < 3:
                            break
                            # pos_merge.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_1")


                # the slam will continue with pos_merge after merge
                else:
                    
                    
                    # if index%1000 == 0:
                    #     plotSaveResult(pos_merge, "irat_merge_part1", 'b')
                    #     pos_merge.save("irat_merge_part1")
                    slam(img, True, vtrans_2[index-1], vrot_2[index-1], time_diff, True) # with odom file
                    # pos_merge(img, False, 0, 0, time_diff, True) # without odom file 

                    if index%20 == 0:
                        print(index)
                        plotResult(slam)
                    
                    # if index%100==0:
                    #     plotSaveResult(slam, None, 'irat-final', 'navy', 'navy', 'orange', 'red', True)
                
                prev_index = index
                # index += 1

            if frame is None:
                break

            

        # =================================================
        # SLAM AND IMAGE SAVE =============================  
        slam.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/irat-mod/paper_irat_4")
        plotSaveResult_merge(slam, "merge_loop4-"+str(index), "navy", "navy", "orange", "red", slam_size)
        # plotResult_2slam(pos_merge, prev_slam_size, slam2_size, "paper_irat_1-0")
        # plotSaveResult(pos_merge, "paper_irat_4-0", 'b')  
        # pos_merge.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_2")  
            
    # =================================================
    # FOR ONLY SLAM WITH LOADED IMAGES AND ODOM =======
    
    else: #only slam

        vtrans_1  = []
        vrot_1    = []

        try:
            with open(odom_file_1, 'r') as file:

                print ('ODOM_FILE FOUNDED')

                line = file.readlines()
                for i in range(len(line)):

                    v_value = [x.strip() for x in line[i].split('\t')]
                    vtrans_1.append(float(v_value[0]))
                    vrot_1.append(float(v_value[1]))
        except:
            print ('NO ODOM_FILE FOUND')

        index = 0

        while True:
            
            index += 1
            print(index)
            frame_prefx  = name_repo_1 + format(index, '05d') + '-image' + '.jpg'
            
            frame = cv2.imread(frame_prefx)

            if index > 5180:
                slam.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/irat-mod/paper_irat_complete")
                plotSaveResult_merge(slam, "merge_single-"+str(index), "blueviolet", "blueviolet", "blueviolet", "blueviolet", 0)
                break

            img_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            img_gray = np.array( img_gray )

            slam(img_gray, True, vtrans_1[index-1], vrot_1[index-1], time_diff, True)

            # print (index)
            # if index % 20 == 0:
                # plotSaveResult_merge(slam, "merge_single-"+str(index), "blueviolet", "blueviolet", "blueviolet", "blueviolet", 0)
                # for link in slam.map.links:
                #     # print(str(link.exp_to_id))
                #     # print(str(link.exp_from_id))
                #     # print(str(link.d))
                #     # print(str(link.heading_rad))
                #     # print(str(link.facing_rad))
                #     # print(str(link.delta_time_s))
                #     print('\n')
            


        # slam.load("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_complete")
        # plotSaveResult_merge(slam, "test-30-irat-complete", 'blueviolet', 'blueviolet', "orange", "orange", 0)

        # slam.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_complete")
        # plotSaveResult_merge(slam, "irat-complete", 'navy', 'navy', "orange", "red", 0)

        # =================================================
        # SLAM AND IMAGE SAVE =============================   
        # plotSaveResult(slam, "paper_irat_0-0", 'b')
        # slam.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_0")
    # plotSaveResult(slam, None, "irat-single", 'navy', 'navy', 'r', 'r', True)

    
if __name__ == '__main__':

    font = {'family': 'normal',
           'weight': 'bold',
           'size': 25}

    matplotlib.rc('font', **font)

    plot.figure(figsize=(10,8))
    
    # time_diff   = 0.3358    # oxford
    time_diff   = 0.05      # irat
    # time_diff   = 1.0       # lacmor
    # time_diff   = 1.0       # circle
    # time_diff   = 1.0       # dir
    # time_diff   = 0.1       # stlucia


    RatSlam_by_load(True,time_diff)
    # load_and_plot()


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
1650 - 2700     ->  INTERNAL CIRCLE 1
2800 - 3450     ->  INTERNAL CIRCLE 2
3460 - 4120     ->  INTERNAL CIRCLE 3
4150 - 5000    ->  INTERNAL CIRCLE 4
'''

'''
IRAT KEY FRAMES
   0 - 1930     ->  MAJOR LOOP
2012 - 2308     ->  INTERNAL CIRCLE 1
2996 - 3177     ->  INTERNAL CIRCLE 2
3680 - 3910     ->  INTERNAL CIRCLE 3
4526 - 4764     ->  INTERNAL CIRCLE 4
'''