'''

RatSLAM example for mapa merging.

'''
import cv2
import numpy as np
import itertools
import sys

# Importing ratslam modules 
from MAP_RatSlamModule import ratslam
from MAP_RatSlamModule import modratslam
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

    plot.title('Experience Map')
    
    xs = []
    ys = []
    
    for exp in slam.map.experiences:

        xs.append(exp.x_m)
        ys.append(exp.y_m)


    plot.plot(xs, ys, color=c_color, ls='-')
    start_m   = plot.scatter(xs[0], ys[0], s=220, c=c_color, edgecolors=c_color_edge, linewidths=1, marker='X')
    end_m     = plot.scatter(xs[-1], ys[-1], s=200, c=c_color,  edgecolors=c_color_edge, linewidths=1, marker='D')

    legend1 = plot.legend([start_m, end_m], ['Start of L-map','End of L-map'], loc='upper right')
    
    plot.gca().add_artist(legend1)

    plot.xlabel("x(m)")
    plot.ylabel("y(m)")

    plot.tight_layout()

    plot.pause(0.01)


def plotSaveResult(slam_l, slam_p, name, c_color, c_color_edge, c_color2, c_color_edge2, size_slam1):

    plot.clf()

    plot.title('Experience Map')
    
    xs_slam1 = []
    ys_slam1 = []
    
    xs_slam2 = []
    ys_slam2 = []
    

    for exp in slam_l.map.experiences:

        xs_slam1.append(exp.x_m + 5)
        ys_slam1.append(exp.y_m + 6)
        
    for exp in slam_p.map.experiences:
        xs_slam2.append(exp.x_m)
        ys_slam2.append(exp.y_m)

    
    plot.scatter(xs_slam1[1:-1], ys_slam1[1:-1], c=c_color, edgecolors=c_color_edge, linewidths=1, marker='.')
    start_l   = plot.scatter(xs_slam1[0], ys_slam1[0], s=220, c=c_color, edgecolors=c_color_edge, linewidths=1, marker='X')
    end_l     = plot.scatter(xs_slam1[-1], ys_slam1[-1], s=200, c=c_color,  edgecolors=c_color_edge, linewidths=1, marker='D')

    plot.scatter(xs_slam2[1:-1], ys_slam2[1:-1], c=c_color2, edgecolors=c_color_edge2, linewidths=1, marker='.')
    start_p   = plot.scatter(xs_slam2[0], ys_slam2[0], s=220, c=c_color2, edgecolors=c_color_edge2, linewidths=1, marker='X')
    end_p     = plot.scatter(xs_slam2[-1], ys_slam2[-1], s=200, c=c_color2,  edgecolors=c_color_edge2, linewidths=1, marker='D')
    
    
    
    legend1 = plot.legend([start_l, end_l], ['Start of L-map','End of L-map'], loc=[0.0, 0.95])
    legend2 = plot.legend([start_p, end_p], ['Start of P-map','End of P-map'], loc=[0.7, 0.95])
    # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
    plot.gca().add_artist(legend1)
    plot.gca().add_artist(legend2)
    
    
    ### complete ####
    plot.xlim(-6.5, 4.5)
    plot.ylim(-2, 10)

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
    plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.pdf', format='pdf', dpi=400)
    plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.png', format='png') 

    ################################################### END #########################################################################

    # plot.tight_layout()
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

        legend1 = plot.legend([start_m1, end_m1], ['Start of L-map','End of L-map'], loc=[0.0, 0.95])
        legend2 = plot.legend([start_m2, end_m2], ['Start of P-map','End of P-map'], loc=[0.7, 0.95])
        # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
        plot.gca().add_artist(legend1)
        plot.gca().add_artist(legend2)
    
    else:
        

        # for exp in slam.map.experiences:

        #     xs_slam1.append(exp.x_m)
        #     ys_slam1.append(exp.y_m)

        for exp in slam.map.experiences:
            
            # rot_x, rot_y = rotate([0, 0], [exp.x_m, exp.y_m], 0.5)

            # xs_slam1.append(rot_x)
            # ys_slam1.append(rot_y)

            xs_slam1.append(exp.x_m)
            ys_slam1.append(exp.y_m)

            
            
        
        start_l   = plot.scatter(xs_slam1[0], ys_slam1[0], s=220, c=c_color1, edgecolors=c_color_edge1, linewidths=1, marker='X')
        plot.scatter(xs_slam1[1:-1], ys_slam1[1:-1], c=c_color1, edgecolors=c_color_edge1, linewidths=1, marker='.')
        end_l     = plot.scatter(xs_slam1[-1], ys_slam1[-1], s=200, c=c_color1,  edgecolors=c_color_edge1, linewidths=1, marker='D')
        
    
        plot.legend([start_l, end_l], ['Start of map','End of map'], loc=[0.0, 0.95])
        

    plot.xlabel("x(m)")
    plot.ylabel("y(m)")

    # plot.xlim(-2, 1.1)
    # plot.ylim(-2.75, -0.2)
    
    # -----------------------------
    plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.pdf', format='pdf', dpi=400) 
    plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.png', format='png') 


    plot.pause(0.1)

def RatSlam_by_video(mergeSlam, time_diff):

    # RatSLAM new object 
    slam = ratslam.Ratslam()
    pos_merge = ratslam.Ratslam()

    # Video's settings
    # data = r'/home/matheus/merge_ratslammodule/outpy.avi'  # MAURO -> video do circulo

    data = r'/home/matheus/merge_ratslammodule-old_commit/dir_2voltas_mod.mp4'  # MAURO -> video do lacmor

    video = cv2.VideoCapture('/home/matheus/merge_ratslammodule-old_commit/videos/dir_2voltas_mod.mp4')
    
    width  = video.get( cv2.CAP_PROP_FRAME_WIDTH  )
    height = video.get( cv2.CAP_PROP_FRAME_HEIGHT )

    # Changing values in _globals variables as example. The globals file
    # set variables from pose cells, local view cells, visual odometry...  
    gb.IMAGE_WIDTH = width
    gb.IMAGE_HEIGHT = height

    loop = 0
    
    # =================================================
    # FOR MERGE SLAM WITH VIDEO AND VISU ODOMET =======
    
    if mergeSlam:

        prev_slam = ratslam.Ratslam()
        pos_merge = ratslam.Ratslam()
        
        prev_slam.load('/home/matheus/merge_ratslammodule-old_commit/MAP_RatSlamModule/Saves/lacmor_loaded') # MAURO -> SE QUISER CARREGAR O MAPA DO LACMOR, CARREGUE ESTE LOAD
        # prev_slam.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_loaded') # MAURO -> SE QUISER CARREGAR O CIRCULO, CARREGUE ESTE LOAD
        
        slam1_size = prev_slam.map.experiences.size
        find_merge = False

        # plotSaveResult(prev_slam, 'lacmor_part1_saved', 'r')

        # slam + merge if found
        while True :
            
            print (loop) 

            loop += 1
            
            flag, frame = video.read()

            if frame is None:
                break

            # if loop%2 == 0  and loop >= 92: # MAURO -> SE QUISER RODAR O VIDEO DO CIRCULO, DESCOMENTE ESSE IF E MUDE O ARQUIVO globals_circle PARA globals_

            # if loop%2 == 0  and loop >= 100: # MAURO -> SE QUISER RODAR O VIDEO DO CIRCULO, DESCOMENTE ESSE IF E MUDE O ARQUIVO globals_circle PARA globals_

            if loop%7 == 0 and loop >= 525: # MAURO -> SE QUISER RODAR O VIDEO DO LACMOR, DESCOMENTE ESSE IF E MUDE O ARQUIVO globals_dir PARA globals_

                img = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
                img = np.array( img )

                if find_merge == False:

                    slam(img, False, 0, 0, time_diff, True)

                    if loop%20==0:
                        plotSaveResult(prev_slam, slam, 'lacmor_partial'+str(loop),  'navy', 'navy', 'orange', 'red', slam1_size)
                   
                    isTrue, vt_id = prev_slam.visual_templates.on_image_merge(slam.visual_templates.current_view, slam.visual_templates.current_mean)

                    if isTrue == True:

                        plotSaveResult(prev_slam, slam, 'lacmor_partial'+str(loop),  'navy', 'navy', 'orange', 'red', slam1_size)
                        # slam.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/lacmor_partial')
                        # break

                        pos_merge = merge(prev_slam, slam, vt_id, slam.current_vt)

                        find_merge = True
                        
                        # plotSaveResult_merge(pos_merge, "circle_merged", slam1_size, 'navy', 'navy', 'orange', 'red',  'posecell', loop)
                        plotSaveResult_merge(pos_merge, 'lacmor_merge'+str(loop), 'navy', 'navy', 'orange', 'red', slam1_size)
                        
                        
                # the slam will continue with pos_merge after merge
                else:
                    
                    pos_merge(img, False, 0, 0, time_diff, True)

                    if loop % 20 == 0:
                        plotSaveResult_merge(pos_merge, 'lacmor_merge'+str(loop), 'navy', 'navy', 'orange', 'red', slam1_size)
        
        # =================================================
        # SLAM SAVE AND PLOT =============================
        # slam.save("prefix")
        # plotSaveResult_merge(pos_merge, 'lacmor_merge'+str(loop), 'navy', 'navy', 'orange', 'red', slam1_size)
        # pos_merge.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/lacmor_merged')

    # =================================================
    # FOR ONLY SLAM WITH VIDEO AND VISU ODOMET =======
    
    else:

        # slam.load("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/lacmor_loaded")
        
        slam2 = ratslam.Ratslam()

        slam.load("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/lacmor_complete")
        print(slam.map.experiences.size)
        
        slam2 = ratslam.Ratslam()
        slam2.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/lacmor_merged')

        slam1_size = slam2.map.experiences.size
        print(slam1_size)

        # slam1_size = slam.map.experiences.size
        
        # while True :
        
        #     loop += 1
            
        #     flag, frame = video.read()
          
        #     if frame is None:
        #         break

        #     # if loop%2 == 0 and loop >= 92 and loop <= 205: # COMPLETE

        #     # if loop%2 == 0 and loop >= 92 and loop <= 153 : #  PARTIAL

        #     if loop%7 == 0: # and loop >= 128 and loop <= 192 : #  LOADED
                
        #         print loop
        #         img = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        #         img = np.array( img )
                
        #         slam(img, False, 0, 0, time_diff, True)

        #         if loop%20==0:
        #             # plotSaveResult_merge(slam, "lacmor_complete", 'blueviolet', 'blueviolet', "orange", "red", 0)
        #             plotSaveResult_merge(slam, "lacmor_loaded", 'navy', 'navy', "orange", "red", 0)
        #             # plotSaveResult(slam, 'lacmor_loaded', 'navy', 'navy', 'expmap')
        
        # plotSaveResult_framelimit(slam, "circle_activation_map_loaded", 'navy', 202)
        # plotSaveResult(slam, 'circle_loaded', 'navy', 'navy', 'expmap')
        # plotSaveResult_merge(slam, "circle_merged", slam1_size, 'navy', 'navy', 'orange', 'red',  'expmap', loop)
        
        # =================================================
        # SLAM SAVE AND PLOT =============================
        # slam.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_loaded')
        # plotSaveResult_merge(slam, "lacmor_loaded", 'navy', 'navy', "orange", "red", 0)
        # slam.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/lacmor_complete")
        
   

    
if __name__ == '__main__':

    font = {'family': 'normal',
           'weight': 'bold',
           'size': 16}

    matplotlib.rc('font', **font)

    plot.figure(figsize=(10,8))
    
    # time_diff   = 0.3358    # oxford
    # time_diff   = 0.05      # irat
    time_diff   = 1.0       # lacmor
    # time_diff   = 1.0       # circle
    # time_diff   = 1.0       # dir
    # time_diff   = 0.1       # stlucia


    RatSlam_by_video(True,time_diff)



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