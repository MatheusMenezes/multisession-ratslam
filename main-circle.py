'''

RatSLAM example for mapa merging.

'''
import cv2
import numpy as np
import itertools
import sys

# Importing ratslam modules 
from MAP_RatSlamModule import ratslam
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
def plotResult(slam, img):
    # =========================================================
    # PLOT THE CURRENT RESULTS ================================
    # b, g, r = cv2.split(img)
    # rgb_frame = cv2.merge([r, g, b])



    plot.clf()

    # # # RAW IMAGE -------------------

    # ax = plot.subplot(2, 1, 1)
    # plot.title('Image')
    # plot.imshow(img, interpolation='nearest', animated=True, label='blah')
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])

    # =========================================================
    # PLOT THE CURRENT RESULTS FROM LOADED IMAGES =============

    # ax = plot.subplot(1, 2, 1)
    # plot.title('Image')
    # plot.imshow(rgb_frame, interpolation='nearest', animated=True, label='blah')
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])

    # -----------------------------
    # RAW ODOMETRY ----------------

    # plot.subplot(2, 2, 2)
    # plot.title('RAW ODOMETRY')
    # plot.plot(px, py, color='navy', ls=':')
    # plot.plot(slam.odometry[0][-1], slam.odometry[1][-1], 'ko')

    # # -----------------------------
    # # POSECELL --------------
    
   
    # ax = plot.subplot(111, projection='3d')
    # # fig = plot.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # plot.title('Pose cell network activation')
    # xp.append(slam.network.visual_templates[-1].pc_x)
    # yp.append(slam.network.visual_templates[-1].pc_y)
    # thp.append(slam.network.visual_templates[-1].pc_th)

    # ax.scatter(xp, yp, thp,color='b')

    # # ax.scatter(xp_m, yp_m, thp_m, color='r')

    # # ax.scatter(slam.network.best_x, slam.network.best_y, color='r')
    # ax.set_xlim(0, 30)
    # ax.set_ylim(0, 30)
    # ax.set_zlim(0, 35)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')

    # -----------------------------
    # EXPERIENCE MAP --------------

    # ax = plot.subplot(2, 1, 2)

    plot.title('Experience Map')
    xs = []
    ys = []

    for exp in slam.map.experiences:

        xs.append(exp.x_m)
        ys.append(exp.y_m)

    # plot.plot(xs, ys, color='b', ls=':')
    plot.scatter(xs[0], ys[0], color='b', marker = '*')
    plot.scatter(xs[1:], ys[1:], color='b')
    # plot.scatter(xs, ys, color='b', marker='.')

    plot.scatter(slam.map.experiences[int(slam.map.current_exp_id)].x_m,
                 slam.map.experiences[int(slam.map.current_exp_id)].y_m, color='w', alpha=0.5, edgecolors='b', marker='s',
                 linewidths=2)

    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.xlim(-2, 2)
    plot.ylim(-2, 2)
    
    # -----------------------------
    # plot.savefig('/home/matheus/merger_ratslammodule/figures/oxford_part1.eps', format='eps', dpi=100)

    plot.tight_layout()

    plot.pause(0.01)


def plotSaveResult(slam, name, c_color, c_color_edge, type_of_plot):

    plot.clf()
    
    ############################################# POSE CELL ACTIVATION #############################################################

    if type_of_plot == 'posecell':

        plot.figure(figsize=(6,7))

        xp  = []
        yp  = []
        thp = []

        ax = plot.subplot(1, 1, 1, projection='3d')
        ax.set_title('Pose Cells Network Activation')

        # set the aspect of the dimessions to equal in the 3d plot
        # ax.set_aspect("equal")

        for pc in slam.network.visual_templates:
            xp.append(pc.pc_x)
            yp.append(pc.pc_y)
            thp.append(pc.pc_th%35.0)
        
    
        ############# Plot of activations of the ratslam with a center and a edge #####################
        # # the indexes [1:] avoid to plot the first default activation (middle of the network dimesions)
        ax.scatter(xp, yp, thp, color=c_color, s=250, alpha=0.1, edgecolors=c_color_edge, linewidths=3, depthshade=False)
        ax.scatter(xp, yp, thp, color=c_color, alpha=1, marker = 'o', edgecolors=c_color_edge, linewidths=1, depthshade=False, label = "Center of activation")
        ax.text(xp[0]  -0.2 , yp[0]  -0.5, thp[0]  +0.4, "--->  start ", fontsize=12, color='k')
        ax.text(xp[-1] -0.2 , yp[-1] -0.5, thp[-1] +0.4, "---> end", fontsize=12, color='k')
        ###############################################################################################

        ###### For the partial map -  to show the first activation of the PCN of the loaded map#######
        # ax.scatter(xp[:-1], yp[:-1], thp[:-1], color=c_color, s=250, alpha=0.1, edgecolors=c_color_edge, linewidths=3, depthshade=False)
        # center = ax.scatter(xp[:-1], yp[:-1], thp[:-1], color=c_color, alpha=1, marker = 'o', edgecolors=c_color_edge, linewidths=1, depthshade=False)
        # center_lmap = ax.scatter(xp[-1], yp[-1], thp[-1], color='navy', alpha=1, marker = 'o', edgecolors='navy', linewidths=1, depthshade=False)
        # radius_lmap = ax.scatter(xp[-1], yp[-1], thp[-1], color='navy', s=250, alpha=0.1, edgecolors='navy', linewidths=3, depthshade=False)
        # ax.text(xp[0]  -0.2 , yp[0]  -0.5, thp[0]  +0.4, "--->  start ", fontsize=12, color='k')
        # ax.text(xp[-2] -0.2 , yp[-2] -0.5, thp[-2] +0.4, "--->  end of distinct", fontsize=9.5, color='k')
        # ax.text(xp[-2] -0.2 , yp[-2] -0.5, thp[-2] -1.4, "        P-map activations", fontsize=9.5, color='k')
        # # ax.text(xp[-1] -0.2 , yp[-1] -0.5, thp[-1] +0.4, "--> matched act in L-map", fontsize=9.5, color='k')
        ###############################################################################################

        # dimensions of the plot
        ax.set_xlim(38, 42)
        ax.set_ylim(38, 42)
        ax.set_zlim(0, 35)
        ax.set_zticks([0, 5, 15, 25, 35])
        ax.set_xticks([38, 40, 42])
        ax.set_yticks([38, 40, 42])

        # labels and rotations
        ax.xaxis.set_rotate_label(False)
        ax.set_xlabel("$x$'")
        
        ax.yaxis.set_rotate_label(False)
        ax.set_ylabel("$y$'   ")
        
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("$\\theta$'")

        ax.view_init(10, -120)
        
        ############# for complete map ##############
        # ax.legend(loc=[0.4,0.825], fontsize='small')
        #############################################

        ############# for loaded map ################
        ax.legend(loc=[0.4,0.825], fontsize='small')
        #############################################

        ############ for partial map #################
        # ax.legend([center, (center_lmap, radius_lmap)], ["Center of activation", "Matched activation in L-map"], loc=[0.3,0.7], fontsize='small')
        ##############################################

        plot.tight_layout()

        # Save directories 

        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/pcn_'+str(name)+'.pdf', format='pdf', dpi=400)
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/pcn_'+str(name)+'.png', format='png') 


    ################################################### END #########################################################################

    ################################################### EXPERIENCE MAP ##############################################################
    
    
    if type_of_plot == 'expmap':

        plot.figure(figsize=(10,8))

        plot.title('Experience Map')
        
        xs = []
        ys = []
        for exp in slam.map.experiences:

            xs.append(exp.x_m)
            ys.append(exp.y_m)

        
        ##### For complete map - link the last node to the first node of the experience map ##
        # xs.append(xs[0])
        # ys.append(ys[0])
        #####################################################################################

        # link the nodes of experience map from frist to last node
        plot.plot(xs, ys, color=c_color, ls='-')
        
        # start the map with different symbol
        start_m   = plot.scatter(xs[0], ys[0], s=220, c=c_color, edgecolors=c_color_edge, linewidths=1, marker='X')
        
        ###### For complete and loaded map. ####################################################
        # nodes among start and end of the map
        # the index of the scatter is due to the added link xs[0] and ys[0] to close the path
        plot.scatter(xs[1:-1], ys[1:-1], c=c_color, s=130, edgecolors=c_color_edge, linewidths=1, marker='o')
        # end of the experience map with different symbol
        # the index of the end_m is due to the added link xs[0] and ys[0] to close the path
        end_m     = plot.scatter(xs[-1], ys[-1], s=200, c=c_color,  edgecolors=c_color_edge, linewidths=1, marker='D')
        #######################################################################################

        ##### For partial map start node of the loaded map. #####################################
        # plot.scatter(xs[1:-2], ys[1:-2], c=c_color, s=130, edgecolors=c_color_edge, linewidths=1, marker='o')
        # end_m     = plot.scatter(xs[-2], ys[-2], s=200, c=c_color,  edgecolors=c_color_edge, linewidths=1, marker='D')
        # start_l   = plot.scatter(xs[-1], ys[-1], s=200, c='navy',  edgecolors='navy', linewidths=1, marker='X')
        #########################################################################################

        # shows the actual pose of the exp map
        #actual_pose = plot.scatter(slam.map.experiences[int(slam.map.current_exp_id)].x_m,
        #             slam.map.experiences[int(slam.map.current_exp_id)].y_m, s=200, color='w', edgecolors='k', marker='^',
        #             linewidths=2)

        # LEGENDS 
        legend1 = plot.legend([start_m, end_m], ['Start of L-map','End of L-map'], loc='lower right')
        
        # For partial map - plot the legend of the loaded map in the partial map ###############
        # plot.legend([start_l], ["Start of L-map"], loc='upper left')
        ########################################################################################
        
        plot.gca().add_artist(legend1)
        
        ### complete ####
        # plot.xlim(-0.9, 1.1)
        # plot.ylim(-0.1, 1.9)

        ### loaded ####
        plot.xlim(-1.6, 0.5)
        plot.ylim(-0.5, 1.7)

        ### partial ####
        # plot.xlim(-0.3, 1.3)
        # plot.ylim(-0.25, 1.6)

        plot.xlabel("x(m)")
        plot.ylabel("y(m)")

        # -----------------------------

        # plot.tight_layout()

        # Save directories
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.pdf', format='pdf', dpi=400)
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.png', format='png') 

    ################################################### END #########################################################################

    plot.pause(0.1)
    plot.close('all')


def plotSaveResult_merge(slam, name, size_slam1, c_color1, c_color_edge1, c_color2, c_color_edge2, type_of_plot, loop):
    # =========================================================
    # PLOT THE CURRENT RESULTS ================================
    plot.clf()
  
    ############################################# POSE CELL ACTIVATION #############################################################

    if type_of_plot == 'posecell':

        plot.figure(figsize=(6,7))

        xp  = []
        yp  = []
        thp = []

        ax = plot.subplot(1, 1, 1, projection='3d')
        ax.set_title('Pose Cells Network Activation')

        # set the aspect of the dimessions to equal in the 3d plot
        # ax.set_aspect("equal")

        for pc in slam.network.visual_templates:
            xp.append(pc.pc_x)
            yp.append(pc.pc_y)
            thp.append(pc.pc_th)

        
        
        # plot de activations of the ratslam with a center and a edge
        # the indexes [1:] avoid to plot the first default activation (middle of the network dimesions)

        ax.scatter(xp[:size_slam1], yp[:size_slam1], thp[:size_slam1], color=c_color1, s=250, alpha=0.1, edgecolors=c_color_edge1, linewidths=3, depthshade=False)
        ax.scatter(xp[:size_slam1], yp[:size_slam1], thp[:size_slam1], color=c_color1, alpha=1, marker = 'o', edgecolors=c_color_edge1, linewidths=1, depthshade=False, label = "Centers of L-map activation")
        ax.scatter(xp[size_slam1:], yp[size_slam1:], thp[size_slam1:], color=c_color2, s=250, alpha=0.1, edgecolors=c_color_edge2, linewidths=3, depthshade=False)
        ax.scatter(xp[size_slam1:], yp[size_slam1:], thp[size_slam1:], color=c_color2, alpha=1, marker = 'o', edgecolors=c_color_edge2, linewidths=1, depthshade=False, label = "Centers of P-map activation")

        ax.text(xp[0]  -0.2 , yp[0]  -0.5, thp[0]  +0.4, "--->  start of L-map ", fontsize=10, color='k')
        ax.text(xp[size_slam1-1] -0.2 , yp[size_slam1-1] -0.5, thp[size_slam1-1] +0.4, "---> end of L-map", fontsize=10, color='k')
        ax.text(xp[size_slam1]  -0.2 , yp[size_slam1]  -0.5, thp[size_slam1]  +0.4, "--->  start of P-map ", fontsize=10, color='k')
        ax.text(xp[-1] -0.2 , yp[-1] -0.5, thp[-1] +0.4, "---> end of P-map", fontsize=10, color='k')

        # dimensions of the plot
        ax.set_xlim(38, 42)
        ax.set_ylim(38, 42)
        ax.set_zlim(0, 35)
        ax.set_zticks([0, 5, 15, 25, 35])
        ax.set_xticks([38, 40, 42])
        ax.set_yticks([38, 40, 42])

        # labels and rotations
        ax.xaxis.set_rotate_label(False)
        ax.set_xlabel("$x$'")
        
        ax.yaxis.set_rotate_label(False)
        ax.set_ylabel("$y$'   ")
        
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("$\\theta$'")

        ax.view_init(10, -120)
        
        ax.legend(loc=[0.4,0.825], fontsize='x-small')

        plot.tight_layout()

        # Save directories 

        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/pcn_'+str(name)+'.pdf', format='pdf', dpi=400)
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/pcn_'+str(name)+'.png', format='png')

    ############################################# EXPERIENCE MAP #############################################################
    
    
    if type_of_plot == 'expmap':

        plot.figure(figsize=(10,8))

        plot.title('Experience Map')
        
        xs = []
        ys = []

        xs_slam1 = []
        ys_slam1 = []

        xs_slam2 = []
        ys_slam2 = []

        # # for 2 slam
        for exp in slam.map.experiences:

            if len(xs_slam1) < size_slam1:
                xs_slam1.append(exp.x_m)
                ys_slam1.append(exp.y_m)
            
            else:
                xs_slam2.append(exp.x_m)
                ys_slam2.append(exp.y_m)

        # for 1 slam
        # for exp in slam.map.experiences:
            
        #     xs_slam1.append(exp.x_m)
        #     ys_slam1.append(exp.y_m)
            

        #SLAM1    
        plot.plot(xs_slam1, ys_slam1, color='navy', ls='-')
        
        start_m1   = plot.scatter(xs_slam1[0], ys_slam1[0], s=300, c='navy', marker='X')
        plot.scatter(xs_slam1[1:], ys_slam1[1:], c='navy', s=130, marker='o')
        end_m1     = plot.scatter(xs_slam1[-1], ys_slam1[-1], s=300, c='navy', marker='D')

        # actual_pose = plot.scatter(slam.map.experiences[int(slam.map.current_exp_id)].x_m,
        #             slam.map.experiences[int(slam.map.current_exp_id)].y_m, s=200, color='w', edgecolors='k', marker='^',
        #             linewidths=2)

         # #SLAM2
        plot.plot(xs_slam2, ys_slam2, color='orange', ls='-')

        start_m2   = plot.scatter(xs_slam2[0], ys_slam2[0], s=300, c='orange', edgecolors='red', linewidths=1, marker='X')
        plot.scatter(xs_slam2[1:], ys_slam2[1:], c='orange', s=130, edgecolors='red', linewidths=1, marker='o')
        # end_m2     = plot.scatter(xs_slam2[-1], ys_slam2[-1], s=300, c='orange',  edgecolors='orange', linewidths=1, marker='D')

        # end_m2 = plot.scatter(slam.map.experiences[int(slam.map.current_exp_id)].x_m,
        #             slam.map.experiences[int(slam.map.current_exp_id)].y_m, s=300, c='orange',  edgecolors='orange', linewidths=1, marker='D')

        plot.plot([xs_slam2[-1], xs_slam1[0] ], [ys_slam2[-1], ys_slam1[0]], color='orange', ls='-')

        if loop == 208 or loop == 394 or loop == 244:
        # if loop == 394:
            plot.plot([xs_slam1[-1], xs_slam2[0] ], [ys_slam1[-1], ys_slam2[0]], color='navy', ls='-')
        
        end_m2 = plot.scatter(slam.map.experiences[int(slam.map.current_exp_id)].x_m,
                    slam.map.experiences[int(slam.map.current_exp_id)].y_m, s=300, c='orange',  edgecolors='red', linewidths=1, marker='D')


        plot.xlabel("x(m)")
        plot.ylabel("y(m)")
        
        legend1 = plot.legend([start_m1, end_m1], ['Start L-map','End L-map'], loc=[0.0, 0.85])
        legend2 = plot.legend([start_m2, end_m2], ['Start P-map','End P-map'], loc=[0.7, 0.85])
        # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
        plot.gca().add_artist(legend1)
        plot.gca().add_artist(legend2)

        plot.xlim(-1.5, 0.5)
        plot.ylim(-0.25, 2)
        # plot.xticks([-1.25, -0.5, 0.5])
        # plot.yticks([0, 1, 2])

        # plot.tight_layout()
        
        # -----------------------------
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.pdf', format='pdf', dpi=400)
        # plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.eps', format='eps', dpi=600) 
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.png', format='png') 
        

    plot.pause(0.01)
    plot.close('all')


def plotSaveResult_merge2(slam_loaded, slam_partial, c_color1, c_color_edge1, c_color2, c_color_edge2, type_of_plot, loop):
    # =========================================================
    # PLOT THE CURRENT RESULTS ================================
    plot.clf()
    
    ############################################# EXPERIENCE MAP #############################################################
    
    
    if type_of_plot == 'expmap':

        plot.figure(figsize=(10,8))

        plot.title('Experience Map')
        
        xs_slam1 = []
        ys_slam1 = []

        xs_slam2 = []
        ys_slam2 = []

        for exp in slam_loaded.map.experiences:

            xs_slam1.append(exp.x_m)
            ys_slam1.append(exp.y_m)

        for exp in slam_partial.map.experiences:
            xs_slam2.append(exp.x_m - 1.25)
            ys_slam2.append(exp.y_m)
        
        # for i in range(slam_partial.map.experiences.size -1) :
        #     exp = slam_partial.map.experiences[i]
        #     xs_slam2.append(exp.x_m - 1.25)
        #     ys_slam2.append(exp.y_m)

        #SLAM1
        
        plot.plot(xs_slam1, ys_slam1, color='navy', ls='-')
        
        start_m1   = plot.scatter(xs_slam1[0], ys_slam1[0], s=300, c='navy', marker='X')
        plot.scatter(xs_slam1[1:-1], ys_slam1[1:-1], c='navy', s=130, marker='o')
        end_m1     = plot.scatter(xs_slam1[-1], ys_slam1[-1], s=300, c='navy', marker='D')

        #SLAM2
        plot.plot(xs_slam2[:-1], ys_slam2[:-1], color='orange', ls='-')

        start_m2   = plot.scatter(xs_slam2[0], ys_slam2[0], s=300, c='orange', edgecolors='red', linewidths=1, marker='X')
        plot.scatter(xs_slam2[1:-2], ys_slam2[1:-2], c='orange', s=130, edgecolors='red', linewidths=1, marker='o')
        end_m2     = plot.scatter(xs_slam2[-2], ys_slam2[-2], s=300, c='orange',  edgecolors='red', linewidths=1, marker='D')
        # plot.scatter(xs_slam2[-1], ys_slam2[-1], s=300, c='navy',  edgecolors='navy', linewidths=1, marker='X')

        # actual_pose = plot.scatter(slam.map.experiences[int(slam.map.current_exp_id)].x_m,
        #             slam.map.experiences[int(slam.map.current_exp_id)].y_m, s=200, color='w', edgecolors='k', marker='^',
        #             linewidths=2)

        # plot.plot([xs_slam2[-1], xs_slam1[0] ], [ys_slam2[-1], ys_slam1[0]], color='orange', ls='-')

        # if loop == 204 or loop == 400:
        #     plot.plot([xs_slam1[-1], xs_slam2[0] ], [ys_slam1[-1], ys_slam2[0]], color='navy', ls='-')


        plot.xlabel("x(m)")
        plot.ylabel("y(m)")
        
        legend1 = plot.legend([start_m1, end_m1], ['Start L-map','End L-map'], loc=[0.0, 0.85])
        legend2 = plot.legend([start_m2, end_m2], ['Start P-map','End P-map'], loc=[0.7, 0.85])
        # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
        plot.gca().add_artist(legend1)
        plot.gca().add_artist(legend2)

        plot.xlim(-1.5, 0.5)
        plot.ylim(-0.25, 2)
        # plot.xticks([-1.25, -0.5, 0.5])
        # plot.yticks([0, 1, 2])

        # plot.tight_layout()
        
        # -----------------------------
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_circle_before_merge.pdf', format='pdf', dpi=400)
        # plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_circle'+str(name)+'.eps', format='eps', dpi=600) 
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_circle_before_merge.png', format='png') 
        

    plot.pause(0.01)
    plot.close('all')


def RatSlam_by_video(mergeSlam, time_diff):

    # RatSLAM new object 
    slam = ratslam.Ratslam()

    # Video's settings
    data = r'/home/matheus/merge_ratslammodule/videos/outpy.avi'  # MAURO -> video do circulo

    # data = r'/home/matheus/merge_ratslammodule/dir_2voltas_mod.mp4'  # MAURO -> video do lacmor

    video = cv2.VideoCapture(data)
    
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
        
        # prev_slam.load('semidir') # MAURO -> SE QUISER CARREGAR O MAPA DO LACMOR, CARREGUE ESTE LOAD
        prev_slam.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_loaded') # MAURO -> SE QUISER CARREGAR O CIRCULO, CARREGUE ESTE LOAD
        
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

            if loop%2 == 0  and loop >= 92: # MAURO -> SE QUISER RODAR O VIDEO DO CIRCULO, DESCOMENTE ESSE IF E MUDE O ARQUIVO globals_circle PARA globals_

            # if loop%2 == 0  and loop >= 100: # MAURO -> SE QUISER RODAR O VIDEO DO CIRCULO, DESCOMENTE ESSE IF E MUDE O ARQUIVO globals_circle PARA globals_

            # if loop%7 == 0 and loop >= 525: # MAURO -> SE QUISER RODAR O VIDEO DO LACMOR, DESCOMENTE ESSE IF E MUDE O ARQUIVO globals_dir PARA globals_

                img = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
                img = np.array( img )

                if find_merge == False:

                    slam(img, False, 0, 0, time_diff, True)

                    if loop%2==0:
                        plotResult(slam, img)
                   
                    isTrue, vt_id = prev_slam.visual_templates.on_image_merge(slam.visual_templates.current_view, slam.visual_templates.current_mean)

                    if isTrue == True:

                        # plotSaveResult(slam, 'circle_partial', 'orange', 'orange', 'expmap')
                        # plotSaveResult(slam, 'circle_partial', 'orange', 'orange', 'posecell')
                        # slam.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_partial')
                        # break
                        slam1_size = prev_slam.map.experiences.size
                        plotSaveResult_merge2(prev_slam, slam, 'navy', 'navy', 'orange', 'orange', 'expmap', loop)
                       

                        pos_merge = merge(prev_slam, slam, vt_id, slam.current_vt)
                        plotSaveResult_merge(pos_merge, "circle", slam1_size, 'navy', 'navy', 'orange', 'orange', 'posecell', loop)

                        find_merge = True
                        
                        # plotSaveResult_merge(pos_merge, "circle_merged", slam1_size, 'navy', 'navy', 'orange', 'orange',  'posecell', loop)
                        # plotSaveResult_merge(pos_merge, "circle_merged", slam1_size, 'navy', 'navy', 'orange', 'orange',  'expmap', loop)
                        
                # the slam will continue with pos_merge after merge
                else:
                    
                    pos_merge(img, False, 0, 0, time_diff, True)
                    
                    if loop == 208 or loop == 394 or loop == 244 or loop == 138:
                    # if loop == 394:
                        plotSaveResult_merge(pos_merge, "circle_merged_loop_" + str(loop), slam1_size, 'navy', 'navy', 'orange', 'red',  'expmap', loop)
                        plotSaveResult_merge(pos_merge, "circle", slam1_size, 'navy', 'navy', 'orange', 'orange', 'posecell', loop)

                        pos_merge.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_merged')

                    if loop % 2 == 0:
                        plotResult(pos_merge, img)
        
        # =================================================
        # SLAM SAVE AND PLOT =============================
        # slam.save("prefix")

    # =================================================
    # FOR ONLY SLAM WITH VIDEO AND VISU ODOMET =======
    
    else:

        slam.load("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_merged")
        print(slam.map.experiences.size)
        
        slam2 = ratslam.Ratslam()
        slam2.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_complete')

        slam1_size = slam2.map.experiences.size
        print(slam1_size)
        
        # while True :
        
        #     loop += 1
            
        #     flag, frame = video.read()
          
        #     if frame is None:
        #         break

        #     # if loop%2 == 0 and loop >= 92 and loop <= 205: # COMPLETE

        #     # if loop%2 == 0 and loop >= 92 and loop <= 153 : #  PARTIAL

        #     if loop%2 == 0 and loop >= 128 and loop <= 192 : #  LOADED
                
        #         print loop
        #         img = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        #         img = np.array( img )
                
        #         slam(img, False, 0, 0, time_diff, True)

        #         plotResult(slam, frame)
        
        # plotSaveResult_framelimit(slam, "circle_activation_map_loaded", 'navy', 202)
        # plotSaveResult(slam, 'circle_loaded', 'navy', 'navy', 'expmap')
        # plotSaveResult_merge(slam, "teste_circle_complete", slam1_size, 'navy', 'navy', 'orange', 'orange',  'expmap', loop)
        
        # =================================================
        # SLAM SAVE AND PLOT =============================
        # slam.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_loaded')

    
if __name__ == '__main__':

    font = {'family': 'Times New Roman',
           'weight': 'bold',
           'size': 18}

    matplotlib.rc('font', **font)
    
    # time_diff   = 0.3358    # oxford
    # time_diff   = 0.05      # irat
    # time_diff   = 1.0       # lacmor
    time_diff   = 1.0       # circle
    # time_diff   = 1.0       # dir
    # time_diff   = 0.1       # stlucia

    # xp =  []
    # yp =  []
    # thp = []

    # xp_m = []       # used for showing the new merged position of activations
    # yp_m = []
    # thp_m = []

    RatSlam_by_video(True, time_diff)
    # RatSlam_by_load(True,time_diff)



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