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
from MAP_RatSlamModule import _globals_circle as gb

sys.path.insert(0, '/home/matheus/merge_ratslammodule/MAP_RatSlamModule/')
from utils import *

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

        for exp in slam.map.experiences:

            if len(xs_slam1) < size_slam1:
                xs_slam1.append(exp.x_m)
                ys_slam1.append(exp.y_m)
            
            else:
                xs_slam2.append(exp.x_m)
                ys_slam2.append(exp.y_m)

        #SLAM1
        
        plot.plot(xs_slam1, ys_slam1, color='navy', ls='-')
        
        start_m1   = plot.scatter(xs_slam1[0], ys_slam1[0], s=220, c='navy', marker='X')
        plot.scatter(xs_slam1[1:-1], ys_slam1[1:-1], c='navy', s=130, marker='o')
        end_m1     = plot.scatter(xs_slam1[-1], ys_slam1[-1], s=220, c='navy', marker='D')

        #SLAM2
        plot.plot(xs_slam2, ys_slam2, color='orange', ls='-')

        start_m2   = plot.scatter(xs_slam2[0], ys_slam2[0], s=220, c='orange', edgecolors='red', linewidths=1, marker='X')
        plot.scatter(xs_slam2[1:-1], ys_slam2[1:-1], c='orange', s=130, edgecolors='red', linewidths=1, marker='o')
        end_m2     = plot.scatter(xs_slam2[-1], ys_slam2[-1], s=220, c='orange',  edgecolors='red', linewidths=1, marker='D')

        # actual_pose = plot.scatter(slam.map.experiences[int(slam.map.current_exp_id)].x_m,
        #             slam.map.experiences[int(slam.map.current_exp_id)].y_m, s=200, color='w', edgecolors='k', marker='^',
        #             linewidths=2)

        plot.plot([xs_slam2[-1], xs_slam1[0] ], [ys_slam2[-1], ys_slam1[0]], color='orange', ls='-')

        if loop == 204 or loop == 400:
            plot.plot([xs_slam1[-1], xs_slam2[0] ], [ys_slam1[-1], ys_slam2[0]], color='navy', ls='-')


        plot.xlabel("X(m)")
        plot.ylabel("Y(m)")
        
        legend1 = plot.legend([start_m1, end_m1], ['Start of L-map','End of L-map'], loc=[0.0, 0.95])
        legend2 = plot.legend([start_m2, end_m2], ['Start of P-map','End of P-map'], loc=[0.7, 0.95])
        # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
        plot.gca().add_artist(legend1)
        plot.gca().add_artist(legend2)

        plot.tight_layout()
        
        # -----------------------------
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.pdf', format='pdf', dpi=400) 
        plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.png', format='png') 
        

        plot.pause(0.01)


def merge(slam1, slam2, vt_id1, vt_id2):
    '''
        Purpose:    The proosal of this routine is to merge two networks by commom paths. 
        
        Algorithm:  The similarity is calculated by Ratslam comparison of local views. 
        
                    The data of slam2 will be passed to slam1 structure.
                    If a local view from slam1 matchs if local views of slam2, the steps are follow:

                        1 - ADDING VISUAL TEMPLATES: The local views of slam2 are adding into slam1

                        2 - ADDING POSECELLS ACTIVATIONS:
                            2.1 - Tranform functions (pcvt_T, pcexp_T) of posecells visual templates (pcvt) and posecells experiences (pcexp)
                            activations of slam2 to coherents activations in slam1 are calculated based on their pcvt and pcexp wich have the same vt_id2 and vt_id1, respectively
                            2.2 - The estructs of pcvt and pcexp of slam2 are adding to slam1:
                                pcvt1 += pcvt2 + pcvt_T; 
                                pcexp1 += pcexp2 + pcexp_T.

                        3 - ADDING EXPERIENCES:
                            3.1 - A tranform function (exp_T) of slam2 experiences nodes is calculated to add this nodes to slam1 experience map (map):
                                map1 += map2 + exp_T

                    After the merge process, the slam1 will have  slam2 visual templates, tranformed activations of posecells and tranformed nodes of experience map.

        Inputs:
            
                    slam1 - A RatSLAM estructure that will recive the second slam2 network
            
                    slam2 - A RatSLAM estructure that will be passed to slam1 network

                    vt_id1, vt_id2 - The matched visual templates ids of slam1 and slam2, respectivelly  

        Outputs:

                    slam1 - The result of the merge will be returned into slam1 network.
    '''
    
    print 'slam1::VISUAL_TEMPLATES_SIZE: ' + str(slam1.visual_templates.templates.size)
    print 'slam2::VISUAL_TEMPLATES_SIZE: ' + str(slam2.visual_templates.templates.size)

    # tamanhos dos arrays de templates e experiences antes das mudancas. Eh usado para calcular a atual
    # experiencia de acordo com as experiencias

    # tamanho inicial do visual_templates.templates
    vt_size     = slam1.visual_templates.templates.size
    # tamanho inicial do network.visual_template
    pcvt_size   = slam1.network.visual_templates.size
    # tamanho inicial do network.experiences
    pcexp_size  = slam1.network.experiences.size
    # tamanho inicial do exp_map de slam1
    map_size    = slam1.map.experiences.size
    # tempo atual do slam
    time_size   = slam1.map.accum_delta_time_s


    # MAURO -> Aqui eu faco o merge dos local view cells (coloco as lvc da rede dois dentro da rede um)

    # ===================================================================
    #  1 - ADDING VISUAL TEMPLATES 

    slam1.visual_templates.merge(slam2.visual_templates.templates[:-1], vt_size)

    # MAURO -> Aqui eu inicio o processo de merge dos posecells. Os posecells tem duas estruturas:
    #          - pcvt -> posecells visual template
    #          - pcexp -> posecell experience
    #          Eu calculo a funcao de transformacao (pcvt_T, pcexp_T) para os dois casos.

    # ===================================================================
    #  2 - ADDING POSECELLS ACTIVATIONS

    # 2.1 - encontrando as funcoes de tranformacao de posecell_visual_template
    for pcvt in slam1.network.visual_templates:
        
        if pcvt.id == vt_id1:        
            pcvt1 = pcvt 
            break
        
        # if pcvt.id == vt_id1 and vt_id1 > 0:        
        #     pcvt1 = pcvt 
        #     break

        # neste caso, o visual template esta nos valores default do programa,
        # entao eh calculado uma estimativa para este valor obtendo-se uma tranformacao
        # dos dois proximos elementos a ele.
        # elif pcvt.id == vt_id1 and vt_id1 == 0:
        #     for pcvt_2 in slam1.network.visual_templates:
        #         if pcvt_2.id == 1:
        #             pcvt1 = pcvt_2
        #             for pcvt_3 in slam1.network.visual_templates:
        #                 if pcvt_3.id == 2:
        #                     pcvt.pc_x = pcvt_2.pc_x - (pcvt_3.pc_x - pcvt_2.pc_x)
        #                     pcvt.pc_y = pcvt_2.pc_y - (pcvt_3.pc_y - pcvt_2.pc_y)
        #                     pcvt.pc_th = pcvt_2.pc_th - (pcvt_3.pc_th - pcvt_2.pc_th)
                            
        #                     pcvt1 = pcvt 
        #                     break
        #             break

        # elif pcvt.id == vt_id1+1: 
        #     pcvt1 = pcvt
        #     break
        
    for pcvt in slam2.network.visual_templates:
        
        if pcvt.id == vt_id2:
            pcvt2 = pcvt 
            break

        # neste caso, o visual template esta nos valores default do programa,
        # entao eh calculado uma estimativa para este valor obtendo-se uma tranformacao
        # dos dois proximos elementos a ele.
        
        # elif pcvt.id == 0:
        #     for pcvt_2 in slam2.network.visual_templates:
        #         if pcvt_2.id == 1:
        #             pcvt2 = pcvt_2
        #             for pcvt_3 in slam2.network.visual_templates:
        #                 if pcvt_3.id == 2:
        #                     pcvt.pc_x = pcvt_2.pc_x - (pcvt_3.pc_x - pcvt_2.pc_x)
        #                     pcvt.pc_y = pcvt_2.pc_y - (pcvt_3.pc_y - pcvt_2.pc_y)
        #                     pcvt.pc_th = pcvt_2.pc_th - (pcvt_3.pc_th - pcvt_2.pc_th) 
        #                     break
        #             break

    # MAURO -> Aqui eu calculo os deltas pra cada coordenada dos pcvt. Se quiseres testar as ativacoes, descomente 
    # e comente as linhas de pcvt_T

    pcvt_delta_x  = pcvt1.pc_x - pcvt2.pc_x
    pcvt_delta_y  = pcvt1.pc_y - pcvt2.pc_y
    pcvt_delta_th = pcvt1.pc_th - pcvt2.pc_th
    
    pcvt_T = [pcvt_delta_x, pcvt_delta_y, pcvt_delta_th]
    # pcvt_T = [0, 0, 0]
    
    # 2.2 - encontrando as funcoes de tranformacao de posecell_experiences

    for pcexp in slam1.network.experiences:
        if pcexp.vt_id == vt_id1:
            pcexp1 = pcexp
            break
    
    for pcexp in slam2.network.experiences:
        if pcexp.vt_id == vt_id2:
            pcexp2 = pcexp
            break
    
    pcexp_delta_x  = pcexp1.x_pc - pcexp2.x_pc
    pcexp_delta_y  = pcexp1.y_pc - pcexp2.y_pc
    pcexp_delta_th = pcexp1.th_pc - pcexp2.th_pc

    # MAURO -> Aqui eu calculo os deltas pra cada coordenada dos pcexp. Se quiseres testar as ativacoes, descomente 
    # e comente as linhas de pcexp_T

    pcexp_T = [pcexp_delta_x, pcexp_delta_y, pcexp_delta_th]
    # pcexp_T = [0, 0, 0]

    # 2.3 - merging into slam1
    slam1.network.merge(slam2.network.visual_templates[:-1], slam2.network.experiences[:-1], vt_id1, vt_id2, pcvt_T, pcexp_T, pcvt_size, vt_size, pcexp_size)

    # ===================================================================
    # 3 - ADDING EXPERIENCES

    index_exp1 = 0
    for exp in slam1.map.experiences:
        index_exp1 += 1
        if exp.vt_id == vt_id1:
            exp1 = exp
            break
    index_exp1 -= 1
    

    index_exp2 = 0
    for exp in slam2.map.experiences:
        index_exp2 += 1
        if exp.vt_id == vt_id2:
            exp2 = exp
            break
    index_exp2 -= 1

    link_T = [0, 0]

    print index_exp1, index_exp2

    if index_exp1 >= 0 and index_exp2 >= 0:
        
        link1 = slam1.map.links[index_exp1]
        link2 = slam2.map.links[index_exp2-1]

        link_delta_heading_rad  = clip_rad_360(link1.heading_rad - link2.heading_rad)
        link_delta_facing_rad   = clip_rad_360(link1.facing_rad - link2.facing_rad)

        # link_delta_heading_rad  = link1.heading_rad - link2.heading_rad
        # link_delta_facing_rad   = link1.facing_rad - link2.facing_rad

        print link_delta_facing_rad, link_delta_heading_rad

        lint_T = [link_delta_heading_rad, link_delta_facing_rad]
    
    #find the x, y and th transformation
    exp_delta_x  = exp1.x_m - exp2.x_m
    exp_delta_y  = exp1.y_m - exp2.y_m
    exp_delta_th = clip_rad_180(exp1.th_rad - exp2.th_rad)
    rot_point = [exp2.x_m, exp2.y_m]

    exp_T = [exp_delta_x, exp_delta_y, exp_delta_th, rot_point, link_T]

    # setting the slam1.map parameters to start exps merge
    slam1.map.accum_delta_facing =  clip_rad_180(exp2.th_rad + exp_delta_th)
    
    id = slam1.map.on_create_experience_merge(slam2.map.experiences[:-1], slam2.map.links[:-1], exp_T, vt_id1, vt_id2, map_size, vt_size, time_size)
    
    slam1.map.on_set_experience( id,  0)

    # slam1.map.iterate()
    
    return slam1


def RatSlam_by_load(mergeSlam , time_diff):
    
    '''
        RatSLAM difinition to recovery frames and odom data already processed 
    '''

    slam        = ratslam.Ratslam()
    
    # =================================================
    # FOR MERGE SLAM WITH LOADED IMAGES AND ODOM ======    
    
    if mergeSlam:

        name_repo_2   = '/home/matheus/data-ratslam/irat/irat-internal-loop-1/'
        odom_file_2   = name_repo_2 + 'irat-odom-loop-1.txt'

    
        vtrans_2  = []
        vrot_2    = []

        try:
            with open(odom_file_2, 'r') as file:

                print ('ODOM_FILE FOUNDED')

                line = file.readlines()
            
                for i in range(len(line)):

                    v_value = [x.strip() for x in line[i].split('\t')]
                    vtrans_2.append(float(v_value[0]))
                    vrot_2.append(float(v_value[1]))
        except:
            print ('NO ODOM_FILE FOUND')

        slam            = ratslam.Ratslam()

        prev_slam       = ratslam.Ratslam()

        pos_merge       = ratslam.Ratslam()

        prev_slam.load('paper_irat_0')

        prev_slam_size = prev_slam.map.experiences.size
        
        find_merge = False

        # a = 0

        # index to repeat the video for loop closure
        index = 1

        while True:
        # for index in range(4530, 5290 ):

            # if index < 10000:
            #     frame_prefx  = name_repo + format(index, '04d') + '-image' + '.png'
            # else:
            frame_prefx  = name_repo_2 +  format(index, '05d') + '-image' + '.png'

            # frame_prefx  = name_repo + 'frame' + format(j_index, '06d') + '.jpg'
            
            frame = cv2.imread(frame_prefx)

            if frame is None:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = np.array( img )

            if find_merge == False:
                
                slam(img, True, vtrans_2[index-1], vrot_2[index-1], time_diff, True) # with odom file
                # slam(img, False, 0, 0, time_diff, True) # without odom file 

                # if index%1000 == 0:
                #     plotSaveResult(slam, "irat_backup")
                #     slam.save("irat_backup")

                if index%50 == 0:
                    plotResult(slam, frame)
                
                # plotResult(slam, frame)
                
                index += 1

                isTrue, vt_id = prev_slam.visual_templates.on_image_merge(slam.visual_templates.current_view, slam.visual_templates.current_mean)

                if isTrue == True and slam.current_vt != 0 and vt_id!=0:

                    # slam.save("irat_backup")
                    # plotSaveResult(slam, "irat_backup")

                    print 'Encontrou match id: ' + str(vt_id)
                    print 'Actual slam id: ' + str(slam.current_vt)

                    # a = input(" insira algum numero pra continuar (a segunda parte sera salva): ")
                    
                    # if a == 1:
                    
                    slam2_size = slam.map.experiences.size

                    pos_merge = merge(prev_slam, slam, vt_id, slam.current_vt)

                    print 'prev_merge exp: ' + str(prev_slam.map.experiences.size)
                    print 'pos_merge exp: ' + str(pos_merge.map.experiences.size)

                    # flag to quit this loop of comparision
                    find_merge = True

                    # plotSaveResult(pos_merge, "paper_irat_0-1", 'b' )
                    plotResult_2slam(pos_merge, prev_slam_size, slam2_size, "paper_irat_0-1" )
                    # pos_merge.save("irat_merge_part2")

                # plotResult(slam, frame)

            # the slam will continue with pos_merge after merge
            else:
                
                # if index%1000 == 0:
                #     plotSaveResult(pos_merge, "irat_merge_part1", 'b')
                #     pos_merge.save("irat_merge_part1")

                pos_merge(img, True, vtrans_2[index-1], vrot_2[index-1], time_diff, True) # with odom file
                # pos_merge(img, False, 0, 0, time_diff, True) # without odom file 

                if index%50 == 0:
                    plotResult_2slam(pos_merge, prev_slam_size, slam2_size, "paper_irat_0-1" )
                
                index += 1
        
            print index

        # =================================================
        # SLAM AND IMAGE SAVE =============================  
        
        plotResult_2slam(pos_merge, prev_slam_size, slam2_size, "paper_irat_1-0")
        # plotSaveResult(pos_merge, "paper_irat_4-0", 'b')  
        pos_merge.save("paper_irat_1")  
            
    # =================================================
    # FOR ONLY SLAM WITH LOADED IMAGES AND ODOM =======

    else: #only slam

        # Data format
        name_repo_1   = '/home/matheus/data-ratslam/irat/irat-major-loop/'
        odom_file_1   = name_repo_1 + 'irat-odom-major.txt'

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

        # slam.load("irat_complete")
        index = 1

        while True:

            # if index < 10000:
            #     frame_prefx  = name_repo + 'frame' + format(index, '04d') + '.jpg'
            # else:
            #     frame_prefx  = name_repo + 'frame' + format(index, '05d') + '.jpg'
            
            frame_prefx  = name_repo_1 + format(index, '05d') + '-image' + '.png'
            
            frame = cv2.imread(frame_prefx)

            if frame is None:
                break

            img_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            img_gray = np.array( img_gray )

            slam(img_gray, True, vtrans_1[index], vrot_1[index], time_diff, True)

            # if index <= 6300 and index >= 5000:
            #     time_diff = 0.065
            #     slam(img_gray, False, 0, 0, time_diff, True)
            #     print time_diff

            # else:
            #     time_diff = 0.1
            #     slam(img_gray, False, 0, 0, time_diff, True)
            #     print time_diff
            
            
            # if index%200 == 0:
            #     plotSaveResult(slam, "irat_complete", 'b')
            #     slam.save("irat_complete_circle_1")

            if index%100 == 0:
                print index
                plotResult(slam, frame)
            index +=1

        # =================================================
        # SLAM AND IMAGE SAVE =============================   
        plotSaveResult(slam, "paper_irat_0-0", 'b')
        slam.save("paper_irat_0-0")
   

def RatSlam_by_video(mergeSlam, time_diff):

    # RatSLAM new object 
    slam = ratslam.Ratslam()

    # Video's settings
    data = r'/home/matheus/merge_ratslammodule/outpy.avi'  # MAURO -> video do circulo

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
            
            print loop 

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

                        # plotSaveResult(slam, 'circle_partial', 'orange', 'red', 'expmap')
                        # plotSaveResult(slam, 'circle_partial', 'orange', 'red', 'posecell')
                        # slam.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_partial')
                        # break

                        pos_merge = merge(prev_slam, slam, vt_id, slam.current_vt)

                        find_merge = True
                        
                        # plotSaveResult_merge(pos_merge, "circle_merged", slam1_size, 'navy', 'navy', 'orange', 'red',  'posecell', loop)
                        plotSaveResult_merge(pos_merge, "circle_merged", slam1_size, 'navy', 'navy', 'orange', 'red',  'expmap', loop)
                        
                # the slam will continue with pos_merge after merge
                else:
                    
                    pos_merge(img, False, 0, 0, time_diff, True)
                    
                    if loop == 204 or loop == 400:
                        plotSaveResult_merge(pos_merge, "circle_merged_loop_" + str(loop), slam1_size, 'navy', 'navy', 'orange', 'red',  'expmap', loop)
                        # pos_merge.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_merged')

                    if loop % 2 == 0:
                        plotResult(pos_merge, img)
        
        # =================================================
        # SLAM SAVE AND PLOT =============================
        # slam.save("prefix")

    # =================================================
    # FOR ONLY SLAM WITH VIDEO AND VISU ODOMET =======
    
    else:

        slam.load("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_merged")

        slam1_size = slam.map.experiences.size
        
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
        plotSaveResult_merge(slam, "circle_merged", slam1_size, 'navy', 'navy', 'orange', 'red',  'expmap', loop)
        
        # =================================================
        # SLAM SAVE AND PLOT =============================
        # slam.save('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/circle_loaded')

    
if __name__ == '__main__':

    font = {'family': 'normal',
           'weight': 'bold',
           'size': 16}

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