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

        xs_slam1.append(exp.x_m + 2)
        ys_slam1.append(exp.y_m + 2)
        
    for exp in slam_p.map.experiences:
        xs_slam2.append(exp.x_m)
        ys_slam2.append(exp.y_m)

    
    plot.scatter(xs_slam1[1:-1], ys_slam1[1:-1], c=c_color, edgecolors=c_color_edge, linewidths=1, marker='.')
    start_l   = plot.scatter(xs_slam1[0], ys_slam1[0], s=220, c=c_color, edgecolors=c_color_edge, linewidths=1, marker='X')
    end_l     = plot.scatter(xs_slam1[-1], ys_slam1[-1], s=200, c=c_color,  edgecolors=c_color_edge, linewidths=1, marker='D')

    plot.scatter(xs_slam2[1:-1], ys_slam2[1:-1], c=c_color2, edgecolors=c_color_edge2, linewidths=1, marker='.')
    start_p   = plot.scatter(xs_slam2[0], ys_slam2[0], s=220, c=c_color2, edgecolors=c_color_edge2, linewidths=1, marker='X')
    end_p     = plot.scatter(xs_slam2[-1], ys_slam2[-1], s=200, c=c_color2,  edgecolors=c_color_edge2, linewidths=1, marker='D')
    
    
    
    legend1 = plot.legend([start_l, end_l], ['Start of L-map','End of L-map'], loc='upper left')
    legend2 = plot.legend([start_p, end_p], ['Start of P-map','End of P-map'], loc='upper right')
    # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
    plot.gca().add_artist(legend1)
    plot.gca().add_artist(legend2)
    
    
    ### complete ####
    plot.xlim(-1.2, 2.25)
    plot.ylim(-0.5, 2.75)

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

        legend1 = plot.legend([start_m1, end_m1], ['Start of L-map','End of L-map'], loc='upper left')
        legend2 = plot.legend([start_m2, end_m2], ['Start of P-map','End of P-map'], loc='upper right')
        # plot.legend([actual_pose], ["Robot's current pose"], loc='lower left', prop={'size': 9})
        plot.gca().add_artist(legend1)
        plot.gca().add_artist(legend2)
    
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

    plot.xlim(-2, 1.1)
    plot.ylim(-2.75, -0.2)
    
    # -----------------------------
    plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.pdf', format='pdf', dpi=400) 
    plot.savefig(r'/home/matheus/merge_ratslammodule/figures/em_'+str(name)+'.png', format='png') 


    plot.pause(0.1)


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
    
    print ('slam1::VISUAL_TEMPLATES_SIZE: ' + str(slam1.visual_templates.templates.size))
    print ('slam2::VISUAL_TEMPLATES_SIZE: ' + str(slam2.visual_templates.templates.size))

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

    print (index_exp1, index_exp2)

    if index_exp1 >= 0 and index_exp2 >= 0:
        
        link1 = slam1.map.links[index_exp1]
        link2 = slam2.map.links[index_exp2-1]

        link_delta_heading_rad  = clip_rad_360(link1.heading_rad - link2.heading_rad)
        link_delta_facing_rad   = clip_rad_360(link1.facing_rad - link2.facing_rad)

        # link_delta_heading_rad  = link1.heading_rad - link2.heading_rad
        # link_delta_facing_rad   = link1.facing_rad - link2.facing_rad

        print (link_delta_facing_rad, link_delta_heading_rad)

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
    slam2       = ratslam.Ratslam()
    
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


        prev_slam       = ratslam.Ratslam()
        pos_merge       = ratslam.Ratslam()

        prev_slam.load('/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_0')

        prev_slam_size = prev_slam.map.experiences.size
        
        find_merge = False

        index = 0

        while True:
            
            index += 1

            frame_prefx  = name_repo_2 +  format(index, '05d') + '-image' + '.png'
            frame = cv2.imread(frame_prefx)

            if frame is None:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = np.array( img )

            if find_merge == False:
                
                slam(img, True, vtrans_2[index-1], vrot_2[index-1], time_diff, True) # with odom file
                
                # if index%20 == 0:
                #     plotSaveResult(prev_slam, slam, "partial_loop4-"+str(index), "navy", "navy", "orange", "red", prev_slam_size)

                isTrue, vt_id = prev_slam.visual_templates.on_image_merge(slam.visual_templates.current_view, slam.visual_templates.current_mean)

                if isTrue == True and slam.current_vt != 0 and vt_id!=0:


                    print ('Encontrou match id: ' + str(vt_id))
                    print ('Actual slam id: ' + str(slam.current_vt))
                    
                    slam2_size = slam.map.experiences.size

                    pos_merge = merge(prev_slam, slam, vt_id, slam.current_vt)

                    find_merge = True

                    plotSaveResult_merge(prev_slam, "merge_loop1-"+str(index), "navy", "navy", "orange", "orange", prev_slam_size)
                    break
                    # pos_merge.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_1")


            # the slam will continue with pos_merge after merge
            else:
                
                # if index%1000 == 0:
                #     plotSaveResult(pos_merge, "irat_merge_part1", 'b')
                #     pos_merge.save("irat_merge_part1")

                pos_merge(img, True, vtrans_2[index-1], vrot_2[index-1], time_diff, True) # with odom file
                # pos_merge(img, False, 0, 0, time_diff, True) # without odom file 

                if index%20 == 0:
                    plotSaveResult_merge(prev_slam, "merge_loop1-"+str(index), "navy", "navy", "orange", "orange", prev_slam_size)
                
        
            print (index)

        # =================================================
        # SLAM AND IMAGE SAVE =============================  
        # pos_merge.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_4")
        # plotSaveResult_merge(prev_slam, "merge_loop4-"+str(index), "navy", "navy", "orange", "red", prev_slam_size)
        # plotResult_2slam(pos_merge, prev_slam_size, slam2_size, "paper_irat_1-0")
        # plotSaveResult(pos_merge, "paper_irat_4-0", 'b')  
        # pos_merge.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/paper_irat_2")  
            
    # =================================================
    # FOR ONLY SLAM WITH LOADED IMAGES AND ODOM =======

    else: #only slam


        # Data format
        name_repo_1   = '/home/matheus/data-ratslam/stlucia/'
        odom_file_1   = name_repo_1 + 'odometry.txt'

        vtrans_1  = []
        vrot_1    = []

        try:
            with open(odom_file_1, 'r') as file:

                print ('ODOM_FILE FOUNDED')

                line = file.readlines()
                for i in range(len(line)):

                    v_value = [x.strip() for x in line[i].split('\t')]
                    v_value = v_value[0].split(' ')
                    # print(v_value)
                    vtrans_1.append(float(v_value[2]))
                    vrot_1.append(float(v_value[3]))
        except:
            print ('NO ODOM_FILE FOUND')

        for index in range(2,10000,18):
        # while True:
            print (index)

            frame_prefx  = name_repo_1 + "{0:0=5d}".format(index) + '-image' + '.png'
            frame = cv2.imread(frame_prefx)
            # if frame is None:
            #     break

            img_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            img_gray = np.array( img_gray )

            slam(img_gray, False, vtrans_1[index-1], vrot_1[index-1], time_diff, True)

            if index%(20)==0:
                plotResult(slam)

            
            # if index%700 == 0:
                
                # plotSaveResult_merge(slam, "irat-complete", 'navy', 'navy', "orange", "red", 0)
                # slam.save("/home/matheus/merge_ratslammodule/MAP_RatSlamModule/Saves/test_stlucia")
        


    
if __name__ == '__main__':

    font = {'family': 'normal',
           'weight': 'bold',
           'size': 25}

    matplotlib.rc('font', **font)

    plot.figure(figsize=(10,8))
    
    # time_diff   = 0.3358    # oxford
    # time_diff   = 0.05      # irat
    # time_diff   = 1.0       # lacmor
    # time_diff   = 1.0       # circle
    # time_diff   = 1.0       # dir
    time_diff   = 0.1       # stlucia


    RatSlam_by_load(False, time_diff)



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