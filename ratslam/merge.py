# from ratslam import modratslam
from ratslam import _globals as gb   
   
   
def merge(self, loadedSlam, partialSlam, vtLoadedMatch, vtPatialMatch):
    '''
        Purpose:    The proosal of this routine is to merge two networks by commom paths. 
        
        Algorithm:  The similarity is calculated by Ratslam comparison of local views. 
        
                    The data of partialSlam will be passed to loadedSlam structure.
                    If a local view from loadedSlam matchs if local views of partialSlam, the steps are follow:

                        1 - ADDING VISUAL TEMPLATES: The local views of partialSlam are adding into loadedSlam

                        2 - ADDING POSECELLS ACTIVATIONS:
                            2.1 - Tranform functions (pcvt_T, pcexp_T) of posecells visual templates (pcvt) and posecells experiences (pcexp)
                            activations of partialSlam to coherents activations in loadedSlam are calculated based on their pcvt and pcexp wich have the same vtPatialMatch and vtLoadedMatch, respectively
                            2.2 - The estructs of pcvt and pcexp of partialSlam are adding to loadedSlam:
                                pcvt1 += pcvt2 + pcvt_T; 
                                pcexp1 += pcexp2 + pcexp_T.

                        3 - ADDING EXPERIENCES:
                            3.1 - A tranform function (exp_T) of partialSlam experiences nodes is calculated to add this nodes to loadedSlam experience map (map):
                                map1 += map2 + exp_T

                    After the merge process, the loadedSlam will have  partialSlam visual templates, tranformed activations of posecells and tranformed nodes of experience map.

        Inputs:
            
                    loadedSlam - A RatSLAM estructure that will recive the second partialSlam network
            
                    partialSlam - A RatSLAM estructure that will be passed to loadedSlam network

                    vtLoadedMatch, vtPatialMatch - The matched visual templates ids of loadedSlam and partialSlam, respectivelly  

        Outputs:

                    loadedSlam - The result of the merge will be returned into loadedSlam network.
    '''
    
    print ('loadedSlam::VISUAL_TEMPLATES_SIZE: ' + str(loadedSlam.visual_templates.templates.size))
    print ('partialSlam::VISUAL_TEMPLATES_SIZE: ' + str(partialSlam.visual_templates.templates.size))

    print ('loadedSlam::VISUAL_EXPERIENCES_SIZE: ' + str(loadedSlam.map.experiences.size))
    print ('partialSlam::VISUAL_EXPERIENCES_SIZE: ' + str(partialSlam.map.experiences.size))

    print ('loadedSlam::VISUAL_LINKS_SIZE: ' + str(loadedSlam.map.links.size))
    print ('partialSlam::VISUAL_LINKS_SIZE: ' + str(partialSlam.map.links.size))

    
    vt_size     = loadedSlam.visual_templates.templates.size
    # tamanho inicial do network.visual_template
    pcvt_size   = loadedSlam.network.visual_templates.size
    # tamanho inicial do network.experiences
    pcexp_size  = loadedSlam.network.experiences.size
    # tamanho inicial do exp_map de loadedSlam
    map_size    = loadedSlam.map.experiences.size
    # tamanho do link
    link_size   = loadedSlam.map.links.size
    # tempo atual do slam
    time_size   = partialSlam.map.accum_delta_time_s
    # deltas do mapa

    # ===================================================================
    #  1 - ADDING VISUAL TEMPLATES 

    loadedSlam.visual_templates.merge(partialSlam.visual_templates.templates[:-1], vt_size)

    # ===================================================================
    #  2 - ADDING POSECELLS ACTIVATIONS

    # 2.1 - encontrando as funcoes de tranformacao de posecell_visual_template
    for pcvt in loadedSlam.network.visual_templates:
        
        if pcvt.id == vtLoadedMatch:        
            pcvt1 = pcvt 
            break
        
    for pcvt in partialSlam.network.visual_templates:
        
        if pcvt.id == vtPatialMatch:
            pcvt2 = pcvt 
            break


    pcvt_delta_x  = pcvt1.pc_x - pcvt2.pc_x
    pcvt_delta_y  = pcvt1.pc_y - pcvt2.pc_y
    pcvt_delta_th = pcvt1.pc_th - pcvt2.pc_th
    
    pcvt_T = [pcvt_delta_x, pcvt_delta_y, pcvt_delta_th]
    
    # 2.2 - encontrando as funcoes de tranformacao de posecell_experiences

    for pcexp in loadedSlam.network.experiences:
        if pcexp.vt_id == vtLoadedMatch:
            pcexp1 = pcexp
            break
    
    for pcexp in partialSlam.network.experiences:
        if pcexp.vt_id == vtPatialMatch:
            pcexp2 = pcexp
            break
    
    pcexp_delta_x  = pcexp1.x_pc - pcexp2.x_pc
    pcexp_delta_y  = pcexp1.y_pc - pcexp2.y_pc
    pcexp_delta_th = pcexp1.th_pc - pcexp2.th_pc

    pcexp_T = [pcexp_delta_x, pcexp_delta_y, pcexp_delta_th]

    # 2.3 - merging into loadedSlam
    loadedSlam.network.merge(partialSlam.network.visual_templates[:-1], partialSlam.network.experiences[:-1], vtLoadedMatch, vtPatialMatch, pcvt_T, pcexp_T, pcvt_size, vt_size, pcexp_size)

    # ===================================================================
    # 3 - ADDING EXPERIENCES

    index_exp1 = 0
    for exp in loadedSlam.map.experiences:
        index_exp1 += 1
        if exp.vt_id == vtLoadedMatch:
            exp1 = exp
            break
    index_exp1 -= 1
    
    index_exp2 = 0
    for exp in partialSlam.map.experiences:
        index_exp2 += 1
        if exp.vt_id == vtPatialMatch:
            exp2 = exp
            break
    index_exp2 -= 1

    link_T = [0, 0]

    print (index_exp1, index_exp2)
    
    #find the x, y and th transformation
    exp_delta_x  = exp1.x_m - exp2.x_m
    exp_delta_y  = exp1.y_m - exp2.y_m
    vt_relative_rad = loadedSlam.visual_templates.vt_relative_rad
    exp_delta_th = clip_rad_180((exp1.th_rad - exp2.th_rad) + vt_relative_rad)
    
    rot_point = [exp2.x_m, exp2.y_m]
    print("delta_th from exp: " + str(exp_delta_th))

    for l1 in loadedSlam.map.links:
        if l1.exp_to_id == exp1.id:
            break
    
    for l2 in partialSlam.map.links:
        if l2.exp_to_id == exp2.id:
            break

    link_delta_heading_rad  = clip_rad_180(l1.heading_rad - l1.heading_rad)
    link_delta_facing_rad   = clip_rad_180(l2.facing_rad - l2.facing_rad)

    link_T = [link_delta_heading_rad, link_delta_facing_rad]
    exp_T = [exp_delta_x, exp_delta_y, exp_delta_th, rot_point, link_T]

    # setting the loadedSlam.map parameters to start exps merge
    loadedSlam.map.accum_delta_facing =  clip_rad_180(exp2.th_rad + exp_delta_th)
    
    id = loadedSlam.map.on_create_experience_merge(partialSlam.map.experiences, partialSlam.map.links, exp_T, vtLoadedMatch, vtPatialMatch, map_size, link_size, vt_size)
    loadedSlam.map.on_set_experience( id,  0)
    
    return loadedSlam