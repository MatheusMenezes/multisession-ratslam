
# -------------------------------------------------------------------[ header ]

import numpy as np
from ._globals import *

# ------------------------------------------------------------[ VisualTemplate class ]

class VisualTemplate(object):
    def __init__(self, id, data, current_mean):
        self.id   = id
        self.data = data[:]
        self.mean = current_mean

# ------------------------------------------------------------[ LocalViewMatch class ]

class LocalViewMatch(object):
    def __init__(self):
        '''
        templates: 1D numpy array that stores all visual templates created

        current-view: 1D numpy array that stores the current template relative
        to the current frame

        current-mean: current mean of the template current-view

        current-vt: id of the current visual template

        vt-error: current error between the current visual template and the
        closest visual template

        prev-vt: id of the previous visual template

        vt-relative-rad: relative facing angle between closest visual template
        and current visual template

        view-rgb: store the current frame captured by the robot

        greyscale: controls if the image is in grayscale
        '''

        
        self.templates       = np.array([])
        self.current_view    = np.zeros(TEMPLATE_SIZE)
        self.current_mean    = 0
        self.current_vt      = 0
        self.vt_error        = 0
        self.prev_vt         = 0
        self.vt_relative_rad = 0

        self.IMAGE_VT_X_RANGE_MAX = IMAGE_VT_X_RANGE_MAX
        self.IMAGE_VT_X_RANGE_MIN = IMAGE_VT_X_RANGE_MIN

        self.view_rgb  = np.array([])
        self.greyscale = True
        

    def on_image(self, view_rgb, greyscale):
        '''
        Purpose: This routine compare a visual template to all the stored templates,
        returning the matching template and the error between these two

        Algorithm: Convert the current frame to a template. Then, compare the current
        template to all stored templates. Finally, check if error between the closest
        template found is enough for consider this a matching template

        Inputs:
            view_rgb: current frame captured by the robot
            greyscale: True if the current frame is on grayscale, False if the
                current frame is not on grayscale

        Outputs: True, if a mathed visual template was found, otherwise, False
        '''
        if( view_rgb.size == 0 ):
            return False

        # first: store the current frame
        self.view_rgb  = view_rgb[:]
        self.greyscale = greyscale

        # second: convert the current frame to a template and update the prev_vt value

        # Matheus: converts to a template size sum y and x values in blocks and normalize them.
        self.current_view = self.__convert_view_to_view_template__( greyscale )

        self.prev_vt      = self.current_vt

        # third: compare the current template to all stored templates
        vt_match_id = 0
        self.vt_error, vt_match_id = self.__compare__( self.vt_error, vt_match_id )

        # finally:   if error between the closest template found is enough for
        # consider this a matching template
        #print "error: " + str(self.vt_error)
        if ( self.vt_error <= VT_MATCH_THRESHOLD ):
            
            self.__set_current_vt__( vt_match_id )
            return True
        else:
            self.vt_relative_rad = 0
            self.__set_current_vt__( self.__create_template__() )
            return False
    

    # def on_image_merge(self, view_rgb, greyscale):
    def on_image_merge(self, template_view, template_mean):
        '''
        Purpose: This routine compare a visual template to all the stored templates,
        returning the matching template and the error between these two without create
        new or set any previous templates.

        Algorithm: Convert the current frame to a template. Then, compare the current
        template to all stored templates. Finally, check if error between the closest
        template found is enough for consider this a matching template

        Inputs:
            view_rgb: current frame captured by the robot
            greyscale: True if the current frame is on grayscale, False if the
                current frame is not on grayscale

        Outputs: True, if a mathed visual template was found, otherwise, False
        '''
        # if( view_rgb.size == 0 ):
        #     return False

        # first: store the current frame
        # self.view_rgb  = view_rgb[:]
        # self.greyscale = greyscale

        # second: convert the current frame to a template and update the prev_vt value

        # Matheus: converts to a template size sum y and x values in blocks and normalize them.
        # self.current_view = self.__convert_view_to_view_template__( greyscale )
        self.current_view = template_view
        self.current_mean = template_mean

        self.prev_vt      = self.current_vt

        # third: compare the current template to all stored templates
        vt_match_id = -1
        self.vt_error, vt_match_id = self.__compare__( self.vt_error, vt_match_id )

        # finally:   if error between the closest template found is enough for
        # consider this a matching template
        
        if ( self.vt_error <= VT_MATCH_THRESHOLD ):
            return True, vt_match_id
        else:
            return False, vt_match_id


    def __clip_view_x_y__(self,x,y):
        '''
        Purpose: This routine verifies if the value of x and y coordinates are in
        a valid range (between 0 and TEMPLATE-X-SIZE or TEMPLATE-Y-SIZE), and if
        they are not, corrected their value.

        Algorithm: Verifies if the value of x and y coordinates are in a valid
        range (between 0 and TEMPLATE-X-SIZE or TEMPLATE-Y-SIZE), and if they
        are not, correct their values to 0 or TEMPLATE-SIZE, depending which
        one is closer to the current value of x and y.

        Inputs:
            x: x coordinate
            y: y coordinate

        Outputs: x and y updated
        '''
        if x < 0:
            x = 0
        elif x > TEMPLATE_X_SIZE - 1:
            x = TEMPLATE_X_SIZE - 1

        if y < 0:
            y = 0
        elif y > TEMPLATE_Y_SIZE - 1:
            y = TEMPLATE_Y_SIZE - 1

        return x,y


    def __convert_view_to_view_template__(self, grayscale):
        '''
        Purpose: This routine transforms the current frame in a 1D array normalized
        that can be used as a template.

        Algorithm: It first defines the ranges of the current frame (self.view-rgb)
        that are going to be used. Then it sums blocks of pixels of
        size x-block-size * y-block-size. Then, the resulting array is divided
        by x-block-size * y-block-size and by 255, if it is a grayscale frame.
        In case it is used an RGB frame, the algorithm is the same, however,
        the current view is also divided by 3. If VT-NORMALISATION > 0, all
        values will be normalized by this value. Finally, the current mean is
        calculated.

        Inputs:
            greyscale: True if the current frame is on grayscale, False if the
            current frame is not on grayscale.

        Outputs: numpy array of 1D (TEMPLATE-X-SIZE * TEMPLATE-Y-SIZE) normalized
        '''
        sub_range_x = IMAGE_VT_X_RANGE_MAX - IMAGE_VT_X_RANGE_MIN
        sub_range_y = IMAGE_VT_Y_RANGE_MAX - IMAGE_VT_Y_RANGE_MIN

        x_block_size = sub_range_x / TEMPLATE_X_SIZE
        y_block_size = sub_range_y / TEMPLATE_Y_SIZE

        # first: select the part of the frame that is going to be used
        #self.current_view = self.view_rgb[ IMAGE_VT_Y_RANGE_MIN : IMAGE_VT_Y_RANGE_MAX, IMAGE_VT_X_RANGE_MIN : IMAGE_VT_X_RANGE_MAX ]
        self.current_view = np.zeros(TEMPLATE_SIZE)

        if grayscale:

            y_block_count = 0
            x_block_count = 0
            data_next = 0
            for y_block in range(IMAGE_VT_Y_RANGE_MIN, IMAGE_VT_Y_RANGE_MAX, int(y_block_size)): # Change Python 3 - Paulo    
                for x_block in range(IMAGE_VT_X_RANGE_MIN, IMAGE_VT_X_RANGE_MAX, int(x_block_size)): # Change Python 3 - Paulo    
                    for x in range(x_block, x_block + int(x_block_size)): # Change Python 3 - Paulo    
                        for y in range(y_block, y_block + int(y_block_size)):
                            self.current_view[data_next] += self.view_rgb[y][x]
                    self.current_view[data_next] /= 255.0
                    self.current_view[data_next] /= (x_block_size * y_block_size)
                    data_next+=1
                    x_block_count+=1
                y_block_count+=1

            '''
            # second: reshape the vector with y rows  to y_block rows and sum the values in y-axis
            self.current_view = np.sum( self.current_view.reshape( y_block_size, TEMPLATE_Y_SIZE * sub_range_x ),0 )
            # third: reshape the vector with 1 row to x_block_size rows
            self.current_view = self.current_view.reshape( x_block_size, TEMPLATE_X_SIZE * TEMPLATE_Y_SIZE )
            # finally: sum all values in y-axis and divide all by 255 * x_block_size * y_block_size
            self.current_view = np.sum( self.current_view , 0 )/( 255.0 * x_block_size * y_block_size )
            '''
        else:
            self.current_view = self.current_view.reshape( sub_range_y, 3 * sub_range_x )
            self.current_view = np.sum( self.current_view.reshape( y_block_size, 3 * TEMPLATE_Y_SIZE * sub_range_x ), 0 )
            self.current_view = self.current_view.reshape( 3 * x_block_size,TEMPLATE_X_SIZE * TEMPLATE_Y_SIZE )
            self.current_view = np.sum( self.current_view, 0 ) / ( 3.0 * 255.0 * x_block_size * y_block_size )

        # just multiply all values by VT_NORMALISATION, divide all by the mean and
        # keep all values between 0 and 1
        if VT_NORMALISATION > 0:
            avg_value = np.mean( self.current_view )
            self.current_view = self.current_view * VT_NORMALISATION / avg_value
            self.current_view = np.minimum( self.current_view, 1.0 )
            self.current_view = np.maximum( self.current_view, 0.0 )

        # NOT USED
        # now do patch normalisation
        # +- patch size on the pixel, ie 4 will give a 9x9
        if VT_PATCH_NORMALISATION > 0:
            patch_size = VT_PATCH_NORMALISATION
            patch_total = (patch_size * 2 + 1) * (patch_size * 2 + 1)
            current_view_copy = self.current_view[:]

            for x in range(TEMPLATE_X_SIZE):
                for y in range(TEMPLATE_Y_SIZE):
                    patch_sum = 0
                    for patch_x  in range(x - patch_size, x + patch_size + 1):
                        for patch_y in range(y - patch_size, y + patch_size + 1):
                            patch_x_clip, patch_y_clip = self.__clip_view_x_y__(patch_x, patch_y)
                            patch_sum += current_view_copy[patch_x_clip + patch_y_clip * TEMPLATE_X_SIZE]
                    patch_mean = patch_sum / patch_total
                    patch_sum = 0
                    for patch_x  in range(x - patch_size, x + patch_size + 1):
                        for patch_y in range(y - patch_size, y + patch_size + 1):
                            patch_x_clip, patch_y_clip = self.__clip_view_x_y__(patch_x, patch_y)
                            patch_sum += ((current_view_copy[patch_x_clip + patch_y_clip * TEMPLATE_X_SIZE] - patch_mean)* (current_view_copy[patch_x_clip + patch_y_clip * TEMPLATE_X_SIZE] - patch_mean))

                    patch_std = np.sqrt(patch_sum / patch_total)
                    if ( patch_std < VT_MIN_PATCH_NORMALISATION_STD ):
                        self.current_view[x + y * TEMPLATE_X_SIZE] = 0.5
                    else:
                        self.current_view[x + y * TEMPLATE_X_SIZE] = max(0, min(1.0, (((current_view_copy[x + y * TEMPLATE_X_SIZE] - patch_mean) / patch_std) + 3.0)/6.0 ))

        # find the current mean
        self.current_mean = np.mean(self.current_view)

        return self.current_view


    def __set_current_vt__(self, current_vt):
        '''
        Purpose: This routine updates the current visual template (current-vt).

        Algorithm: Verifies if the current-vt is different from the input, if
        it is, updates the prev-vt and current-vt values, if it is not, just
        updates the current-vt.

        Inputs:
            current-vt: id of the current visual template

        Outputs: -
        '''
        if self.current_vt != current_vt:
            self.prev_vt = self.current_vt
        self.current_vt = current_vt

    def __create_template__(self):
        '''
        Purpose: This routine creates a new VisualTemplate ( Localview cell )
        and add this to the collection.

        Algorithm: Create a new VisualTemplate object that stores the current
        frame and the current mean of this frame. Then, this new VisualTemplate
        is added to the collection.

        Inputs: -

        Outputs: Id of the VisualTemplate created
        '''
        newcell = VisualTemplate( self.templates.size, # template id
                                  self.current_view,
                                  self.current_mean )

        # Add newcell to the collection
        self.templates = np.append(self.templates, newcell)

        return self.templates.size -1

    # @cuda.jit('float32,int32(float32, int32)')
    def __compare__(self, vt_err, vt_match_id):
        '''
        Purpose: This routine compares a visual template to all the stored
        templates, returning the closest template and the error between these two.


        Algorithm: For each visual template, tries matching the view at different
        offsets. After finding the smallest offset, calculates the error between
        these two visual templates, the matching id and the relative angle between
        them.

        Inputs:
            vt-err: variable that will store the error between two closest
            visual templates

            vt-match-id: variable that will store the closest template id

        Outputs: the error between the two matching visual templates and the
        id of the closest template
        '''
        if self.templates.size == 0:
            vt_err        = DBL_MAX
            self.vt_error = vt_err
            return vt_err, vt_match_id

        mindiff      = DBL_MAX     # stores the smaller difference
        vt_err       = DBL_MAX     # stores the smaller error
        min_template = 0           # stores the id of the closest template
        epsilon      = 0 #0.005
        min_offset   = 0           # stores the relative facing direction angle between the closest templates

        # for each vt try matching the view at different offsets
        # handles 2d images shifting only in the x direction
        if VT_PANORAMIC:
            for vt in self.templates:
                if( abs( self.current_mean - vt.mean ) > VT_MATCH_THRESHOLD + epsilon ):
                    continue
                for offset in range( 0, TEMPLATE_X_SIZE, VT_STEP_MATCH ):
                    cdiff     = 0
                    columnAux = 0
                    while columnAux < TEMPLATE_SIZE - offset:
                        cdiff     += np.sum( np.abs( self.current_view[ columnAux: columnAux + TEMPLATE_X_SIZE - offset]
                                               - vt.data[ columnAux + offset : columnAux + TEMPLATE_X_SIZE ] ) )
                        cdiff     += np.sum( np.abs( self.current_view[ columnAux + TEMPLATE_X_SIZE - offset : columnAux + TEMPLATE_X_SIZE ]
                                                - vt.data[ columnAux:columnAux + offset ] ) )
                        columnAux += TEMPLATE_X_SIZE
                    if cdiff < mindiff:
                        mindiff      = cdiff
                        min_template = vt.id
                        min_offset   = offset

            # Matheus: get vt_relative_rad by min_offset value
            self.vt_relative_rad = min_offset / TEMPLATE_X_SIZE * 2.0 * np.pi

            if self.vt_relative_rad > np.pi:
                self.vt_relative_rad = self.vt_relative_rad - 2.0 * np.pi

            vt_err        = mindiff/ TEMPLATE_SIZE
            vt_match_id   = min_template
            self.vt_error = vt_err
        else:
            for vt in self.templates:
                if( abs( self.current_mean - vt.mean ) > VT_MATCH_THRESHOLD + epsilon ):
                    continue
                # VT_SHIFT_MATCH is the range (in pixel units) of horizontal offsets over which the current
                # image is compared to all learnt image templates
                for offset in range( 0, VT_SHIFT_MATCH * 2 - 1, VT_STEP_MATCH ):
                    cdiff     = 0
                    columnAux = 0
                    while columnAux < TEMPLATE_SIZE - 2 * VT_SHIFT_MATCH:
                        cdiff     += np.sum( np.abs( self.current_view[ columnAux + VT_SHIFT_MATCH : columnAux + TEMPLATE_X_SIZE - VT_SHIFT_MATCH ]
                                                 - vt.data[ columnAux + offset : columnAux + TEMPLATE_X_SIZE - 2 * VT_SHIFT_MATCH + offset ] ) )
                        columnAux += TEMPLATE_X_SIZE
                    if cdiff < mindiff:
                        mindiff      = cdiff
                        min_template = vt.id
                        min_offset   = 0

            self.vt_relative_rad = min_offset / TEMPLATE_X_SIZE * 2.0 * np.pi
            vt_err               = mindiff / ( TEMPLATE_SIZE - 2 * VT_SHIFT_MATCH * TEMPLATE_Y_SIZE )
            vt_match_id          = min_template
            self.vt_error        = vt_err

        return vt_err, vt_match_id

    

    def save(self, prefix):
        '''
        Purpose: This routine saves all the visual templates stored in the collection.

        Algorithm: Create a file and store first the id, then the mean and the
        data of visual template.

        Inputs: -

        Outputs: -
        '''
        with open( str(prefix) +'/localviewcells.txt', 'w') as file:
            file.writelines(str(TEMPLATE_Y_SIZE*TEMPLATE_X_SIZE)+'\n')
            for i in range(self.templates.size):
                vt = self.templates[i]
                file.writelines(str(vt.id) + '\n')
                file.writelines(str(vt.mean) + '\n')
                np.savetxt(file, vt.data, newline=" ")
                file.writelines('\n')
            file.writelines("-\n")
            file.writelines(str(self.current_vt) + '\n')
            file.writelines(str(self.prev_vt) + '\n')

    def load(self, prefix):
        '''
        Purpose: This routine loads all the visual templates saved and add them
        to the collection.

        Algorithm: Open a file with all the visual template stored and for each
        visual template, create a new VisualTemplate object and store it in the
        collection.

        Inputs: -

        Outputs: -
        '''
        
        with open( str(prefix) +'/localviewcells.txt', 'r') as file:
            tam = int(file.readline())
            line = file.readline()
            while line != "-\n":
                id = float(line)
                line = file.readline()
                mean = float(line)
                line = file.readline()
                datastr = np.array(line.split(" "))
                data = np.array([])
                for it in datastr:
                    if it != '\n':
                        it = float(it)
                        data = np.append(data, np.array(it))
                line = file.readline()
                self.current_view = data
                self.current_mean = mean
                self.__create_template__()
            line = file.readline()
            self.current_vt = float(line)
            line = file.readline()
            self.prev_vt = float(line)
    
    #MAURO -> FUNCAO DE MERGE DOS TEMPLATES
    
    def merge(self, templates, vt_size):
        
        for vt in templates:
            
            # print 'id do novo template: ' + str(vt.id + vt_size)

            newcell = VisualTemplate(vt.id + vt_size, # template id
                                    vt.data,
                                    vt.mean )

            # Add newcell to the collection
            self.templates = np.append(self.templates, newcell)

              
    def __call__(self, img, grayscale):
        self.on_image( img, grayscale )
        return self.current_vt, self.vt_relative_rad

