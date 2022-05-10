import numpy as np
import cv2


from feature_extraction import FeatureExtraction
from collections import deque

from skimage.transform import resize
from scipy.ndimage.measurements import label

pix_per_cell = 8
cells_per_block =2
orient = 10

class VehicleDetection(object):
    
    def __init__(self, normalisation, classifier, initalise_frame):
        
        self.X_scaler = normalisation
        self.classifier = classifier
        self.image_shape = initalise_frame.shape
        self.previous_detections = deque(maxlen=20)
        
        
    def detect_vehicles(self, image, is_heatmap = False, is_windows= False):
        
        scales = np.array([ 0.6, 0.85, 1.4, 1.8])
        ytop = np.array([360, 400, 420, 450])
        ybottom = np.array([460, 620, 650, 720])
        xleft = np.array([500, 250, 250, 150])
        xright = np.array([1100,1280,1280,1280])
        
        detected_windows = np.empty([0, 4], dtype=np.int64)
        #all_windows = np.empty([0,4], dtype=np.int64)
        
        window_count = []
        
        for scale, y_top, y_bottom, x_left, x_right in zip(scales, ytop, ybottom, xleft, xright):
            scaled_detections, scaled_all_windows = self.search_windows_scale(image, scale, y_top, y_bottom, x_left, x_right, 64)
            
            detected_windows = np.append(detected_windows, scaled_detections, axis=0)
            #all_windows = np.append(all_windows, scaled_all_windows, axis=0)
            
            
        self.previous_detections.append(detected_windows)
         
        #Draw the detected windows on the input image
        img_det_win = self.draw_detections(image, detected_windows)
        #img_all_win = self.draw_detections(image, all_windows)
        
        
        ## Create the heatmap of the classfier detections
        heatmap_empty = np.zeros_like(image)[:,:,0]
        heatmap = self.add_heat(heatmap_empty, np.concatenate(np.array(self.previous_detections)))
        
        #Detect the number of cars in the heatmap
        labels = label(heatmap)
                
        #Draw the bounding boxes onto the image
        final_image = self.draw_labeled_bboxes( image, labels)
        
        if is_heatmap:
            return final_image, heatmap, img_det_win
        elif is_windows:
            return img_det_win
        else:
            return final_image
           
    def search_windows_scale(self, img, scale, y_start, y_stop, x_left, x_right ,window_size):


            detections = np.empty([0, 4], dtype=np.int)
            all_windows = np.empty([0, 4], dtype=np.int)

            #Restrict the image the search area
            image = img[y_start:y_stop, x_left:x_right, :]
            image = image.astype(np.float32)/255
            

            imshape =  image.shape
            img_scaled = cv2.resize(image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

            imshape_scaled = img_scaled.shape
            extraction = FeatureExtraction(img_scaled)

            # Define blocks and steps to transverse
            nxblocks = (imshape_scaled[1] // pix_per_cell) - cells_per_block + 1
            nyblocks = (imshape_scaled[0] // pix_per_cell) - cells_per_block + 1
            nfeat_per_block = orient*cells_per_block**2


            nblocks_per_window = (window_size // pix_per_cell) - cells_per_block + 1
            cells_per_step = 3 #this is used instead of the overlap

            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

            
            for xb in range(nxsteps):
                for yb in range(nysteps):

                    y_pos = yb*cells_per_step
                    x_pos = xb*cells_per_step

                    xleft = x_pos*pix_per_cell
                    ytop = y_pos*pix_per_cell

                    features = extraction.get_features(xleft, ytop).reshape(1,-1)
                    test_features = self.X_scaler.transform(features)
                    test_prediction = self.classifier.predict(test_features)


                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window_size*scale)
                    window = [[xbox_left+x_left, ytop_draw+y_start, xbox_left+win_draw + x_left, ytop_draw+win_draw+y_start]]
                    all_windows = np.append(all_windows , window, axis=0)
                    
                    
                    if test_prediction == 1:

                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window_size*scale)
                        window = [[xbox_left+x_left, ytop_draw+y_start, xbox_left+win_draw + x_left, ytop_draw+win_draw+y_start]]
                        detections = np.append(detections , window, axis=0)
                        #cv2.rectangle(original_image, (xbox_left + x_left, ytop_draw+y_start), (xbox_left+win_draw + x_left, ytop_draw+win_draw+y_start), (0,0,255),6)


            return detections, all_windows
        
    def draw_detections(self, image, windows):
        image_copy = np.copy(image)

        for window in windows:
            cv2.rectangle(image_copy, (window[0], window[1]), (window[2], window[3]), (0,0,255), 6)
            
        return image_copy
    
    def add_heat(self, heatmap, windows, threshold = 10):
        
        for window in windows:
            heatmap[window[1]:window[3], window[0]:window[2]] += 1
            
        
        return self.apply_threshold(heatmap, min(threshold,len(self.previous_detections) ))
    
    def apply_threshold(self, heatmap, threshold):
        
        heatmap[heatmap <= threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)
        return heatmap
    
    def draw_labeled_bboxes(self, img, labels):
        
        image_output = np.copy(img)
        
        for car_number in range(1, labels[1]+1):
            #find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            
            #Identify x and y values of those pixels 
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            #define the bounding box based on min/max x and y
            
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            
            #Draw the bounding box on the image
            cv2.rectangle(image_output, bbox[0], bbox[1], (0,0,255), 6)
            
        return image_output
            
