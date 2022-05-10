# Classic Vehicle Detection

## Project Overview

This project builds a vehicle detection pipeline using classic techniques from computer vision and machine learning. Such techniques have fell out of favour in recent times and instead deep learning systems trained end to end have shown to produce better performance in terms of accuracy and computational efficiency of implementation. The point of this project then (aside from nostalgia) is to gain experience with the feature extraction and feature engineering process that underpinned the performance of object detection techniques of old. Doing so also gives an appreciation and intuition of the learnt feature extracting layers found in deep learning architectures.

| File                                | Description                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `Code/feature_extractor.py`      | `CameraCalibration` class used to take an input image and extract the features from a given window region in the image. |
| `Code/vehicle_detection.py`     | `VehicleDetection` class used to apply a SVM classifier across an image through windows of various sizes to detect the presence of all vehicles. |
| `Code/vehicle_detection.ipynb`   | Ipython notebook used to train a linear SVM on a training data set of vehicle and non vehicle images|

## Data Set

The key component to the object detection pipeline is a robust image classifier, capable of detecting if a given image contains a vehicle or not. To do this I train a binary classifier on a data set made up of 17760 rgb images 64x64 pixels in size. In this dataset 8792 are images that contain a vehicle and 8968 are images that contain non-vehicle images taken from a front facing camera on a road vehicle.

![](https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/data_set.jpg?raw=true)



## Feature Extraction

Deciding on the best features to extract from the image for robust classification was done with a combination of trail, experimentation and intuition. The final set of features extracted comprised of a combination of  **HOG (Histogram of Oriented Gradients)**,  **spatial information** and **histograms of colour channels**.  The implementation for the feature extraction pipeline is found in the `FeatureExtraction` class. This class take an input image on initialisation and then extracts the required feature vector for classification from a specified region of the image. 

The class implements the feature extraction in an efficient method by only computing the expensive hog features once upon initialisation. The class abstracts all implementation detail however and the features can be extracted from any location and for any given region size

``` python
extractor = FeatureExtraction(Image)
feature_vector = extractor.get_features(x = 0, y = 0, region_size = 0)
```

The image data is originally represented in RGB colour space. A series of colour space transforms were explored to in order to find a space in which the pixels associated with vehicles lie clustered together separated from pixels associated with the background items in the image. This aids the classifier to find an effective decision boundary to separate the classes in this new space. For this purpose image data was first transformed into YCrCb space.

### Histograms of Orientated Gradients (HOG)

The histogram of oriented gradients technique is a popular feature descriptor that uses the gradient information of the pixels to provide some notion of the shape of the object within the image. The [**scikit-image**](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) package contains an implementation of the HOG technique.

Experiments were performed to find the parameters for the HOG feature extraction technique that maximised the classification accuracy for a common classifier. The following parameters were found to give the best performance.


10 **orientations** : number of orientation bins that the gradient information will be spilt up into the histogram

8 **pixels_per_cell** : Cell size over which each gradient histogram is computed

2 **cells_per_block** : local area over which the histogram counts in a given cell will be normalised

The result of applying the HOG with the above parameters to a singled channelled gray scaled image are shown below.

![](https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/hog_orig.png?raw=true)

![](https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/hog_trans.png?raw=true)


Such a transform results in a returned tensor of size **7x7x2x2x10**. This hog transform is used on each channel of the colour image resulting in a tensor of **3x7x7x2x2x10**. This tensor is the unravelled into a single 1D vector of size **5880** for the HOG feature vector.

### Spatial Information

Spatial information of the image is added to the feature space by taking the original image, resizing it to a smaller resolution (removing the higher frequency information) and the unravelling to a single 1D vector before adding to the feature space. This procedure is performed on each colour channel.

``` python
def bin_spatial(self, img, size=(32,32)):
        colour1 = cv2.resize(img[:,:,0], size).ravel()
        colour2 = cv2.resize(img[:,:,1], size).ravel()
        colour3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((colour1, colour2, colour3))
```

### Colour Histogram

Colour information is added to the feature space by taking histograms of the the each of the colour channels and forming these into a 1D vector.

``` python
def colour_hist(self, img, nbins=32):  
        #compute the histrogram of colour channels separatley
        channel0_hist = np.histogram(img[:,:,0], bins=nbins, range=(0, 256))
        channel1_hist = np.histogram(img[:,:,1], bins=nbins, range=(0, 256))
        channel2_hist = np.histogram(img[:,:,2], bins=nbins, range=(0, 256))
        
        hist_features = np.concatenate((channel0_hist[0], channel1_hist[0], channel2_hist[0]))
        
        return hist_feature
```

## Training a Linear SVM

Using the feature extraction pipeline on the data set above a binary classifier was trained to detect vehicle and non-vehicle images . Due to the dimension of the feature space, a linear support vector machine was chosen due to computational limitations. The training procedure for the SVM is shown in the `vehicle_detection.ipynb` notebook.

Sklearn toolkit was used to train the linear SVM and the input data was pre-processed to have zero mean and unit variance. Cross validation was done to find the optimal hyper-parameter C


## Sliding Window Search Image

The pipeline described so far is capable of extracting features from a **64x64** pixel region of an image, and then using a binary classifier on these features to detect if a vehicle is present in this region or not. To detect all vehicles in the larger image scene the classifier window is slid across the image at various regions and at various scales.

Smaller scales are only required in the distance of the lane, and larger scales in the road closest to the vehicle. The chosen scales and the amount of overlap was chosen to maximise the number of correct classifications and minimise the number of false positives.

The linear support vector machine classifier was trained on features extracted from a **64 x 64** pixel region. Searching windows regions at scales differing from this then will require resizing back to this size.

<img src="https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/scale_smallest.png?raw=true" alt="Girl in a jacket" width="400" height="220"> <img src="https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/scale_mid.png?raw=true" alt="Girl in a jacket" width="400" height="220">


<img src="https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/scale_large.png?raw=true" alt="Girl in a jacket" width="400" height="220"> <img src="https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/scale_largest.png?raw=true" alt="Girl in a jacket" width="400" height="220">

The pipeline for performing a sliding windows search at various scales and locations in an image is implemented by the `VehicleDection` class in `vehicle_detection.py`.  The main bulk of the computation is performed by the following method

``` python
def search_windows_scale(self, img, scale, y_start, y_stop, x_left, x_right ,window_size):

            detections = np.empty([0, 4], dtype=np.int)

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
					# Extract features and preprocess for classification
                    features = extraction.get_features(xleft, ytop).reshape(1,-1)
                    test_features = self.X_scaler.transform(features)
                    test_prediction = self.classifier.predict(test_features)

                    if test_prediction == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window_size*scale)
                        window = [[xbox_left, ytop_draw+y_start, xbox_left+win_draw, ytop_draw+win_draw+y_start]]
                        detections = np.append(detections , window, axis=0)
                        cv2.rectangle(original_image, (xbox_left + x_left, ytop_draw+y_start), (xbox_left+win_draw + x_left, ytop_draw+win_draw+y_start), (0,0,255),6)

            return original_image, detections


```

## Improving classification robustness

The pipeline above currently reports multiple detection at different scales with overlapping windows, and is very sensitive to false positives detection. To make the vehicle detection more robust to these problems I generate a heat map of the intersecting positive classification regions.

``` python

def add_heat(self, heatmap, windows, threshold = 15):
        
        for window in windows:
            heatmap[window[1]:window[3], window[0]:window[2]] += 1
        return heatmap
```
The heat map can then be thresholded to a desired tolerance. This has the desired effect of removing any unwanted false positive windows  and leaves only regions that have been detected by many windows at many scales. These regions indicate a very likely presence of vehicles.

``` python
def apply_threshold(self, heatmap, threshold):
        
        heatmap[heatmap <= threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)
        return heatmap
```

Applying the heatmap and then threshold to the image below leaves two distinct heat signature associated which each of the vehicles in the image.

<img src="https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/origin.png?raw=true" alt="Girl in a jacket" width="400" height="220"> <img src="https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/heatmap.png?raw=true" alt="Girl in a jacket" width="400" height="220">

The bounding rectangle of each of the heat signatures can be computed using the `label()` function from `scipy.ndimage.measurements` to detect individuals groups of detection.
[scipy-label](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html)
![](https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/detected.png?raw=true)


## Video Detection

The vehicle detection pipeline can then be applied to video data. This allows increase robustness to the detection pipeline by integrating the heat map and the thesholding of the heat map over a series of video frames. The implementation detail of this can be found in the `detect_vehicles()` method in the `VehicleDetection` class.

The video below shows the vehicle tracking pipeline on video data 


[![png](https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/video.png?raw=true)](https://youtu.be/z8k6Vp2df3g)



## Results

The vehicle detection pipeline successfully identifies vehicles in the video stream with very few false positives arising. The bounding boxes around the cars are not very stable and in some cases separate around a single vehicle. This effect could be reduced by further tuning of the sliding window search locations and scales to increase robustness.

Using classical feature extraction techniques as done in this project requires much tuning of parameters relating to the feature extraction and sliding window search method. The current implementation is also extremely slow and much optimisation would be required to be able to run this in real time. Individual vehicles become hard to track when close together, suggesting it would become very difficult to accurately locate many cars in heavy traffic conditions. 




<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2MDI2ODU0NjVdfQ==
-->