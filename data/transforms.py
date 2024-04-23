import matplotlib.pyplot as plt
from configs import config
import os 
import numpy as np
import cv2
from sklearn.cluster import KMeans
from monai.transforms import (
    Compose,
    LoadImaged,
    Rotate90d,
    EnsureChannelFirstd,
    MapTransform,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
    FillHolesd,
    KeepLargestConnectedComponentd,
    MedianSmoothd,
    GaussianSharpend,
    ScaleIntensityRanged,
    )

class HistogramMatchd(MapTransform):
    def __init__(self, keys, template_path, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.template = (plt.imread(template_path)* 255).astype(np.uint8)
        
    def match_histogram(self, input, template):
        input = input[0]
        input_hist, _ = np.histogram(input.flatten(), bins=256, range=[0,255], density=True)
        template_hist, _ = np.histogram(template.flatten(), bins=256, range=[0,255], density=True)
        input_cdf = input_hist.cumsum()
        template_cdf = template_hist.cumsum()
        lut = np.interp(input_cdf, template_cdf, range(256))
        matched_img = cv2.LUT(input.astype(np.uint8), lut.astype(np.uint8))
        matched_img = np.expand_dims(matched_img,0)
        return matched_img
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.match_histogram(d[key], self.template)
        return d
    
    
class CLAHEd(MapTransform):
    def __init__(self, keys, cliplimit=4.0, tilegridsize=(8,8), a_min = 0, a_max=255,  allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.cliplimit = cliplimit
        self.a_min=a_min
        self.a_max=a_max
        self.tilegridsize=tilegridsize
        
    def clahe(self, input):
        input = input[0]
        input[input<=self.a_min]=self.a_min
        input[input>=self.a_max]=self.a_max
        clahe = cv2.createCLAHE(clipLimit=self.cliplimit,tileGridSize=self.tilegridsize)
        input = clahe.apply(input,0)  
        input = np.expand_dims(input,0)
        return input
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.clahe(d[key])
        return d   
    
    
class HistogramEqualizationd(MapTransform):
    def __init__(self, keys, a_min = 0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
    def equalize_hist(self, input):
        input = input[0]  # Assuming single channel image
        input_equalized = cv2.equalizeHist(input)
        input_equalized[input_equalized<self.a_min]=self.a_min
        input_equalized = np.expand_dims(input_equalized, 0)
        return input_equalized
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.equalize_hist(d[key])
        return d

    
class Cannyd(MapTransform):
    def __init__(self, keys, t1=100, t2=200, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.threshold1 = t1
        self.threshold2 = t2
        
    def canny_edge_detection(self, input):
        input = input[0]
        edges = cv2.Canny(input, self.threshold1, self.threshold2)
        edges = np.expand_dims(edges, 0)
        return edges
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.canny_edge_detection(d[key])
        return d


class Sobeld(MapTransform):
    def __init__(self, keys, dx=1, dy=1, ksize=3, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.dx = dx
        self.dy = dy
        self.ksize = ksize
        
    def sobel_edge_detection(self, input):
        input = input[0]
        edges_x = cv2.Sobel(input, cv2.CV_64F, self.dx, 0, ksize=self.ksize)
        edges_y = cv2.Sobel(input, cv2.CV_64F, 0, self.dy, ksize=self.ksize)
        edges = np.sqrt(edges_y ** 2)# + edges_x ** 2)
        edges = np.uint8(edges)
        edges = np.expand_dims(edges, 0)
        return edges
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.sobel_edge_detection(d[key])
        return d
    
class Otsud(MapTransform):
    def __init__(self, keys, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        
    def otsu_thresholding(self, input):
        input = input[0]  # Assuming input is a single channel image
        _, thresh = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = np.expand_dims(thresh, 0)
        return thresh
    
    def __call__(self, data):
        for key in self.key_iterator(data):
            if self.output_key is None:
                self.output_key = key
            data[self.output_key] = self.otsu_thresholding(data[key])
        return data


class IterativeWatershedd(MapTransform):
    def __init__(self, keys, positive_key, negative_key, iterations = 1, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.iterations = iterations
        
    def watershed(self, image, pos_marker, neg_marker, iterations):
        im = image[0].astype(np.uint8)
        markers = pos_marker[0].astype(int) + neg_marker[0].astype(int)
        labels = cv2.watershed(np.stack([im,im,im],-1), markers)
        for _ in range(3): 
            markers=np.zeros_like(markers)
            markers[labels==3]=3
            markers[labels==2]=2
            markers[neg_marker[0]==1]=1
            labels =cv2.watershed(np.stack([im,im,im],-1), markers)
        labels[labels==1]=0
        labels[labels==-1]=0
        labels = np.expand_dims(labels.astype(np.uint8),0)
        return labels
    
    def __call__(self, data):
        for key in self.key_iterator(data):
            if self.output_key is None:
                self.output_key = key
            data[self.output_key] = self.watershed(data[key], data[self.positive_key], data[self.negative_key], self.iterations)
        return data
    
    
class MorphologicalErosiond(MapTransform):
    def __init__(self, keys, kernel_size=(3, 3), iterations=1, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.output_key = output_key
        
    def perform_erosion(self, input):
        input = input[0]
        kernel = np.ones(self.kernel_size, np.uint8)
        erosion = cv2.erode(input, kernel, iterations=self.iterations)
        erosion = np.expand_dims(erosion, 0)
        return erosion
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.perform_erosion(d[key])
        return d

class MorphologicalDilationd(MapTransform):
    def __init__(self, keys, kernel_size=(3, 3), iterations=1, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.output_key = output_key
        
    def perform_dilation(self, input):
        input = input[0]
        kernel = np.ones(self.kernel_size, np.uint8)
        dilation = cv2.dilate(input, kernel, iterations=self.iterations)
        dilation = np.expand_dims(dilation, 0)
        return dilation
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.perform_dilation(d[key])
        return d
    

class PositiveExtractiond(MapTransform):
    def __init__(self, keys, num_clusters=2, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.num_clusters = num_clusters
        self.output_key = output_key
        
    def positive(self, input):
        image = input[0]
        positive_points = np.transpose(np.where(image == 255))
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(positive_points)
        centroids = kmeans.cluster_centers_.astype(int)  # Assuming you want to exclude the first dimension
        positive_marker = np.zeros_like(image)
        positive_marker[centroids[0,0], centroids[0,1]] = 2
        positive_marker[centroids[1,0], centroids[1,1]] = 3
        positive_marker = np.expand_dims(positive_marker, 0)
        return positive_marker
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.positive(d[key])
        return d
    
    
class NegativeExtractiond(MapTransform):
    def __init__(self, keys, num_clusters=2, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.num_clusters = num_clusters
        self.output_key = output_key
        
    def negative(self, input):
        return (255-input)/255
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.negative(d[key])
        return d
    

class MeanShiftd(MapTransform):
    def __init__(self, keys, sr=10, cr=10, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.sr = sr
        self.cr = cr
        self.output_key = output_key
        
    def mean_shift(self, input):
        input = input[0]
        image_color = cv2.cvtColor(input.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        mean_shift = cv2.pyrMeanShiftFiltering(image_color, self.sr, self.cr)[:,:,0]
        mean_shift = np.expand_dims(mean_shift, 0)
        return mean_shift
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.mean_shift(d[key])
        return d
    


class KMeansSegmentd(MapTransform):
    def __init__(self, keys, num_clusters=2, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.num_clusters = num_clusters
        self.output_key = output_key
        
    def segment(self, input):
        input = input[0]
        reshaped_data = input.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(reshaped_data)
        labels = kmeans.predict(reshaped_data)
        larger_class_label = np.argmax(np.bincount(labels))
        labels = np.where(labels == larger_class_label, 1, 0)
        segmented_image = labels.reshape(input.shape)
        segmented_image = np.expand_dims(segmented_image, 0).astype(np.uint8)
        return segmented_image
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.segment(d[key])
        return d
    
    
class DistanceTransformd(MapTransform):
    def __init__(self, keys, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        
    def distance_transform(self, input):
        input = input[0]
        dt = cv2.distanceTransform(input, distanceType=cv2.DIST_L2, maskSize=5)
        dt = 255*(dt-dt.min())/(dt.max()-dt.min())
        dt = np.expand_dims(dt, 0).astype(np.uint8)
        return dt
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.distance_transform(d[key])
        return d



train_transforms = Compose([ # used for training set only, can include data augmentation transforms
    # BASIC TRANSFORMATIONS
    LoadImaged(keys='image'),
    EnsureChannelFirstd(keys='image'),
    Rotate90d(keys='image', k=3),
    ToNumpyd(keys='image', dtype=np.uint8),
    # PREPROCESSING
    HistogramMatchd(keys='image', template_path=os.path.join(config.data['path'],config.transforms['template_path'])),
    CLAHEd(keys='image', cliplimit=config.transforms['cliplimit'], tilegridsize=config.transforms['tilegridsize'], a_min=120, a_max=190),
    # Segmentation
    MeanShiftd(keys='image', sr = 50, cr = 50, output_key= 'mean_shift'),
    Otsud(keys='mean_shift', output_key='image_thresholded'),
    PositiveExtractiond(keys='image_thresholded', output_key = 'positive_marker'),
    MorphologicalDilationd(keys='positive_marker', kernel_size=(26, 91)), 
    NegativeExtractiond(keys='image_thresholded', output_key = 'negative_marker'),
    IterativeWatershedd(keys='image', positive_key='positive_marker', negative_key='negative_marker', iterations=1, output_key='image_segmented'),
])

test_transforms = Compose([ # used for validation and testing sets
    # BASIC TRANSFORMATIONS
    LoadImaged(keys='image'),
    EnsureChannelFirstd(keys='image'),
    Rotate90d(keys='image', k=3),
    ToNumpyd(keys='image', dtype=np.uint8),
    # PREPROCESSING
    HistogramMatchd(keys='image', template_path=os.path.join(config.data['path'],config.transforms['template_path'])),
    CLAHEd(keys='image', cliplimit=config.transforms['cliplimit'], tilegridsize=config.transforms['tilegridsize'], a_min=120, a_max=190),
    # Segmentation
    MeanShiftd(keys='image', sr = 50, cr = 50, output_key= 'mean_shift'),
    Otsud(keys='mean_shift', output_key='image_thresholded'),
    PositiveExtractiond(keys='image_thresholded', output_key = 'positive_marker'),
    MorphologicalDilationd(keys='positive_marker', kernel_size=(26, 91)), 
    NegativeExtractiond(keys='image_thresholded', output_key = 'negative_marker'),
    IterativeWatershedd(keys='image', positive_key='positive_marker', negative_key='negative_marker', iterations=1, output_key='image_segmented'),
])