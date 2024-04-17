import matplotlib.pyplot as plt
from configs import config
import os 
import numpy as np
import cv2
from monai.transforms import (
    Compose,
    LoadImaged,
    Rotate90d,
    EnsureChannelFirstd,
    MapTransform,
    SqueezeDimd,
    ToNumpyd,
    ToTensord
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
    def __init__(self, keys, cliplimit=4.0, tilegridsize=(8,8), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.cliplimit = cliplimit
        self.tilegridsize=tilegridsize
        
    def clahe(self, input):
        input = input[0]
        clahe = cv2.createCLAHE(clipLimit=self.cliplimit,tileGridSize=self.tilegridsize)
        input = clahe.apply(input,0)   
        input = np.expand_dims(input,0)
        return input
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.clahe(d[key])
        return d   
    
    
class Canny(MapTransform):
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
    
    
train_transforms = Compose([
    LoadImaged(keys='image'),
    EnsureChannelFirstd(keys='image'),
    Rotate90d(keys='image', k=3),
    ToNumpyd(keys='image', dtype=np.uint8),
    HistogramMatchd(keys='image', template_path=os.path.join(config.data['path'],config.transforms['template_path'])),
    CLAHEd(keys='image', cliplimit=config.transforms['cliplimit'], tilegridsize=config.transforms['tilegridsize'])
])
test_transforms = Compose([
    LoadImaged(keys='image'),
    EnsureChannelFirstd(keys='image'),
    Rotate90d(keys='image', k=3),
    ToNumpyd(keys='image', dtype=np.uint8),
    HistogramMatchd(keys='image', template_path=os.path.join(config.data['path'],config.transforms['template_path'])),
    CLAHEd(keys='image', cliplimit=config.transforms['cliplimit'], tilegridsize=config.transforms['tilegridsize']),
    Canny(keys='image', t1=config.transforms['t1'], t2=config.transforms['t2'])
])