#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation detection AP metrics for surgical action triplet evaluation.
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
camma, ihu, icube, unistra, france
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import sys
import warnings
from .detection import Detection

import ast

#%%%%%%%%%% Segmentation AND ASSOCIATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Segmentation(Detection):
    def __init__(self, num_class=100, num_tool=6, num_verb=10, num_target=15, 
                 num_instrument_verb=60, num_instrument_target=90, threshold=0.5):
        super().__init__(num_class, num_tool, num_verb, num_target, 
                         num_instrument_verb, num_instrument_target, threshold)      
                
    def iou(self, mask1, mask2):
        """
        Compute IoU for binary masks instead of bounding boxes.
        mask1 and mask2 are numpy arrays of shape (H, W) with binary values.
        """
       
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
    
    def is_match(self, det_gt, det_pd, threshold):
        """
        Modify match function to check IoU on segmentation masks.
        det_gt[-1] and det_pd[-1] should now be binary mask arrays.
        """
        if det_gt[0] == det_pd[0]:  # Class match
            if self.iou(det_gt[-1], det_pd[-1]) >= threshold:  # IoU check for masks
                return True
        return False 

    
    def is_partial_match(self, det_gt, det_pd):  
        if det_gt[0] == det_pd[0]: # cond 1: correct identity        
            if self.iou(det_gt[-1], det_pd[-1]) > 0.0: # cond 2: insufficient iou
                return True
        return False
        
    def is_id_switch(self, det_gt, det_pd, det_gts, threshold):        
        if self.iou(det_gt[-1], det_pd[-1]) > threshold: # cond 2: insufficient/sufficient iou 
            gt_ids = np.array(det_gts[:,0])
            if det_pd[0] in gt_ids: # cond 1: switched identity 
                # return np.where(gt_ids==det_pd[0])[0][0]  I think he was working with like batched inputs or something similar I need to allow for batched inputs. 
                return np.where(gt_ids==det_pd[0])
        return False    
    
    def is_id_miss(self, det_gt, det_pd, threshold):        
        if self.iou(det_gt[-1], det_pd[-1]) > threshold: # cond 2: insufficient/sufficient iou 
                return True
        return False
    
    def list2stack(self, x):
        """
        Convert a list of lists into a NumPy array that supports masks.
        The last element in each list is a mask (NumPy array), which makes `np.stack` fail.
        Instead, we use `np.array` with `dtype=object`.
        """
        if not x:
            return np.array([[]], dtype=object)  # Handle empty input safely
        
        assert isinstance(x[0], list), "Each frame must be a list of lists, each list a prediction"
                
        x = np.array(x, dtype=object)  # Allow storing arrays in list elements
        
        # Sort based on confidence score (index 2), ensuring stable sorting for object dtype
        x = x[np.argsort([-item[2] for item in x])]  
        
        return x

    