"""
Demo for Miami, FL 02/2020

Contains the API functions for objet detection on a given image
"""

# python imports
import numpy as np

# torch imports
from torchvision import models
import torch as th

class ObjectDetector():

    def __init__(self):
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def det(self,image):
        dets = self.model(images)
        return dets

    def detection(self,*args):
        return self.det(*args)

    def __str__(self):
        return "Object Detector using torch model: [{}]".format(str(self.model))

def get_bboxes_fasterrcnn(output,filter_cls=None):
    boxes = output['boxes'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()

    # filter only person class
    inds = np.where(labels >= 0)[0]
    if filter_cls:
        inds = np.where(labels == filter_cls)[0]
    #print(inds)
    if len(inds) == 0:
        return []
    # fill remaining values
    bboxes = np.zeros( (len(inds),5) )
    bboxes[:,:4] = boxes[inds]
    bboxes[:,4] = scores[inds]
    return bboxes

def load_frcnn():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)     
    model.eval()
    return model

def det_frcnn(det_model,raw_image):
    image = raw_image / raw_image.max()
    image = image.transpose(2,0,1)
    timage = th.Tensor([image])
    output = det_model(timage)[0]
    bboxes = get_bboxes_fasterrcnn(output,filter_cls=1)
    return bboxes


