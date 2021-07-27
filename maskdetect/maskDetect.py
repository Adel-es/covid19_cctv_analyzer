#from configs.config_handler import Config
import cv2 as cv
import numpy as np
import maskdetect.faceDetection.faceDetector as face
import maskdetect.maskDetection.maskDetector as mask 

import sys, os
root_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_path)
from configs import runInfo 
from utils.types import MaskToken, TrackToken


def cropHuman(trackingReslt, frameId, raw_img) : 
    if len(trackingReslt[frameId]) == 0 : 
        return 
    else : 
        for bbox, tid in trackingReslt[frameId] : 
            croppedHuman = raw_img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            imageName = "{}_{}.jpg".format(frameId, tid); 
            print(imageName)
            cv.imwrite("images/" + imageName, croppedHuman)
    
        
def runMaskDetect(trackingResult, reidResult, distanceResult, MaskResult) : 
    InputVideo = runInfo.input_video_path
    # outputVideo = runInfo.output_video_path
    startFrame = runInfo.start_frame 
    endFrame = runInfo.end_frame

    outputWriter = None
    faceDetector = face.Detector()
    maskDetector = mask.Classifier()
    origWidth = 0
    origHeight = 0

    inputCapture = cv.VideoCapture(InputVideo)
    frameId = -1; 
    
    print(" == [ MaskDetect ] : run Mask Detection == ")
    while(inputCapture.isOpened()) : 
        
        frameId = frameId + 1
        frameResult = [] 
        ret, raw_img = inputCapture.read() 
        
        '''        
        if outputWriter is None :         
            origWidth = raw_img.shape[1]
            origHeight = raw_img.shape[0]
            outputWriter = cv.VideoWriter(outputVideo, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (origWidth, origHeight))
        '''
        if ret == False : 
            break 
        if frameId < startFrame : 
            continue 
        if frameId > endFrame : 
            break 
        
        if reidResult[frameId] == -1 : 
            MaskResult.append(frameResult)
            continue
        
        if(len(trackingResult[frameId]) != len(distanceResult[frameId])) : 
            print("[ERROR] in frame{} : tracking Result and distance Result length is different!! ".format(frameId))
            print("trackingResult length is {}".format(len(trackingResult[frameId])))
            print("distanceResult length is {}".format(len(distanceResult[frameId])))
            MaskResult.append(frameResult)        
            continue 
    
        for idx in range(0, len(trackingResult[frameId])) : 
            if distanceResult[frameId][idx] == False : 
                frameResult.append(MaskToken.NotNear)
                continue 
            
            #run Face Detector 
            bbox = trackingResult[frameId][idx].bbox 
            cropedPerson = raw_img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]      
            PersonWidth = cropedPerson.shape[1]
            PersonHeight = cropedPerson.shape[0]
            resizedPerson = cv.resize(cropedPerson,  (faceDetector.width, faceDetector.height))                   
            rgbResizedImage = cv.cvtColor(resizedPerson, cv.COLOR_BGR2RGB)
            
            objList = faceDetector.inference(rgbResizedImage)    
             
            if len(objList) == 0 : 
                frameResult.append(MaskToken.NotMasked)
                continue 
            
            if not 'bbox' in objList[0].keys() : 
                frameResult.append(MaskToken.NotMasked)
                continue
            
            #run MaskDetection             
            obj = objList[0]
            face_bbox = obj['bbox']  # [ymin, xmin, ymax, xmax]
            xmin, xmax = np.multiply([face_bbox[1], face_bbox[3]], PersonWidth)
            ymin, ymax = np.multiply([face_bbox[0], face_bbox[2]], PersonHeight)
            cropedFace = cropedPerson[int(ymin):int(ymin) + (int(ymax) - int(ymin)), int(xmin):int(xmin) + (int(xmax) - int(xmin))]
            cropedFace = cv.resize(cropedFace, (maskDetector.width, maskDetector.height)) 
            cropedFace = cropedFace/255.0                    

            faceMaskResult, scores = maskDetector.inference(np.array([cropedFace]))
            if faceMaskResult[0] == True : 
                frameResult.append(MaskToken.Masked)
            else : 
                frameResult.append(MaskToken.NotMasked)   
        
        MaskResult.append(frameResult)
        
    inputCapture.release() 
    print("== [MaskDetect ] : finish Mask Detection == ")

