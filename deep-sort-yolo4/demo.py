#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from multiprocessing import Process, Manager

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from distance import checkDistance
from distance import getCentroid
from collections import Counter # for test

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')

input_video_path = 'OxfordTownCentreDataset.avi'
output_video_path = 'output_yolov4.avi'
start_frame = 300
end_frame = 900

class TrackResult:
    def __init__(self, bbox, tid):
        self.bbox = bbox
        self.tid = tid

def detectAndTrack(trackingRslt):
    # Get detection model
    yolo = YOLO()

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Get tracker (Deep SORT)
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Prepare input videovideo_path
    video_capture = cv2.VideoCapture(input_video_path)
    frame_index = -1
    
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        
        # for test
        frame_index += 1
        if frame_index < start_frame:
            continue
        if frame_index > end_frame:
            break
        # for test

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)

        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                        zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        aFrameTracking = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            aFrameTracking.append( TrackResult(bbox, track.track_id) )
        
        trackingRslt.append(aFrameTracking)
    # end of while()
    
    video_capture.release()
    
    
def fakeReid(trackingRslt, reidRslt):
    # Find most frequent tid(person) in video frames
    idList = []
    for aFrameTracking in trackingRslt:
        for idx, person in enumerate(aFrameTracking):
            idList.append(person.tid)
    if len(idList) == 0:
        print("Nobody in this video: {}".format(input_video_path))
        print("Tracking result: {}".format(trackingRslt))
        confirmed_id = -1
    else:
        confirmed_id = Counter(idList).most_common(n=1)[0][0]
    
    # Fill in the reidRslt
    for aFrameTracking in trackingRslt:
        confirmed_idx = -1
        for idx, person in enumerate(aFrameTracking):
            if person.tid == confirmed_id:
                confirmed_idx = idx
                break
        reidRslt.append(confirmed_idx)


if __name__ == '__main__':
    startTime = time.time()
    with Manager() as manager:
        # 공유 객체 생성
        tracking = manager.list()
        reid = manager.list()
        distance = manager.list()
        
        # 프로세스 실행 (영상 단위 처리)
        detectTrackProc = Process(target=detectAndTrack, args=(tracking, ))
        reidProc = Process(target=fakeReid, args=(tracking, reid))
        distanceProc = Process(target=checkDistance, args=(tracking, reid, distance))
        
        detectTrackProc.start()
        detectTrackProc.join()
        
        reidProc.start()
        reidProc.join()
        
        distanceProc.start()
        distanceProc.join()
        
        
        # ==== UI ====
        # Prepare input video
        video_capture = cv2.VideoCapture(input_video_path)
        frame_index = -1
        
        # Prepare output video
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
        
        # for test
        while True:
            if start_frame == 0:
                break
            ret, frame = video_capture.read()
            frame_index += 1
            if ret != True:
                break
            if frame_index < start_frame-1:
                continue
            elif frame_index == start_frame-1:
                break
            else:
                print("Frame capture error! Check start_frame and end_frame: {}, {}".format(start_frame, end_frame))
        # for test
        
        for aFrameTracking, aFrameReid, aFrameDistance in zip(tracking, reid, distance):
            ret, frame = video_capture.read()
            frame_index += 1
            if ret != True:
                break
            # for test
            if frame_index > end_frame:
                break
            # for test
            
            # Draw detection and tracking result for a frame
            TEXT_UP_FROM_BBOX = 2
            for person in aFrameTracking:
                cv2.rectangle(frame, (int(person.bbox[0]), int(person.bbox[1])), (int(person.bbox[2]), int(person.bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, "ID: " + str(person.tid), (int(person.bbox[0]), int(person.bbox[1])-TEXT_UP_FROM_BBOX), 0,
                            8e-4 * frame.shape[0], (0, 255, 0), 3)
            
            if aFrameReid != -1: # if there is confirmed case
                # Draw red bbox for confirmed case
                confirmed = aFrameTracking[aFrameReid]
                cv2.rectangle(frame, (int(confirmed.bbox[0]), int(confirmed.bbox[1])), (int(confirmed.bbox[2]), int(confirmed.bbox[3])), (0, 0, 255), 2)
                cv2.putText(frame, "ID: " + str(confirmed.tid), (int(confirmed.bbox[0]), int(confirmed.bbox[1])-TEXT_UP_FROM_BBOX), 0,
                            8e-4 * frame.shape[0], (0, 0, 255), 3)
                
                # Draw distance result for a frame
                c_stand_point = getCentroid(bbox=confirmed.bbox, return_int=True)
                for idx, is_close in enumerate(aFrameDistance):
                    if not is_close:
                        continue
                    closePerson = aFrameTracking[idx]
                    stand_point = getCentroid(bbox=closePerson.bbox, return_int=True)
                    cv2.line(frame, c_stand_point, stand_point, (0, 0, 255), 2)
                
            out.write(frame)
        out.release()
        print("Runing time:", time.time() - startTime)