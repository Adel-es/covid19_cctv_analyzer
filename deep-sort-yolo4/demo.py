#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

import os, sys
from absl.testing.absltest import main
# 상위 디렉토리 절대 경로 추가
# ../JCW/covid19_cctv_analyzer
root_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_path)
sys.path.append(root_path + '/top-dropblock')
from main import config_for_topdb, run_top_db_test                  # config_for_topdb()
from torchreid.engine import engine

warnings.filterwarnings('ignore')

from gpuinfo import GPUInfo
import torch
import gc

from multiprocessing import Process, Manager
# import multiprocessing as mp
# useMultiProcessing = True
# def run_topdb(gallery, top_db_engine, top_db_cfg):
#     run_top_db_test(gallery_data=gallery, engine=top_db_engine, cfg=top_db_cfg)

class TrackResult:
    def __init__(self, bbox, tid):
        self.bbox = bbox
        self.tid = tid
    
manager = Manager()
tracking_list = manager.list()

def main():
    yolo = YOLO()
    print(" * main start * ") 
    print(GPUInfo.get_users(1))
    GPUInfo.get_info()
    # exit(0)
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = True
    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = 'video.webm'
    if asyncVideo_flag :
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4"
    # top_db_engine, top_db_cfg = config_for_topdb( root_path )
    cam_id = 0;     # 임의로 cam_no 정의
    frame_no = -1   # 임의로 frame_no 정의
    
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        frame_no += 1 # frame no 부여
        if frame_no == 60: # test용: 5번만 test해보기
            break
        
        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image) ### useYOLO
        
        if tracking:
            features = encoder(frame, boxes)

            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]
        else:
            detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracking_frame_list = []
        if tracking:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1.5e-3 * frame.shape[0], (0, 255, 0), 1)
                
                # gallery_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # frame[y:y+h , x:x+w]
                
                # cv2.imwrite('tempData/'+str(track.track_id)
                #             +'_'+str(frame_no)
                #             +'_'+str(cam_id)+'.jpg', gallery_image)
                # gallery.append((gallery_image, track.track_id, cam_id))
                boundary_box = (int(bbox[1]),int(bbox[3]), int(bbox[0]),int(bbox[2])) # frame[y:y+h , x:x+w]
                tracking_frame_list.append( TrackResult(boundary_box, track.track_id) )
                
        tracking_list.append(tracking_frame_list)
        # print(gallery)
        # torch.cuda.empty_cache()
        # print(" * right before test * ") 
        # GPUInfo.get_users(1)        
        # GPUInfo.get_info()
        # if useMultiProcessing:
        #     p = ctx.Process(target=run_topdb, args=(gallery, top_db_engine, top_db_cfg))
        #     # p = Process(target=run_topdb, args=(gallery, top_db_engine, top_db_cfg))
        #     p.start()
        #     p.join()
        # else:
        # print(torch.cuda.is_available())
        # run_top_db_test(gallery_data=gallery, engine=top_db_engine, cfg=top_db_cfg)
        
        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                cls = det.cls
                cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1.5e-3 * frame.shape[0], (0, 255, 0), 1)

        #cv2.imshow('', frame)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1./(time.time()-t1))) / 2
            print("FPS = %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # gc.collect()
    # torch.cuda.empty_cache()
    # print(" * after frame loop * ") 
    # GPUInfo.get_users(1)
    # GPUInfo.get_info()
    
    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    p = Process(target=main)
    p.start()
    p.join()

    # CUDA_VISIBLE_DEVICES를 0으로 설정하지 않으면 topdb 돌릴 때 아래와 같은 err가 뜬다 ㅠㅠ
    # TypeError: forward() missing 1 required positional argument: 'x'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    top_db_engine, top_db_cfg = config_for_topdb( root_path )
    run_top_db_test(engine=top_db_engine, cfg=top_db_cfg, tracking_list=tracking_list)
    

    