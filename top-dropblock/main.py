import sys
import os
import os.path as osp
import warnings
import time
import argparse

import torch
import torch.nn as nn

from default_config import (
    get_default_config, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs, engine_test_kwargs
)
import torchreid
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)

from torchreid.utils import read_image
import glob

import cv2
from videocaptureasync import VideoCaptureAsync
import imutils.video
from PIL import Image
import numpy as np

def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'triplet_dropbatch':
            engine = torchreid.engine.ImageTripletDropBatchEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                weight_db_t=cfg.loss.dropbatch.weight_db_t,
                weight_db_x=cfg.loss.dropbatch.weight_db_x,
                top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'triplet_dropbatch_dropbotfeatures':
            engine = torchreid.engine.ImageTripletDropBatchDropBotFeaturesEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                weight_db_t=cfg.loss.dropbatch.weight_db_t,
                weight_db_x=cfg.loss.dropbatch.weight_db_x,
                weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'triplet':
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        else:
            exit("ERROR")
    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )
        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms

    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('-s', '--sources', type=str, nargs='+', help='source datasets (delimited by space)')
    parser.add_argument('-t', '--targets', type=str, nargs='+', help='target datasets (delimited by space)')
    parser.add_argument('--transforms', type=str, nargs='+', help='data augmentation')
    parser.add_argument('--root', type=str, default='', help='path to data root')
    parser.add_argument('--gpu-devices', type=str, default='',)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_random_seed(cfg.train.seed)

    if cfg.use_gpu and args.gpu_devices:
        # if gpu_devices is not specified, all available gpus will be used
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
    
    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    
    datamanager = build_datamanager(cfg)
    
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        args.start_epoch = resume_from_checkpoint(cfg.model.resume, model, optimizer=optimizer)

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))
    

def main_concat_with_track( config_file_path, data_root_path ):
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--config-file', type=str, default='', help='path to config file')
    # parser.add_argument('-s', '--sources', type=str, nargs='+', help='source datasets (delimited by space)')
    # parser.add_argument('-t', '--targets', type=str, nargs='+', help='target datasets (delimited by space)')
    # parser.add_argument('--transforms', type=str, nargs='+', help='data augmentation')
    # parser.add_argument('--root', type=str, default='', help='path to data root')
    # parser.add_argument('--gpu-devices', type=str, default='',)
    # parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
    # args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    # if args.config_file:
    #     cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(config_file_path)
    cfg.data.root = data_root_path
    
    # reset_config(cfg, args)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_random_seed(cfg.train.seed)

    # if cfg.use_gpu and args.gpu_devices:
    #     # if gpu_devices is not specified, all available gpus will be used
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
    
    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    
    datamanager = build_datamanager(cfg)
    
    print(type(datamanager))
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids, # class 종류 개수를 특정할 수 있나?
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    
    if cfg.use_gpu:
        device = torch.device("cuda:3")
        model = nn.DataParallel(model).to(device)
        # model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    # if cfg.model.resume and check_isfile(cfg.model.resume):
    #     args.start_epoch = resume_from_checkpoint(cfg.model.resume, model, optimizer=optimizer)

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)

    return engine, cfg
    # gallery_data = read_gallery_image() # type : [(img, pid, camid), ...]
    
    # engine.test_only(gallery_data = gallery_data, **engine_test_kwargs(cfg))

def read_gallery_image():
    # gallery_dir_path = "/home/gram/JCW/covid19_cctv_analyzer/top-dropblock/data/query/"
    # gallery_img_path = [ gallery_dir_path + i for i in ["01.PNG", "002.PNG"] ]
    
    # gallery_dir_path = "/home/gram/JCW/covid19_cctv_analyzer/top-dropblock/data/tempDataset/bounding_box_test/"
    gallery_dir_path = "/home/gram/JCW/covid19_cctv_analyzer/deep-sort-yolo4/tempData/"
    
    gallery_img_path = [ gallery_dir_path + i for i in os.listdir(gallery_dir_path)]
    gallery_img_path = glob.glob(osp.join(gallery_dir_path, '*.jpg'))
        
    data = []
    pid = 1     # temp
    camid = 0   # temp
    for img_path in gallery_img_path:
        img = read_image(img_path)
        pid = int(img_path.split("/")[-1].split("_")[0])
        data.append((img, pid, camid))
        # pid += 1
        
    return data

def config_for_topdb(root_path):
    config_file_path = root_path + "/top-dropblock/configs/im_top_bdnet_test_concat_track.yaml"
    data_root_path = root_path + "/top-dropblock/data"
    return main_concat_with_track(config_file_path, data_root_path)

# def run_top_db_test(gallery_data, engine, cfg):
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
#     engine.test_only(gallery_data = gallery_data, **engine_test_kwargs(cfg))

# def run_top_db_test(engine, cfg):
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
#     gallery_data = read_gallery_image() # type : [(img, pid, camid), ...]
#     engine.test_only(gallery_data = gallery_data, **engine_test_kwargs(cfg))

def crop_frame_image(frame, bbox):
    return Image.fromarray(frame).crop( (int(bbox[2]),int(bbox[0]), int(bbox[3]),int(bbox[1])) ) # (start_x, start_y, start_x + width, start_y + height) 
    # return frame[ int(bbox[0]):int(bbox[1]), int(bbox[2]):int(bbox[3]) ] # frame[y:y+h , x:x+w]
    
def run_top_db_test(engine, cfg, shm_bbox):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
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
    frame_no = 0   # 임의로 frame_no 정의
    
    ret, frame = video_capture.read()  # frame shape 640*480*3
    gallery_data = []
    for bbox_info in shm_bbox:
        if frame_no == 30: # test용: 5번만 test해보기
            break
        if bbox_info[2] > frame_no:
            # evaluate person-reid
            if len(gallery_data) > 0:
                matching_qid_distance = engine.test_only(gallery_data = gallery_data, **engine_test_kwargs(cfg))
                # query와 동일한 사람을 찾은 결과를 받아서
                # 결과 사진을 jpg로 저장함.
                for ri in matching_qid_distance: # ri = (query pid, query pid에 매칭되는 gallery data의 1st index)
                    foundImage = gallery_data[ri[1]][0]
                    g_pid = gallery_data[ri[1]][1]
                    cv2.imwrite('/home/gram/JCW/covid19_cctv_analyzer_multi_proc/top-dropblock/data/equal_query/'
                            +str(g_pid)+'_qid_'+str(ri[0])+'.jpg', #gpid_qpid.jpg
                            np.asarray(foundImage, dtype=np.uint8) )
                gallery_data = []        
            # capture next frame
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True:
                break
            frame_no += 1 # frame no 부여
            continue
        
        if bbox_info[2] == frame_no:
            # bbox = bbox_info[0]
            # cv2.imwrite('/home/gram/JCW/covid19_cctv_analyzer_multi_proc/top-dropblock/data/webcam/'
            #             +str(bbox_info[1])+'_'+str(bbox_info[2])+'.jpg', 
            #             frame[bbox[0]:bbox[1], bbox[2]:bbox[3]] )
            # crop image from frame
            crop_image = crop_frame_image(frame, bbox_info[0])
            gallery_data.append( (crop_image, bbox_info[1], bbox_info[3]) ) # (image, pid, camid)
    
        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()
        
    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    engine, cfg = config_for_topdb("../")
    gallery_data = read_gallery_image() # type : [(img, pid, camid), ...]
    run_top_db_test(gallery_data, engine, cfg)
    # main()