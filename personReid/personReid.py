
import os, sys
# 상위 디렉토리 절대 경로 추가
# ~/covid19_cctv_analyzer
root_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_path + "/personReid/top-dropblock")
sys.path.append(root_path + "/personReid/LA_Transformer")

from main import config_for_topdb, run_top_db_test

from configs import runInfo
from collections import Counter # for test

# input_video_path = 'OxfordTownCentreDataset.avi'
input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame
query_image_path = runInfo.query_image_path

    
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

def personReid_topdb(trackingRslt, reidRslt):
    # CUDA_VISIBLE_DEVICES를 0으로 설정하지 않으면 topdb 돌릴 때 아래와 같은 err가 뜬다
    # TypeError: forward() missing 1 required positional argument: 'x'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    top_db_engine, top_db_cfg = config_for_topdb( root_path , query_image_path=query_image_path)
    run_top_db_test(engine=top_db_engine, cfg=top_db_cfg, 
                    start_frame=start_frame, end_frame=end_frame,
                    input_video_path=input_video_path, output_video_path=output_video_path, 
                    tracking_list=trackingRslt, reid_list=reidRslt, 
                    query_image_path=query_image_path)
    # 지금 reidRslt에서 확진자가 없는 경우(-1)는 나오지 않는다. (reid 정확성 문제 때문에)

def personReid_la_transformer(trackingRslt, reidRslt):
    calculation_mode = 'custom'
    
    if calculation_mode == 'custom':
        from la_transformer import config_la_transformer, run_la_transformer
    elif calculation_mode == 'original':
        from la_transformer_original_calc import config_la_transformer, run_la_transformer
        
    model, data_transforms = config_la_transformer(root_path)
    run_la_transformer(model=model, data_transforms=data_transforms,
                    root_path=root_path, query_image_path=query_image_path,
                    start_frame=start_frame, end_frame=end_frame,
                    input_video_path=input_video_path, output_video_path=output_video_path, 
                    tracking_list=trackingRslt, reid_list=reidRslt,
                    debug_enable=False,
                    debug_logging_file_path=root_path+"/la_trans_log.txt")

def runPersonReid(trackingRslt, reidRslt, select_reid_model):
    
    if select_reid_model == 'topdb':
        personReid_topdb(trackingRslt, reidRslt)
    elif select_reid_model == 'la':
        personReid_la_transformer(trackingRslt, reidRslt)
    elif select_reid_model == 'fake':
        fakeReid(trackingRslt, reidRslt)
    else:
        print("Plz Select PersonReid model")
        sys.exit()