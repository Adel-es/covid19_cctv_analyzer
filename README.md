# 실행 방법
## Setting
아래를 순서대로 실행한다.
```
$ cd deep-sort-yolo4
$ ./model_data/download_weights.sh
$ python convert.py
```

## Optional Setting
`deep-sort-yolo4/demo.py` 에서 세부 설정 값을 변경할 수 있다.
- `input_video_path` : 분석할 입력 영상 경로
- `output_video_path` : 출력 영상 경로
- `start_frame` : 입력 영상에서 분석을 시작할 frame 시점
- `end_frame` : 입력 영상에서 분석을 종료할 frame 시점

## Execution
```
$ cd deep-sort-yolo4
$ python demo.py
```

# Deep-SORT-YOLOv4
referenced from https://github.com/LeonLok/Deep-SORT-YOLOv4

# top-dropblock
referenced from https://github.com/RQuispeC/top-dropblock

