import os 
import numpy as np
from ultralytics import YOLO
import cv2
from sort.sort import Sort  # https://github.com/abewley/sort

'''
   TableGame MOT 
   
   
   - training YOLO
   - test with video using YOLO + SORT
   - export TensorRT and RKNN
   - benchmarking 
   
   
'''   
       
def train_yolo(modelname, yolo_yaml, benchmark = False):

    # Load pretrained YOLO V11
    model = YOLO(modelname)
    print(f"All classes: {model.names}")

    if benchmark:
        for format in ['-', 'engine']: # 'rknn'
            model.benchmark(
                data=yolo_yaml,
                format= format,
                #int8=True, 
                half=True,
                imgsz=640,
                device='cuda:0' # GPU를 명시적으로 사용
                )
        '''
        # pytorch model performance
        results = model.val(
           data='yolo_hand.yaml', 
           imgsz=640, 
           half=True, 
           device='cuda:0' # Jetson Orin Nano의 GPU를 명시적으로 사용
        )
        print(results)
        '''
    else:
        model.train(
        data=yolo_yaml,
        epochs=40,
        imgsz=640,
        batch=16,
        workers=2,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,    # Helps regularization
        warmup_epochs=3,        # Warmup for stable initial convergence
        augment=True)
    


def export_to_rknn(pytorch_modelname):
    '''
    note: rknn-toolkit2-2.3.2 uses typing-extensions==4.4.0 and onnx==1.13.1 

    related errors:
        AttributeError: module 'typing_extensions' has no attribute 'deprecated'
        AttributeError: module 'onnx' has no attribute 'mapping'
    '''

    # Load the YOLO11 model
    model = YOLO(pytorch_modelname)
         
    # Export the model to RKNN format
    # 'name' can be one of rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118
    model.export(format="rknn",
                 name="rk3588",
                 opset=11   
                 )  # creates '/yolo11n_rknn_model'



#-----------------------------------------------------
# TEST
#-----------------------------------------------------

def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # The union area
    union_area = box1_area + box2_area - intersection_area

    # The IoU
    return intersection_area / union_area
    
    
def test_yolo(model_name, img_path, viz = True, result_file = None, debug = False) :

    # 1. 모델 불러오기
    model = YOLO(model_name)  # trained with VisDrone DET dataset, not pretrained YOLO11n

    # 2. 비디오 열기
    frame  = cv2.imread(img_path)
    if frame is None:
        return 
    height, width = frame.shape[:2] 
    frame = cv2.resize(frame, (width//2, height//2))
    height, width = frame.shape[:2] 
    print(f"image size: w={width}, h={height}")

    results = model.predict(frame, conf=0.4, verbose=False)
    result = results[0] # single frame (for batch size = 1) 
    frame = cv2.flip(frame,0)
    if result.boxes is not None:
        # YOLO 결과 bbox 추출 [x1, y1, x2, y2, score]
        for box in result.boxes.data.tolist():
            print(f"box:{box}")
            x1, y1, x2, y2, conf, cls = box
            if viz or out:
                tl = (int(x1), int(y1))
                br = (int(x2), int(y2))
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)

        if viz:
            cv2.imshow("MOT Tracking", frame)
            k = cv2.waitKey(0)
            if k == ord('q'):
                return

    cv2.destroyAllWindows()


def tracking_video(model_name, video_path, viz = True, result_file = None, debug = False) :

    # 1. 모델 불러오기
    model = YOLO(model_name)  # trained with VisDrone DET dataset, not pretrained YOLO11n
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # SORT 초기화

    # 2. 비디오 열기
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps =  cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    t_sleep = 1 # int(1000/fps) 
    width, height = width//2, height//2
    # 출력 비디오 설정

    opt_tracking = False
    viz_colors = [(0,0,255),(0,255,0), (255,0,0)]
    viz_clsnames = ['hand', 'chip', 'card'] 
     
    if result_file:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(result_file, fourcc, fps, (width, height))

    import time
    ts = time.time()
    fn = 0
    id2class = {}

    card_w = 90*4
    card_h = 160*4
    
    while True:
        ret, frame_o = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame_o, (width, height))

        # -----------------------------
        # 3. YOLO 탐지
        # -----------------------------
        results = model.predict(frame, conf=0.4, verbose=False)
        fn  +=1
        dets_list = []
        result = results[0] # single frame (for batch size = 1) 
        if result.boxes is not None:
            # YOLO 결과 bbox 추출 [x1, y1, x2, y2, score]
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                dets_list.append([x1, y1, x2, y2, conf, cls]) # in absolute coordinate 
                if not opt_tracking and (viz or out):
                    tl = (int(x1), int(y1))
                    br = (int(x2), int(y2))
                    cls = int(cls)
                    cv2.rectangle(frame, tl,  br, viz_colors[cls], 2)
                    #cv2.putText(frame, f"{viz_clsnames[cls]}", (int(x1), int(y1 - 10)), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, viz_colors[cls], 2)     
                     
        if opt_tracking:
                
            if len(dets_list) > 0:
                dets = np.array(dets_list)
            else:
                dets = np.empty((0, 5))

            # -----------------------------
            # 4. SORT tracking
            # -----------------------------
            tracks = tracker.update(dets) # tracks = [x1, y1, x2, y2, track_id]
        
            # -----------------------------
            # 5. Book keeping IDs and CLSs since SORT doesnot Keep the CLS (only track_id) 
            # -----------------------------
            # 5.1 remove disapperared  
            current_track_ids = {int(track[4]) for track in tracks}
            if debug:
                print(f"current_track_ids : {current_track_ids }")
            ids_to_remove = set(id2class.keys()) - current_track_ids
            for track_id in ids_to_remove:
                if debug:
                    print(f"delete track_id:{track_id} {id2class[track_id] }")
                del id2class[track_id] # 추적되지 않는 ID의 CLS 정보 삭제
        
            final_tracks = [] 
            iou_threshold = 0.9  # Set a high threshold for a confident match
            for track in tracks:
                x1, y1, x2, y2, track_id = track.astype(int)
           
                # cls info from detector!
                track_box = track[:4]
                track_id = track[4]
                matched_cls = None
                #print(f"track id:{track_id}")
                # A. 딕셔너리에서 클래스 조회
                if track_id in id2class:
                    matched_cls = id2class[track_id]
                # B. ID가 처음 생성된 경우 (New Track) 
                else:
                    #print(f"new object:{track_id}")
                    best_iou = 0
                    for det in dets_list:
                        det_box = det[:4]
                        det_cls = int(det[5]) # Assuming class is the 6th element and needs to be an integer
                        iou = calculate_iou(track_box, det_box)
                        #print(f"iou:{iou}")
                        # If IoU is high, we've found the class for this track
                        #if iou > iou_threshold:
                        if iou > best_iou:
                            #best_iou = iou
                            best_cls = det_cls
                            break # Found a match, move to the next track
                    if best_cls is not None:
                        matched_cls = best_cls
                        id2class[track_id] = matched_cls # **⭐ 클래스 정보 저장 (핵심 수정)**    
                        if debug:
                            print(f"add new track_id:{track_id} {matched_cls}")        
                    else:
                        if debug:
                            print("cannot find matched: track_id:{track_id} ")        

                # Add the track info along with the class
                if matched_cls is not None:
                    # Format: [x1, y1, x2, y2, track_id, class_id]
                    final_tracks.append(np.append(track, matched_cls))       
            
                # annotation 
                #
                if viz or out:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), viz_colors[matched_cls], 2)
                    cv2.putText(frame, f"{viz_clsnames[matched_cls]}", (x1, y1 - 10),
                    #cv2.putText(frame, f"ID:{track_id:.0f}/cls:{matched_cls}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, viz_colors[matched_cls], 2)     
            
        
        if fn%fps == 0:
            print(f"fn={fn}/{frame_count}")
                
        if viz or out:                
            cv2.putText(frame, f"fn={fn}/{frame_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if out:
            out.write(frame)
        if viz:
            cv2.imshow("MOT Tracking", frame)
            k = cv2.waitKey(t_sleep)
            if k == ord('q'):
                break
            elif k == ord('s'):
                t_sleep  = 0
            elif k == ord('p'):
                t_sleep  = 30
 
    print(f"speed={fn/(time.time() -ts):.1f}s")
    if cap:
        cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
        
    import sys
    
    if len(sys.argv) < 2:
        print(f"usage python {sys.argv[0]} <mp4(test)/yaml(train)/pt(export)>")
        exit()

    model_name = "yolo11n_tablegame.pt"
    #model_name = "yolo11n_hand.pt"

    if sys.argv[1].endswith('.yaml'): # train model 
        yolo_yaml = sys.argv[1] 
        train_yolo(model_name, yolo_yaml, benchmark = False)    

    elif sys.argv[1].endswith('.jpg') or sys.argv[1].endswith('.png'): # test image 
        print(f"YOLO test {sys.argv[1]}: using yolo model:{model_name}")  
        test_yolo(model_name, sys.argv[1])

    elif sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.avi'): # test video 
        print(f"tracking {sys.argv[1]}: using yolo model:{model_name} to mot_{sys.argv[1]}")  
        #tracking_video(model_name, sys.argv[1], result_file =  f"mot_{sys.argv[1]}")
        tracking_video(model_name, sys.argv[1], result_file = "mot.mp4") 

    elif sys.argv[1].endswith('.pt'): # export model to RKNN 
        pt_modelname = sys.argv[1]
        export_to_rknn(pt_modelname)

