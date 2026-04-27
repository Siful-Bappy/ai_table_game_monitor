'''

   Ultralytics YOLO RKNN 

   (c) 2025 heejune@seoultech.ac.kr 


   - convert yolo pytorch model to RKNN model using Ultralyptics 
   - run the rknn YOLO model 
   - post-processing of YOLO model, NMS etc


'''
import numpy as np
import cv2


def yolo_pt2onnx(pt_name_or_path):

    # Export YOLOv8n (smallest) to ONNX
    from ultralytics import YOLO

    model = YOLO(pt_name_or_path)
    model.export(format="onnx", opset=12)  # export to yolov8n.onnx

def yolo_onnx2rknn(onnx_path, tgt_platform = 'rk3588'):

    from rknn.api import RKNN

    rknn = RKNN()
    # 1. CONFIG 
    rknn.config(
        mean_values=[[123.7, 116.3, 103.0]],  # Scaled ImageNet Means (R, G, B)
        std_values=[[58.4, 57.1, 57.4]],      # Scaled ImageNet STDs (R, G, B)
        target_platform=tgt_platform)
    '''
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=tgt_platform
    )
    '''

    # 2. ONNX 모델 로드
    ret = rknn.load_onnx(model=onnx_path)
    if ret == 0:
        print(f"loaded onnx model, {onnx_path}")
    else:
        print('load onnx failed')
        return

    # 3. build
    print(f"Start to build RKNN model...")
    ret = rknn.build(do_quantization=False)
    if ret == 0:
        print(f"Success in building RKNN model")
    else:
        print('build failed')
        return

    # 4. export RKNN
    rknn_path = onnx_path.replace(".onnx", ".rknn")
    ret = rknn.export_rknn(rknn_path)
    if ret == 0:
        print(f"Success in exporting:{rknn_path}")
    else:
        print('export failed')
        return

class YOLO_RKNN():

    def __init__(self, rknn_path = "yolo11n_tablegame.rknn", target = 'rk3588'):

        from rknn.api import RKNN
        self.rknn = RKNN()

        # Load RKNN model
        self.rknn.load_rknn(rknn_path)

        # Initialize runtime
        ret = self.rknn.init_runtime(target)
        if ret != 0:
            print('init runtime failed')
            raise RuntimeError("RKNN init_runtime failed")

    def close(self):
        if hasattr(self, "rknn") and self.rknn:
            self.rknn.release()
            self.rknn = None

    def __del__(self):
    
        if hasattr(self, "rknn") and self.rknn:
            self.rknn.release()
    
    def predict(self, original_image):

        ''' 
           somehow RKNN doesnot provide the scaling and NMS etc.
           Only runt the NN model 
        '''
        # 1. resize for 640x640
        height, width = img.shape[:2]   
        MODEL_INPUT_SIZE = 640 
        scale_w = width / MODEL_INPUT_SIZE
        scale_h = height / MODEL_INPUT_SIZE
        image_640 = cv2.resize(original_image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        
        # 2. inference model       
        outputs = self.rknn.inference(inputs=[image_640], data_format='nhwc')
        # 3. NMS
        dets = self.postprocess(outputs)
        # 4. scaling back 
        for det in dets:
            xyxy, conf, cls_id = det      
            # Scaling is correct if xyxy contains 640x640 pixel values:
            xyxy[0] = xyxy[0] * scale_w  # x1 * width_scale
            xyxy[1] = xyxy[1] * scale_h  # y1 * height_scale
            xyxy[2] = xyxy[2] * scale_w  # x2 * width_scale
            xyxy[3] = xyxy[3] * scale_h  # y2 * height_scale

        return dets

    def postprocess(self, outputs, conf_thres=0.125, iou_thres=0.45):

        # outputs: list -> take first (RKNN outputs list)
        out = outputs[0]
        #print(f"out.shape:{out.shape}") # shape: (1, 7, 8400) 
        if out.shape[1] < out.shape[2]:  # (1, 7, 8400)
            out = out.transpose(0, 2, 1)  # → (1, 8400, 7)

        preds = out[0]                 # (8400, 7)
        boxes = preds[:, :4]           # xc,yc,w,h?
        #scores = preds[:, 4:5]         # objectness
        class_scores = preds[:, 4:]    # class probabilities
        #print(f"boxes[{boxes.shape}]") 
        #print(f"scores[{scores.shape}]") 
        '''
        for n in range(0, len(scores), 10):
            print(f"{scores[n:n+10].T}")
        '''
        #print(f"classes_scores[{class_scores.shape}]") 

        # objectness × class score
        cls_ids = np.argmax(class_scores, axis=1)
        cls_scores = class_scores[np.arange(len(preds)), cls_ids]
        #conf = scores[:, 0] * cls_scores
        conf = cls_scores
        # threshold
        keep = conf > conf_thres
        '''
        CLASS_THRESHOLDS = {
            0: 0.015,  # Lower threshold for 'hand'
            1: 0.25,  # Default for 'chip'
            2: 0.25   # Default for 'card'
        }
        keep_list = []
    
        # Iterate through all detection proposals
        for i in range(len(conf)):
            current_cls_id = cls_ids[i]
            current_conf = conf[i]        
            # Check if the confidence exceeds the specific class threshold
            # Use .get() with a fallback (like 0.25) for safety if more classes exist
            threshold = CLASS_THRESHOLDS.get(current_cls_id, conf_thres) 
            if current_conf > threshold:
                keep_list.append(True)
                #if current_cls_id == 0:
                #    conf[i] = conf_thres + 0.1
                #    
            else:
                keep_list.append(False)
        keep = np.array(keep_list)
        ''' 

        boxes = boxes[keep]
        conf = conf[keep]
        cls_ids = cls_ids[keep]

        # xywh → xyxy
        '''
        print(f"boxes:{boxes}")
        print(f"conf:{conf}")
        print(f"cls_ids:{cls_ids}")
        '''
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2
      
        '''
        dets = []
        for i in range(len(conf)):
            print(f"before nms: {xyxy[i]}, {conf[i]}, {cls_ids[i]}")
            dets.append((xyxy[i], conf[i], cls_ids[i]))
            #dets.append((boxes[i], conf[i], cls_ids[i]))

        #return dets 
        '''
        
        # NMS
        idxs = cv2.dnn.NMSBoxes(
            bboxes=xyxy.tolist(),
            scores=conf.tolist(),
            score_threshold=conf_thres,
            nms_threshold=iou_thres
        )

        #print(f"idxs:{idxs}")
        if len(idxs) == 0:
            return []

        idxs = idxs.flatten()
        dets = []
        for i in idxs:
            #print(f"after nms: {xyxy[i]}, {conf[i]}, {cls_ids[i]}")
            dets.append((xyxy[i], conf[i], cls_ids[i]))

        return dets 


def test_yolo_rknn(rknn_path, testimg_path, target = 'rk3588'):


    yolo_rknn = YOLO_RKNN(rknn_path, target)

    img = cv2.imread(testimg_path)
    
    import time
    ts = time.time()
    num = 1
    print("Test {num} inferences")

    for n in range(num):
        dets = yolo_rknn.predict(img_640)
        # draw results
        for (xyxy, conf, cls_id) in dets:
            xyxy = xyxy.astype(int)
            x1, y1, x2, y2 = xyxy
            print(f"dect: cls={cls_id},bbox={xyxy}, conf={conf:.2f}")
            x1, y1, x2, y2 = xyxy.astype(int)
            #cv2.rectangle(img, (200, 400), (400, 600), (255,255,255), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{cls_id}:{conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    print(f"time per image:{(time.time()-ts)/num:.3f}")

    # display
    cv2.imshow("RK3588 YOLO", img)
    if cv2.waitKey(0) == 27:
        pass
    cv2.destroyAllWindows()



def test_video_yolo_rknn(rknn_path, video_path, target = 'rk3588'):

    yolo_rknn = YOLO_RKNN(rknn_path, target)

    #img = cv2.imread(testimg_path)
    # 2. 비디오 열기
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps =  cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    t_sleep = 1# int(1000/fps) 
    width, height = width//2, height//2
    # 출력 비디오 설정

    opt_tracking = False
    viz_colors = [(0,0,255),(0,255,0), (255,0,0)]
    viz_clsnames = ['hand', 'chip', 'card'] 
       

    import time
    ts = time.time()
    fn = 0
    while True:       
        ret, frame = cap.read()
        
        if not ret:
            break        
                  
        frame = cv2.resize(frame, (width, height))
        fn +=1
            
        dets = yolo_rknn.predict(frame)
        # draw results
        for (xyxy, conf, cls_id) in dets:
            xyxy = xyxy.astype(int)
            x1, y1, x2, y2 = xyxy
            print(f"dect: cls={cls_id},bbox={xyxy}, conf={conf:.2f}")
            #cv2.rectangle(img, (200, 400), (400, 600), (255,255,255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls_id}:{conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
               
        # display
        cv2.imshow("RK3588 YOLO", frame)
        k = cv2.waitKey(t_sleep)
        if k == ord('q'):        
            break
        elif k == ord('s'):            
            t_sleep = 0
        elif k == ord('p'):
             t_sleep = 30    

    print(f"time per image:{(time.time()-ts)/fn:.3f}")
    cap.release()
    cv2.destroyAllWindows()
 

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        yolo_pt2onnx("yolo11n")
    elif sys.argv[1].endswith(".pt"):
        yolo_pt2onnx(sys.argv[1])
    elif sys.argv[1].endswith(".onnx"):
        yolo_onnx2rknn(sys.argv[1])
    elif sys.argv[1].endswith(".rknn"):
        if len(sys.argv)< 3:
            print(f"python {sys.argv[0]} rkknfile imagefile")
            exit()
        if sys.argv[2].endswith(".mp4") or sys.argv[2].endswith(".avi"):
            test_video_yolo_rknn(sys.argv[1], sys.argv[2])
        elif sys.argv[2].endswith(".jpg") or sys.argv[2].endswith(".png"):
            test_yolo_rknn(sys.argv[1], sys.argv[2])
        else:
            print(f"Cannot support this type of input: {sys.argv[2]}")
            exit()            
    else:
        print("Not support!")


