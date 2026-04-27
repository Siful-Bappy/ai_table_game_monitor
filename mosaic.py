'''

  anonymization  by detecting hands and face and mosaicing them 

  
  (c) 20225 heejune@seoultech.ac.kr


  - use Google Mediapipe 
  - mosaic
  - generating YOLO training data for hand
  - @TODO use Tracking (SORT) 

'''
import os
import time
import numpy as np
import cv2
import mediapipe as mp

class Mosaicer:

    def __init__(self, mode = "hand"):
       
        if mode.find("hand") >=0:
            mp_hands = mp.solutions.hands
            self.hand_detector = mp_hands.Hands(static_image_mode=False, 
                                            max_num_hands=10, 
                                            min_detection_confidence=0.3,  # lower threshold (default 0.5)
                                            min_tracking_confidence=0.3)  # lower threshold (default 0.5))
        if mode.find("face") >=0:
            mp_face = mp.solutions.face_mesh
            self.face_detector = mp_face.FaceMesh(static_image_mode=False, max_num_faces=4)

        if mode.find("holistic") >=0:
            mp_holistic = mp.solutions.holistic
            self.holistic_detector = mp_holistic.Holistic(static_image_mode=False)
        #self.mp_drawing = mp.solutions.drawing_utils
        self.mode = mode
   

    def get_hand_face_mask(self, img):
 
        mask = np.zeros(img.shape[:2], dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        if self.mode.find("holistic") >= 0:
            #with self.mp_holistic.Holistic(static_image_mode=True) as holistic:
            results = self.holistic_detector.process(img)
            print(f"L:{results.left_hand_landmarks}, R:{results.right_hand_landmarks}, F:{results.face_landmarks}")
         
            # Hands
            if results.left_hand_landmarks:
                x_list = [int(lm.x * img.shape[1]) for lm in results.left_hand_landmarks.landmark]
                y_list = [int(lm.y * img.shape[0]) for lm in results.left_hand_landmarks.landmark]
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)
            
            if results.right_hand_landmarks:
                x_list = [int(lm.x * img.shape[1]) for lm in results.right_hand_landmarks.landmark]
                y_list = [int(lm.y * img.shape[0]) for lm in results.right_hand_landmarks.landmark]
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)
          
            # Head region from face landmarks
            if results.face_landmarks:
                x_list = [int(lm.x * img.shape[1]) for lm in results.face_landmarks.landmark]
                y_list = [int(lm.y * img.shape[0]) for lm in results.face_landmarks.landmark]
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)
            
           
        if self.mode.find("face") >= 0:
            face_results = self.face_detector.process(img)
            if face_results.multi_face_landmarks:
                #print(f"num faces: {len(face_results.multi_face_landmarks)}")
                for face_landmarks in face_results.multi_face_landmarks:
                    x_list = [int(lm.x * w) for lm in face_landmarks.landmark]
                    y_list = [int(lm.y * h) for lm in face_landmarks.landmark]
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    if y_min > h/2:
                        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)
    
        h_bboxes = []
        if self.mode.find("hand") >= 0:
            hand_results = self.hand_detector.process(img)
            num_hands = 0 
            if hand_results.multi_hand_landmarks:
                num_hands = len(hand_results.multi_hand_landmarks)
                #print(f"num hands: {num_hands}")
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    
                    h_bboxes.append(((x_min + x_max)/2, (y_min +y_max)/2, (x_max - x_min), (y_max - y_min)))  # x_center, y_center, width, height
                
                    lm = hand_landmarks.landmark 
                    if lm[0].y * h > 4*h/10:  # wrist is bottom
                        # hand masking 
                        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)
                        # fore-arm masking                                         
                        x_wrist = int(lm[0].x * w)
                        y_wrist = int(lm[0].y * h)
                        x_midbase = int(lm[9].x * w)
                        y_midbase = int(lm[9].y * h)
                        # 손바닥 방향 벡터 (중지->손목)
                        vx = x_wrist - x_midbase
                        vy = y_wrist - y_midbase
                        # 팔의 예상 연장 거리 (손길이의 약 1.5~2배)
                        extend_len = int(np.hypot(vx, vy) * 2.0)
                        # 팔의 끝점 (손목에서 벡터 방향으로 연장)
                        x_forearm = int(x_wrist + vx / np.hypot(vx, vy) * extend_len)
                        y_forearm = int(y_wrist + vy / np.hypot(vx, vy) * extend_len)
                        # 팔 영역을 직사각형으로 추정
                        cv2.line(mask, (x_wrist, y_wrist), (x_forearm, y_forearm), 255, 150)
                        #cv2.circle(mask, (x_forearm, y_forearm), 8, 255, -1)    
        
        return mask, num_hands, h_bboxes 

    def mosaic_image(self, img, mask, mosaic_scale=0.05):
        """
            img: 원본 이미지 (BGR)
            mask: 모자이크를 적용할 영역의 마스크 (0과 255)
            mosaic_scale: 모자이크 크기 비율 (작을수록 더 뭉개짐)
        """
        # 마스크 영역만 추출
        mask_bool = mask > 0  # True/False

        # 원본 이미지 복사
        img_mosaic = img.copy()

        # 모자이크 처리
        # 1. 마스크 영역 추출
        roi = img[mask_bool]

        # 2. 모자이크: resize 다운샘플 후 업샘플
        h, w = mask.shape
        small = cv2.resize(img, (0,0), fx=mosaic_scale, fy=mosaic_scale, interpolation=cv2.INTER_NEAREST)
        # 업샘플
        mosaiced = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 적용
        img_mosaic[mask, :] = mosaiced[mask, :]

        return img_mosaic


        

def mosaic_vid(video_file, record_file = None):
   
    import json
    
    mosaicer= Mosaicer(mode = "hand")

    # 1.2 open test video 
    cap = cv2.VideoCapture(video_file)  # or use 0 for webcam
    if cap is None:
        print(f"cannot open {video_file}")
        return 
        
    # Get width and height
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"resolution(wxh)={width}x{height}@{fps}")

    # 1.3 recording
    if record_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')   # or 'mp4v' for .mp4
        out = cv2.VideoWriter(record_file, fourcc, fps, (width, height))  
   
    fn = 1
    t_sleep = 1 #30
    
    #with mp_holistic.Holistic(static_image_mode=True) as holistic:
    
    while True:
    
        fn +=1
        ret, frame = cap.read()
        if ret == False or frame is None or fn > 24*5*60:
            break
            
        '''
        if fn < 24:  # skip 1 sec for this video @TODO make robust (ignore some wrong affine estimation case)
           continue
        '''
        #frame = cv2.resize(frame, (w,h))       
        

        # 2.  masking areas  
        mask, num_hands, _ =  mosaicer.get_hand_face_mask(frame)   
        '''  # TOO SLOW
         fg_mask = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        '''

        img_mosaiced = mosaicer.mosaic_image(frame, mask > 0 , mosaic_scale=0.05)
        #cv2.putText(img_mosaiced, f"fn={fn}, {fn/fps:.1f}s", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_mosaiced, f"fn={fn}, hands = {num_hands}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
      
        img_mask_bgr = cv2.cvtColor(mask,  cv2.COLOR_GRAY2BGR)
        #img_disp = cv2.vconcat([frame, img_mosaiced, img_mask_bgr])
        img_disp = img_mosaiced
        cv2.imshow("Anonymization", img_disp)
    
        if record_file is not None:
            out.write(img_mosaiced)   # write each frame
 
        k = cv2.waitKey(t_sleep)
        if k == -1:
           continue
        if k == ord('q'):
            break
        elif k == ord('p'):
            t_sleep = 30
        else:
            t_sleep = 0

    cap.release()                
    cv2.destroyAllWindows()
    if record_file is not None:
        out.release()
    
      

def build_handdataset(video_file, dataset_name= None):
   
    import json
    
    mosaicer= Mosaicer(mode = "hand")

    out_base_dir = dataset_name
    if out_base_dir:
        os.makedirs(out_base_dir, exist_ok = True)
        image_dir = os.path.join(out_base_dir, "images")
        os.makedirs(image_dir, exist_ok = True)
        image_train_dir = os.path.join(out_base_dir, "images", "train")
        os.makedirs(image_train_dir, exist_ok = True)
        image_val_dir = os.path.join(out_base_dir, "images", "val")
        os.makedirs(image_val_dir, exist_ok = True)
        label_dir = os.path.join(out_base_dir, "labels")
        os.makedirs(label_dir, exist_ok = True)
        label_train_dir = os.path.join(out_base_dir, "labels", "train")
        os.makedirs(label_train_dir, exist_ok = True)
        label_val_dir = os.path.join(out_base_dir, "labels",  "val")
        os.makedirs(label_val_dir, exist_ok = True)

    # 1.2 open test video 
    cap = cv2.VideoCapture(video_file)  # or use 0 for webcam
    if cap is None:
        print(f"cannot open {video_file}")
        return 
        
    # Get width and height
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = int(cap.get(cv2.CAP_PROP_FPS))
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"resolution(wxh)={width}x{height}@{fps}")

    # 1.3 recording
    if record_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')   # or 'mp4v' for .mp4
        out = cv2.VideoWriter(record_file, fourcc, fps, (width, height))  
   
    fn = 1
    t_sleep = 1 #30
    sampling_freq = fps//2  # every 0.5 sec  
    fcount_sampled = fcount//sampling_freq 
    fcount_train   = fcount_sampled*8//10  
    fcount_val     = fcount_sampled - fcount_train 
    print(f"Total Frames:{fcount}, sampling to {fcount_sampled}: train:{fcount_train}, val:{fcount_val}")  
    fcount_sampled = 0
    
    #with mp_holistic.Holistic(static_image_mode=True) as holistic:
    
    while True:
    
        fn +=1
        ret, frame = cap.read()
        if ret == False or frame is None: #   or fn > 24*5*60:
            break
            
        '''
        if fn < 24:  # skip 1 sec for this video @TODO make robust (ignore some wrong affine estimation case)
           continue
        '''
        #frame = cv2.resize(frame, (w,h))       
        

        # 2.  masking areas  
        mask, num_hands, hbboxes =  mosaicer.get_hand_face_mask(frame)   
        if False:
            '''  # TOO SLOW
            fg_mask = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            '''
            #img_mosaiced = mosaicer.mosaic_image(frame, mask > 0 , mosaic_scale=0.05)
            #cv2.putText(img_mosaiced, f"fn={fn}, {fn/fps:.1f}s", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            img_mask_bgr = cv2.cvtColor(mask,  cv2.COLOR_GRAY2BGR)
            img_disp = img_mosaiced
            #img_disp = cv2.vconcat([frame, img_mosaiced, img_mask_bgr])
        else:
            img_disp = frame.copy()
            # (x_min + x_max)/2, (y_min +y_max)/2, (x_max - x_min), (y_max - y_min)))
            for hbbox in hbboxes:
                tl = (int(hbbox[0] - hbbox[2]//2), int(hbbox[1]  -  hbbox[3]//2))
                br = (int(hbbox[0] + hbbox[2]//2), int(hbbox[1]  +  hbbox[3]//2))
                cv2.rectangle(img_disp, tl, br, (255, 255, 255), 2) 

        # @TODO tracking with hbboxes!!
        # 

        cv2.putText(img_disp, f"fn={fn}, hands = {num_hands}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Anonymization", img_disp)
    
        if out_base_dir and fn%sampling_freq == 0:
            sub_dir = "train" if fcount_sampled < fcount_train else "val" 
            print(f"sub_dir :{sub_dir} in  {fcount_sampled} / {fcount_train}")
            fcount_sampled +=1
            cv2.imwrite(os.path.join(image_dir, sub_dir, f"{dataset_name}_{fn:010d}.jpg"), frame)
            with open(os.path.join(label_dir, sub_dir, f"{dataset_name}_{fn:010d}.txt"), "w") as f:
                for hbbox in hbboxes:
                    # Correct: Use write() and manually add the newline character (\n)
                    f.write(f"0 {hbbox[0]/width:.6f} {hbbox[1]/height:.6f} {hbbox[2]/width:.6f} {hbbox[3]/height:.6f}\n")
                    #print(f"0 xc:{hbbox[0]:.0f} yc:{hbbox[1]:.0f} w:{hbbox[2]:.0f} h:{hbbox[3]:.0f}\n")
                    #print(f"0 {hbbox[0]/width:.6f} {hbbox[1]/height:.6f} {hbbox[2]/width:.6f} {hbbox[3]/height:.6f}\n")
    
        if record_file is not None:
            out.write(img_mosaiced)   # write each frame
 
        k = cv2.waitKey(t_sleep)
        if k == -1:
           continue
        if k == ord('q'):
            break
        elif k == ord('p'):
            t_sleep = 30
        else:
            t_sleep = 0

    cap.release()                
    cv2.destroyAllWindows()
    if record_file is not None:
        out.release()    
    
if __name__ == "__main__":
    
    import sys
    if len(sys.argv) <2:
        print(f"usage: python {sys.argv[0]} videofile")
        exit()
    video_file = sys.argv[1]
    record_file = None # "mosaiced_porker.mp4"
    #mosaic_vid(video_file, record_file)  
    build_handdataset(video_file, "gkl1")  

