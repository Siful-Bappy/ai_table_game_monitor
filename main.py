import os
import time
import cv2
import numpy as np
import logging
logging.basicConfig( level=logging.INFO,
       format='%(asctime)s-%(levelname)s-%(message)s'  # NO datefmt!
    )
logger = logging.getLogger("VPS")   
#logger.setLevel(logging.INFO) 

def decompose_affine(M):

    """Affine matrix to  angle, scale, translation
    
       Y = s*R X + t
       
       assuming center at (0,0)
       
       return angle in degree, scale, translation 
    
    """
    tx, ty = M[0,2], M[1,2]
    a, b = M[0,0], M[0,1]
    scale = np.sqrt(a*a + b*b)
    angle = np.degrees(np.arctan2(b, a))
    return angle, scale, (tx, ty)


def augment_image(img, center, rot, t,  scale = 1.00):

    ''' 
       Geometric transform for test image for alignment 
       center: rotation center, at present we should use (0,0)  
    
    '''
    h, w = img.shape[:2]
       
    M_gt = cv2.getRotationMatrix2D(center, rot, scale)
    M_gt[0,2] = t[0]
    M_gt[1,2] = t[1]
    
    
    img_aug = cv2.warpAffine(img, M_gt, (w, h), borderMode = cv2.BORDER_REPLICATE)
    logger.debug(f"[Ground truth] {M_gt} for center:{center}, angle:{rot}, scale:{scale}, t={t}")
    print(f"[Ground truth] center:{center}, angle:{rot}, scale:{scale}, t={t}")      
    return img_aug, M_gt
    

class TableTracker:
 
    ''' 3 functions:  
         
         - alignment 
         - FG extractor     
         - tracker  (YOLO + SORT) 

    '''

    def __init__(self, img_template, yolo_modelname, mog_bgsub, keep_align = False):
       
        from ultralytics import YOLO
        from sort.sort import Sort  # https://github.com/abewley/sort

        self._prepare_template(img_template)  # the first image file 
        if mog_bgsub:
            #self.back_sub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=25, detectShadows=False)
            self.back_sub = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=400.0, detectShadows=False)

            #self.back_sub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True)
            self.back_sub.apply(img_template, learningRate =1)
        self.T_change = 50 # 30 
        self.yolo_modelname = yolo_modelname
        if self.yolo_modelname:
            self.yolo = YOLO(yolo_modelname) # int8=True, # trained with VisDrone DET dataset, not pretrained YOLO11n
            self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # SORT 초기화
            self.id2class = {}
   
    def _prepare_template(self, template_bgr):
        self.template_bgr = template_bgr
        self.template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        self.template_edges = cv2.Canny(self.template_gray, 100, 200)
        

    def align_optflow_ecc(self, img, center = (0, 0), M_gt = None, debug = False):

        if M_gt is None:        
            M_gt = np.array( [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])            

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

  
        # 1. Optical flow 
        ts = time.time()
        flow = cv2.calcOpticalFlowFarneback(self.template_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        t_opticalflow = time.time() - ts
                
        # why filtering non-zero flow? What if the non-zero flows are for the outliers? 
        fx, fy = flow[...,0], flow[...,1]
        if False:
            mag = np.sqrt(fx**2 + fy**2)
            mask = mag > 0.5
        else:
            mask = self.template_edges > 0
        
        ys, xs = np.where(mask)
        pts1 = np.stack([xs, ys], axis=-1).astype(np.float32)
        
        if center[0] != 0:
            pts1[...,0] -= w/2
        pts2 = pts1 + flow[ys, xs].astype(np.float32)
        if center[1] != 0:
            pts2[...,1] -= h/2    
  
        # 2. estimate affine matrix 
        ts = time.time()                              
        #M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3,refineIters=10)
        M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3,refineIters=10)
        t_estimation1 = time.time() - ts
        if debug:
            angle, scale, t = decompose_affine(M)
   
        if False:
            # ECC alignment (translation refinement)
            M_ecc = np.eye(2,3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
            try:
                ts = time.time()   
                cc, M_ecc = cv2.findTransformECC(self.template_gray, gray, M.astype(np.float32), cv2.MOTION_AFFINE, criteria, None, 1)              
            except cv2.error as e:
                return
                                
        else: 
            M_ecc = M
            

        ts = time.time()   
        angle2, scale2, t2 = decompose_affine(M_ecc)      
        #M = cv2.getRotationMatrix2D(center, -angle2, 1/scale2)
        if True:
           logger.debug(f"[Estimated] {M} => inv => {M} angle:{angle2:.3f}, scale:{scale2:.5f}, t={t2}")
           logger.info(f"[Estimated] center = {center}, angle:{angle2:.3f}, scale:{scale2:.5f}, t={t2}")                      
           M_eye = np.dot(np.vstack([M_gt, [0,0,1]]), np.vstack([M, [0,0,1]]))
           logger.debug(f"check Indentity:{M_eye}")
           err = np.linalg.norm(M_eye- np.eye(M_eye.shape[0]), ord='fro')
           logger.info(f"check Indentity:{err}")

        #c = np.array(center)
        #M_warp = np.hstack([M_ecc[:, :2], (M_ecc[:, 2:3] + c.reshape(2,1) - M_ecc[:, :2] @ c.reshape(2,1))])
        #aligned = cv2.warpAffine(gray2, M_warp, (w, h))
        aligned = cv2.warpAffine(gray, M, (w, h), borderMode = cv2.BORDER_REPLICATE)    
        t_warp = time.time() - ts
        
        
        if True: # alignment performance measure             
            edges = cv2.Canny(gray, 100, 200)
    
            if False:
                #edges = cv2.dilate(cv2.Canny(self.template_gray, 100, 200), None, iterations=1)
                edges = cv2.dilate(cv2.Canny(aligned, 100, 200), None, iterations=1)
            iou = np.sum((self.template_edges > 0) & (edges > 0)) / np.sum(self.template_edges > 0)
            #iou = np.sum((self.template_edges > 0) & (edges > 0)) / np.sum((self.template_edges > 0) | (edges > 0))
            logger.debug("Line overlap ratio (IoU):", iou)

            #color_vis = cv2.merge([edges1, edges2, np.zeros_like(edges1)])
            #cv2.imshow("Edge Alignment (red=img1, green=aligned img2)", color_vis)
    
   
            # Convert to 3-channel for visualization
            '''
            aligned_color = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
            # Overlay edges in red
            color_vis = aligned_color.copy()
            color_vis[edges1 > 0] = (0, 0, 255)
            cv2.imshow("Template(red), Aligned(green))", color_vis)
            '''
       
        return aligned 
        
    def extract_fg(self, frame, update = False):
            
        if len(frame.shape) == 3 and frame.shape[2] == 3:  # RGB image 
            learningRate = 1 if update else 0
            fg_mask = self.back_sub.apply(frame, learningRate)
        else: # custom algorithm         
            #color_vis[cv2.absdiff(self.template_gray, aligned) > self.T_change] = (255, 0, 0)
            return cv2.absdiff(self.template_gray, frame) > self.T_change
        
        return fg_mask
               
        
    def detect(self, frame, debug = False):
         
        ''' detects  hands, chips, cards '''
        
        if self.yolo is None:
            return []
        
        ts = time.time()    
        results = self.yolo.predict(frame, conf=0.4, verbose=False)
        logger.debug(f"yolo:{(time.time() - ts)*1000:.3f}ms")
        dets_list = []
        result = results[0] # single frame 
        if result.boxes is not None:
            # YOLO 결과 bbox 추출 [x1, y1, x2, y2, score]
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                dets_list.append([x1, y1, x2, y2, conf, cls])  # bbox TL, BR
                
        return dets_list
        
                    
    def tracking(self, dets_list,  debug = False): # need_cls = True,         
        ''' tracking hands, chips, cards '''
        
        import numpy as np 
             
        # -----------------------------
        # 1. SORT 트래킹
        # -----------------------------
        if len(dets_list) > 0:
            dets = np.array(dets_list)
        else:
            dets = np.empty((0, 5))
        tracks = self.tracker.update(dets) # tracks = [x1, y1, x2, y2, track_id]
        if not need_cls:
            return tracks
            
        # -----------------------------
        # 2. Book keeping IDs and CLSs  
        # -----------------------------
        # 2.1 remove disapperared  
        current_track_ids = {int(track[4]) for track in tracks}
        logger.debug(f"current_track_ids : {current_track_ids }")
        ids_to_remove = set(self.id2class.keys()) - current_track_ids
        for track_id in ids_to_remove:
              logger.debug(f"- delete {track_id}/{self.id2class[track_id] }")
              del self.id2class[track_id] # 추적되지 않는 ID의 CLS 정보 삭제
        
        final_tracks = [] 
        iou_threshold = 0.9  # Set a high threshold for a confident match
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
           
            # cls info from detector!
            track_box = track[:4]
            track_id = track[4]
            matched_cls = None
            logger.debug(f"track id:{track_id}")
            # A. 딕셔너리에서 클래스 조회
            if track_id in self.id2class:
                matched_cls = self.id2class[track_id]
            # B. ID가 처음 생성된 경우 (New Track) 
            else:
                logger.debug(f"new object:{track_id}")
                best_cls = None
                best_iou = 0
                for det in dets_list:
                    det_box = det[:4]
                    det_cls = int(det[5]) # Assuming class is the 6th element and needs to be an integer
                    iou = self.calculate_iou(track_box, det_box)
                    logger.debug(f"iou:{iou}")
                    # If IoU is high, we've found the class for this track
                    #if iou > iou_threshold and iou > best_iou:
                    if iou > best_iou:
                        best_iou = iou
                        best_cls = det_cls
                        break # Found a match, move to the next track
                if best_cls is not None:
                    matched_cls = best_cls
                    self.id2class[track_id] = matched_cls # **⭐ 클래스 정보 저장 (핵심 수정)**    
                    logger.debug(f"+ add {track_id}/{matched_cls}")        
                else:
                    logger.debug("cannot find matched: track_id:{track_id}  best_iou={best_iou}")        

            # Add the track info along with the class
            if matched_cls is not None:
                # Format: [x1, y1, x2, y2, track_id, class_id]
                final_tracks.append(np.append(track, matched_cls))       
                 
        #return tracks
        return final_tracks
   
    @staticmethod
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
         
        
        
    def overlay_layout_edges(self, img):
         
        img[self.template_edges > 0] = (0, 0, 255)
                       
        #print(f"t opt:{t_opticalflow:.3f}, est1:{t_estimation1:.3f}, est2:{t_estimation2:.3f}, warp:{t_warp:.3f}, post:{t_post:.3f}")
        #print(f"t opt:{t_opticalflow:.3f}, est1:{t_estimation1:.3f}, warp:{t_warp:.3f}, post:{t_post:.3f}")                                
        return img
    
    def overlay_layout(self, img):
         
       pass # draw ....
       
       
          
    @staticmethod 
    def run(template_file, video_file, keep_align, viz = False, record_file = None):
   
        import json
        from tablelayout import TableLayout
    
        # 1. Configuration  
        f_skip  = 2
    
        #1.1 template    
        img_template = cv2.imread(template_file)
        height, width = img_template.shape[:2]
        #1.2 layout 
        try:
            layout_path = template_file.replace(".jpg", ".json") 
            with open(layout_path, 'r', encoding='utf-8') as f:
                layout = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load json: {e}")
            return
        layout = TableLayout.denormalize_layout(layout, width, height)
        layout_pixmap = np.zeros((height,width), np.int32) # enough labelrange   
        layout_id2label = {0: ""}  
        layout_label2id = {}  
        for i, label in enumerate(layout.keys()):
            id = i + 1
            layout_id2label[id] = label  
            layout_label2id[label] = id  
            coords = layout[label]
            pts = np.array(coords).reshape((1,-1,2))
            cv2.fillPoly(layout_pixmap, [pts], color=id) 
        '''
        cv2.imshow("label map",layout_pixmap.astype(np.uint8)*15) # just for checking 
        logger.info(layout_id2label)
        logger.info(layout_label2id)
        cv2.waitKey(0)
        cv2.waitKey(0)
        exit(0)
        '''
            
        # 1.3 create tracker           
        #yolo_modelname = 'yolo11n_tablegame.engine' # "yolo11n-obb"             
        yolo_modelname = 'yolo11n_tablegame.pt' # "yolo11n-obb"             
        mog_bgsub = False # True  
        tracker = TableTracker(img_template, yolo_modelname, mog_bgsub, False)

        # 1.4 open test video 
        cap = cv2.VideoCapture(video_file)  # or use 0 for webcam
        if cap is None:
            logger.warning(f"cannot open {video_file}")
            return 
        
        # Get width and height
        width_o  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_o = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps  = int(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"Original video resolution(wxh)={width_o}x{height_o}@{fps}")
        logger.info(f"Processing video resolution(wxh)={width}x{height}@{fps}")

        # 1.4 recording
        if record_file is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')   # or 'mp4v' for .mp4
            out = cv2.VideoWriter(record_file, fourcc, fps//f_skip, (width, height))  
   
        viz_colors = [(0,0,255),(0,255,0), (255,0,0)]
        viz_clsnames = ['hand', 'chip', 'card'] 
     
        # object size check and filtering 
        min_size_hand, max_size_hand = 20, 60 # in px
        min_size_chip, max_size_chip =  3, 20  # in px 
        min_height_card, max_height_card = 10, 33 # in px 
        statistics_card_len =[]
        # @TODO overlap checking and filtering using calculate_iou(box1, box2)
     
        #---------------------------------------------
        # 1.2 Card Classifier 
        #---------------------------------------------
        opt_classify_card = True 
        if opt_classify_card:
            import mobilenet_card_classifier as mobilenet_card
            import json
            # image size to classification
            # note: original card's aspect ratio is normally 9:16, but mobilnet's 1:1. 
            #       and Mobile net use 224x224 for training and 256x256
            width_card = 224
            height_card = 224 

            # label names 
            card_label_path = "card_labels.json"
            logger.info("[CARD CLS] loading card labels")
            with open(card_label_path, "r", encoding="utf-8") as f:
                if f is not None:
                    card_label_list = json.load(f)
                else:
                    logger.warning("[CARD CLS] cannot load card labels:{card_label_path}")

            # make directory for saving card data
            save_card = True
            if save_card:
               save_dir = os.path.join("save_cards", str(int(time.time())))
               for card_label in card_label_list:
                   os.makedirs(os.path.join(save_dir, card_label), exist_ok = True)

            # load classification model 
            # checkpoint_path = "mobilenet_card.pt" 
            checkpoint_path = "working/best.pt" 
            logger.info(f"[CARD CLS] loading card classifier model:{checkpoint_path}")
            card_classifier = mobilenet_card.load_model(checkpoint_path)
            card_preprocessor  = mobilenet_card.load_image_preprocessor()
            if card_classifier is None or card_preprocessor is None:
                logger.warning("cannot load model or transformer")
                return        
     
        # 2. Run 
        fn = 1
        t_sleep = 1 #30
        while True:
    
            fn +=1
            ret, frame_o = cap.read()   # High Resolution should be keept for CARD CLASSIFICATION
            if ret == False or frame_o is None:
                break
            
            if fn%f_skip:  # skip frame for testing for low fps 
                continue
            '''
            if fn < 24:  # skip 1 sec for this video @TODO make robust (ignore some wrong affine estimation case)
                continue
            '''
             
            frame = cv2.resize(frame_o, (width,height))       
        
            # 2.1. aligment 
            if keep_align:
                if True: # modify for test
                    frame, M_gt = augment_image(frame, (0,0), 5, (2.0, 5.0))
                else:
                    M_gt = np.array( [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            
                aligned = tracker.align_optflow_ecc(frame, (0,0), M_gt)
            else: 
                aligned  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            '''  
            # 2.2 fg extraction
             
            if mog_bgsub:
                update  = True  if 24 <= fn < 34  else False  # only use 1sec to 1.5 sec clean case 
                logger.debug(f"{fn}:{update}")
                fg_mask = tracker.extract_fg(frame, update)
            else:
                fg_mask =  tracker.extract_fg(aligned)   
                # TOO SLOW
                #fg_mask = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                #               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
             
            img_annotated = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
            img_annotated[fg_mask] = (255, 0, 0)  # BLUE
            '''
             
            detect_list = tracker.detect(frame, debug = False) 
            #logger.debug(f"num of detects:{len(detect_list)}")
            # maybe better to grouping into same class
            
            
            opt_tracking = False
            # 2.4. tracking 
            if opt_tracking:
                tracks = tracker.tracking(detect_list, debug = True)  # tracks = [x1, y1, x2, y2, track_id]
                if viz and tracks is not None:
                    for track in tracks:
                        x1, y1, x2, y2, track_id = track.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}/CLS:{tracker.id2class[track_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                
                cnts = [0,0,0]
                banker_cards = []   # cards detected in the banker zone
                player_cards = []   # cards detected in the player zone
                positions = [[],[],[]]
                for det in detect_list:
                    x1, y1, x2, y2, conf, cls = det   # TL, BR, conf, cls
                    x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)
                    cx, cy = (x1+x2)//2, (y1+y2)//2

                    # Resolve layout zone for every detection (needed for card routing, not just viz)
                    layout_label = "bg"
                    if 0 < cx < width and 0 < cy < height:
                        layout_label_id = layout_pixmap[cy, cx]
                        layout_label    = layout_id2label[layout_label_id]
                        if not layout_label:
                            layout_label = "bg"

                    if cls in [0, 1]:  # hand / chip
                        cnts[cls] +=1
                    elif cls == 2:
                        # filtering using size hints
                        if y2-y1 > x2-x1:  # portrait
                            if min_height_card <= (y2-y1) <= max_height_card:
                                cnts[cls] +=1
                                if opt_classify_card:
                                    img_card = cv2.resize(frame_o[2*y1:2*y2,2*x1:2*x2,:], (width_card,height_card))
                                    if "bank" in layout_label.lower():
                                        banker_cards.append(img_card)
                                    else:
                                        player_cards.append(img_card)
                            else:
                                pass # ignore it

                        else:  # landscape (3rd card) -> rotate
                            if min_height_card <= (x2-x1) <= max_height_card:
                                cnts[cls] +=1
                                if opt_classify_card:
                                    img_card = cv2.resize(cv2.rotate(frame_o[2*y1:2*y2,2*x1:2*x2,:], cv2.ROTATE_90_CLOCKWISE), (width_card,height_card))
                                    if "bank" in layout_label.lower():
                                        banker_cards.append(img_card)
                                    else:
                                        player_cards.append(img_card)
                            else:
                                pass # ignore it

                    if viz:  # draw bbox after routing
                        cv2.rectangle(frame, (x1, y1), (x2, y2), viz_colors[cls], 2)
                        positions[cls].append(layout_label)

                
                logger.info(f"fn={fn}, {cnts[0]} hands, {','.join(positions[0])}")
                logger.info(f"         {cnts[1]} chips, {','.join(positions[1])}")
                logger.info(f"         {cnts[2]} cards, {','.join(positions[2])}")
                if viz:  
                    cv2.putText(frame, f"{cnts[0]} hands:, {cnts[1]} chips, {cnts[2]} cards", (20, height - 20*4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f" - hands: {','.join(positions[0])}", (20, height - 20*3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f" - chips: {','.join(positions[1])}", (20, height - 20*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f" - cards: {','.join(positions[2])}", (20, height - 20*1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # 2.3. Layout overlay  
            if keep_align:                        
                tracker.overlay_layout_edge(frame) #img_annotated)
            else:
                frame = TableLayout.draw_polygons_from_dict(frame, layout, showLabel = False) 
                tracker.overlay_layout(frame)
                #img_annotated = TableLayout.draw_polygons_from_dict(frame, layout) # img_annotated, layout) 
                #tracker.overlay_layout(img_annotated)
            
            # 2.5. key game event detections
            # @TODO
        
        
            # 2.6. detail info: card classification
            if opt_classify_card:
                all_cards = banker_cards + player_cards
                if all_cards:
                    pred_ids, probs = mobilenet_card.classify_cards(
                        card_classifier, card_preprocessor, all_cards)

                    # Save every card crop to its predicted class folder
                    if save_card:
                        for i, img_c in enumerate(all_cards):
                            lbl = card_label_list[pred_ids[i]]
                            cv2.imwrite(os.path.join(save_dir, lbl, f"{fn}_{i}.jpg"), img_c)

                    if viz:
                        THUMB_W, THUMB_H = width_card // 2, height_card // 2

                        LABEL_H = 20   # height of the label bar below each card

                        def _make_strip(cards, id_offset):
                            """Full card thumbnail + label bar below (no overlap)."""
                            thumbs = []
                            for j, img_c in enumerate(cards):
                                th  = cv2.resize(img_c, (THUMB_W, THUMB_H))
                                lbl = card_label_list[pred_ids[id_offset + j]]
                                # Separate black bar below the card image
                                bar = np.zeros((LABEL_H, THUMB_W, 3), dtype=np.uint8)
                                cv2.putText(bar, lbl[:12], (3, LABEL_H - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                                thumbs.append(cv2.vconcat([th, bar]))
                            return cv2.hconcat(thumbs)

                        # Banker cards → top-LEFT corner
                        if banker_cards:
                            strip     = _make_strip(banker_cards, 0)
                            sh, sw    = strip.shape[:2]
                            sw        = min(sw, width // 2)
                            frame[0:sh, 0:sw] = strip[:, :sw]
                            cv2.putText(frame, "BANKER", (4, sh + 14),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        # Player cards → top-RIGHT corner
                        if player_cards:
                            strip  = _make_strip(player_cards, len(banker_cards))
                            sh, sw = strip.shape[:2]
                            sw     = min(sw, width // 2)
                            frame[0:sh, width - sw:width] = strip[:, :sw]
                            (tw, _), _ = cv2.getTextSize(
                                "PLAYER", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.putText(frame, "PLAYER", (width - tw - 4, sh + 14),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                         
            # 2.7 display  
            if viz:
                # Frame number + timestamp — centered at the top
                hud = f"fn={fn}  {fn/fps:.1f}s"
                (tw, _), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(frame, hud, ((width - tw) // 2, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("layout(yello), FG(blue)", frame)
    
            if record_file is not None:
                out.write(frame)   # write each frame
                
            '''
            cv2.putText(img_annotated, f"fn={fn}, {fn/fps:.1f}s", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            #img_disp = cv2.vconcat([frame, img_annotated])
            if mog_bgsub:
                img_disp = cv2.vconcat([frame, img_annotated, cv2.cvtColor(fg_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)])
            else:
                img_disp = cv2.vconcat([frame, img_annotated])
                #img_disp = cv2.vconcat([frame, img_annotated, cv2.cvtColor(fg_mask.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)])
              
            cv2.imshow("layout(yello), FG(blue)", img_disp)
    
            if record_file is not None:
                out.write(img_annotated)   # write each frame
            '''
 
            if viz: 
                k = cv2.waitKey(t_sleep)
                if k == ord('q'):
                    break
                elif k == ord('p'):
                    t_sleep = 1000//fps
                elif k == ord('s'):
                    t_sleep = 0
        if cap:
            cap.release()                
        cv2.destroyAllWindows()
        if record_file is not None:
            out.release()  
    
         
        '''
        import matplotlib.pyplot as plt
        plt.hist(statistics_card_len, bins=100, edgecolor='black', alpha=0.7)
        plt.show()
        '''
        

def test(img_file):
   
    img = cv2.imread(img_file)
    h, w = img.shape[:2]
    tracker = TableTracker(img)
 
    #center = (w/2, h/2)  : NOK
    center = (0, 0)
    img2, M_gt = augment_image(img, center, 5, (2.0, 5.0))
   
    aligned = tracker.align_optflow_ecc(img, (0,0))
    img_annotated = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    
    # FG
    gray_fg =  tracker.extract_fg(aligned)   
    img_annotated[gray_fg] = (255, 0, 0)  # BLUE
    
    # layout  
    tracker.overlay_layout(img_annotated)
        
    cv2.imshow("template(red), change(blue)", img_annotated)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="AI Table Game Monitor — inference on baccarat video.")
    parser.add_argument("template",  help="Layout image file  (e.g. gkl_table1_layout.jpg)")
    parser.add_argument("video",     help="Input video file   (e.g. gkl_mask1.avi)")
    parser.add_argument("--save",    action="store_true",
                        help="Save the annotated inference video to disk")
    parser.add_argument("--output",  default="baccarat_annotated.mp4",
                        help="Output video filename  (default: baccarat_annotated.mp4, only used with --save)")
    parser.add_argument("--no-viz",  action="store_true",
                        help="Disable on-screen display (useful when running headless with --save)")
    args = parser.parse_args()

    record_file = args.output if args.save else None
    viz         = not args.no_viz
    keep_align  = False

    if record_file:
        logger.info(f"Inference video will be saved → {record_file}")
    else:
        logger.info("Video saving disabled  (pass --save to enable)")

    TableTracker.run(args.template, args.video, keep_align, viz, record_file)




