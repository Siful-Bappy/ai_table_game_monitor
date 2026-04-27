'''

   video to images for the list of time slots
   
   
'''

import os
import cv2


def is_in_range(fn, fps, ranges):

    for s, e in ranges:
        if fps*s <= fn <=  fps*e:
            return True
      
    return False        

def video2imgs(video_file, ranges, save_dir):

    cap = cv2.VideoCapture(video_file)  # or use 0 for webcam
    if cap is None:
        print(f"cannot open {video_file}")
        return 
        
    # Get width and height
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"resolution(wxh)={width}x{height}@{fps}")

    os.makedirs(save_dir, exist_ok = True) 

    scale = 2
    fn = 0
    
    state = 'saving'
    
    while True:
        fn +=1
        #if fn % fps == 0:
        #    print(f"t={fn/fps}sec")
            
        ret, frame = cap.read()
        if ret == False or frame is None:
            break
            
        if is_in_range(fn, fps, ranges):
             new_state = 'saving'
             frame = cv2.resize(frame, (width//scale, height//scale))
             cv2.imwrite(os.path.join(save_dir, f"img_{fn:09}.jpg"), frame)
        else:
             new_state = 'not saving'
        
        if state != new_state:
            print(f"at {fn/fps}sec, State: {state} to {new_state}")
            state = new_state
                      
    cap.release()    
    
def demo1():

    video_file  = os.path.join("downloads", "h.mkv")
    ranges = [[30.0, 33.0],[2*60 + 23, 2*60 + 33]]
    save_dir = "dataset_h"
    
    video2imgs(video_file, ranges, save_dir)
    
def demo2():

    video_file  = "2025_10_02 15_36.mp4"
    ranges = [[1.0, 27.0]] # sec
    save_dir = "dataset"
    
    video2imgs(video_file, ranges, save_dir)

def demo3():

    video_file  = "baccarat_annotated.mp4"
    ranges = [[.0, 26.0]] # sec
    save_dir = "dataset_2"
    
    video2imgs(video_file, ranges, save_dir)

         
if __name__ == "__main__":
    
    demo3()
        
    
