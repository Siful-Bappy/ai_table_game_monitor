#!/usr/bin/env python3
"""
TableLayout - polygon editing tool using OpenCV mouse & keyboard

Usage:
    python table_layout.py path/to/image.jpg
    python table_layout.py path/to/polygons.json

Controls (when editing an image):
    Left click : add a point (drawn red; points numbered 0,1,2...)
    e          : remove previous point (repeat to remove more)
    f          : finalize current point list as a polygon (saved to layout dict and drawn in white)
    q          : save layout dict to JSON and quit

When loading a .json file, the program will draw saved polygons and label each polygon center with its dictionary key.

JSON format produced/consumed:
    {
      "1": [[x1,y1],[x2,y2],...],
      "2": [[...]],
      ...
    }

Requires: OpenCV (cv2), numpy, json
"""

import sys
import os
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict


class TableLayout:

    def __init__(self, img: np.ndarray, window_name: str = "TableLayout", layout: Dict[str, List[List[int]]] = {}):
        self.orig = img.copy()
        self.img = img
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        self.window_name = window_name

        # layout: dict mapping string keys ("1","2",...) to list of [x,y]
        self.layout: Dict[str, List[List[int]]] = layout

        # current polygon points being edited (list of (x,y) tuples)
        self.current_pts: List[Tuple[int, int]] = []

        # store cached drawing of finalized polygons on a separate layer
        self.polygons_layer = self.orig.copy()

        # callback binding
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        #self.draw_polygons_from_dict(self.img, self.layout)
        self._redraw()

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # add point
            self.current_pts.append((int(x), int(y)))
            print(f"Added point {len(self.current_pts)-1}: ({x},{y})")
            self._redraw()

    def _draw_current_points(self, canvas: np.ndarray):
        # draw each current point in red with its index
        for idx, (x, y) in enumerate(self.current_pts):
            cv2.circle(canvas, (x, y), radius=4, color=(0, 0, 255), thickness=-1)  # red filled
            cv2.putText(canvas, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # if more than 1 point, draw lines between them (preview)
        if len(self.current_pts) >= 2:
            pts = np.array(self.current_pts, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 255), thickness=1)

    def _draw_polygons_layer(self):
        # draw all polygons from self.layout onto self.polygons_layer
        self.polygons_layer = self.orig.copy()
        for key, pts in self.layout.items():
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(self.polygons_layer, [arr], isClosed=True, color=(255, 255, 255), thickness=2)

            # also optionally fill a translucent polygon (commented out by default)
            # overlay = self.polygons_layer.copy()
            # cv2.fillPoly(overlay, [arr], color=(200,200,200))
            # cv2.addWeighted(overlay, 0.2, self.polygons_layer, 0.8, 0, self.polygons_layer)

    def _redraw(self):
        # compose the display image from polygons layer + current points
        self._draw_polygons_layer()
        self.img = self.polygons_layer.copy()
        self._draw_current_points(self.img)
        cv2.imshow(self.window_name, self.img)

    def undo_point(self):
        if self.current_pts:
            removed = self.current_pts.pop()
            print(f"Removed point {removed}")
            self._redraw()
        else:
            print("No points to remove.")

    def finalize_polygon(self):
        if not self.current_pts:
            print("No points to finalize.")
            return
        # create new key as next integer string (1-based)
        next_key = str(len(self.layout) + 1)
        # convert tuples to simple lists for JSON
        pts_list = [[int(x), int(y)] for (x, y) in self.current_pts]
        self.layout[next_key] = pts_list
        print(f"Finalized polygon {next_key} with {len(pts_list)} points.")
        # clear current pts after finalizing
        self.current_pts = []
        self._redraw()

    def save_json(self, path: str):
        # Save layout dictionary as JSON
        try:
            with open(path, 'w', encoding='utf-8') as f:
                layout_norm = self.normalize_layout(self.layout, self.img_width, self.img_height)                     
                json.dump(layout_norm, f, indent=2, ensure_ascii=False)
            print(f"Saved layout to {path}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")

    @staticmethod
    def draw_polygons_from_dict(img: np.ndarray, layout: Dict[str, List[List[int]]], showLabel = True, thickness = 1):
        
        #canvas = img.copy()
        canvas = img # for speed
        
        # draw polygons and put key at polygon center
        for key, pts in layout.items():
            if not pts:
                continue
            
            # 1. polygons     
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [arr], isClosed=True, color=(0, 255, 255), thickness=thickness)
            if not showLabel:
                 continue
                 
            # 2. label        
            try: # compute polygon centroid using moments or average
                M = cv2.moments(arr)
                if abs(M.get('m00', 0)) > 1e-5:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    # fallback to average of points
                    cx = int(np.mean(arr[:, 0]))
                    cy = int(np.mean(arr[:, 1]))
            except Exception:
                cx = int(np.mean(arr[:, 0]))
                cy = int(np.mean(arr[:, 1]))

            cv2.putText(canvas, str(key), (cx - 5*len(str(key)), cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), thickness, cv2.LINE_AA)
      
        return canvas
     
    @staticmethod
    def normalize_layout(layout, width, height):
     
        # convert to [0, 1]
        layout_norm = {}
        for k in layout.keys():
            ps_norm = []
            for p in layout[k]:
                ps_norm.append([p[0]/width, p[1]/height])       
            layout_norm[k] = ps_norm
            
        return layout_norm
          

    @staticmethod
    def denormalize_layout(layout_norm, width, height):
     
        # convert to [0-width, 0-height]
        layout = {}
        for k in layout_norm.keys():
            ps = []
            for p in layout_norm[k]:
                ps.append([min(round(p[0]*width), width -1), min(round(p[1]*height), height -1)])       
            layout[k] = ps    
            
        return layout


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in ['.jpg', '.jpeg', '.png']


def is_json_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == '.json'


def main(argv):
    if len(argv) < 2:
        print("Usage: python table_layout.py <image.jpg|image.png|layout.json>")
        return

    path = argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    if is_json_file(path) and len(argv) < 3:
        # load json and draw polygons onto a blank image or image if the json has reference?
        try:
            with open(path, 'r', encoding='utf-8') as f:
                layout = json.load(f)
        except Exception as e:
            print(f"Failed to load json: {e}")
            return
              
        canvas = np.ones((360, 640, 3), dtype=np.uint8)
        height, width = canvas.shape[:2]                    
        layout = TableLayout.denormalize_layout(layout, width, height)    
        out = TableLayout.draw_polygons_from_dict(canvas, layout)
        winname = f"TableLayout [View mode] - {os.path.basename(path)}"
        cv2.imshow(winname, out)
        print("Loaded polygons from JSON. Press any key or ESC to exit.")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 
            
    elif is_image_file(path):   
        canvas = cv2.imread(path)
        if canvas is None:
            print("Failed to load image.")
            return
            
        layout = {}        
        # save to JSON next to image with same base name
        base = os.path.splitext(path)[0]
        out = base + '.json'
 
    elif is_json_file(path) and len(argv) > 2:    
        try:
            with open(path, 'r', encoding='utf-8') as f:
                layout = json.load(f)
            out = path  
            import  shutil
            shutil.copy2(path, path + "_")
        except Exception as e:
            print(f"Failed to load json: {e}")
            return

        canvas = cv2.imread(argv[2])
        if canvas is None:
            print("Failed to load image.")
            return
    else:
        print("Unsupported file type. Provide .jpg/.png image or a .json layout file.")
        return 
   
    height, width = canvas.shape[:2]      
    layout = TableLayout.denormalize_layout(layout, width, height)  
    tl = TableLayout(canvas, window_name=f"TableLayout[Edit mode] - {os.path.basename(path)}", layout= layout)
            
    print("Instructions: left click to add points, 'e' undo last point, 'f' finalize polygon, 'q' save+quit")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('e'):
            tl.undo_point()
        elif key == ord('f'):
            tl.finalize_polygon()
        elif key == ord('q'):           
            tl.save_json(out)
            break
            # allow ESC to quit without saving
        elif key == 27:
            print('ESC pressed. Exiting without saving.')
            break
    
    cv2.destroyAllWindows()
              
    
'''
def temp(path):

    with open(path, 'r', encoding='utf-8') as f:
         layout = json.load(f)
    
    layout_norm = TableLayout.normalize_layout(layout, 640, 360)
    
    with open(path, 'w', encoding='utf-8') as f:
         json.dump(layout_norm, f, indent=2, ensure_ascii=False)
'''           


if __name__ == '__main__':

    main(sys.argv)
    #temp(sys.argv[1])
    
    
    

