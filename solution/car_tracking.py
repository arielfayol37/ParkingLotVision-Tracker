import os
import random

import cv2
from inference import get_roboflow_model
import time

from my_tracker import Tracker
import copy
import sys

# print("HELOOOOOO")

"""
Algorithm to track cars using a RNN and a CNN by @ Fayol Ateufack Zeudom

 - The RNN takes the 20 previous (x, y, a, h) := x-x-coordinate y-y-coordinate a-aspect-ratio h-height =: 
    of a car and predicts the 8 next (x, y, a, h)
 
 - The CNN takes an input of shape (1, 2, 37, 37, 3) where shape[1][0] and shape[1][1] are two car images
    of shape 37 by 37 each with 3 color channels (r, g, b) that have been normalized

 - We will have region map of shape (N, M) where N and M are the dimensions of the video
   This region map will contain the predicted positions of cars in the current frame

    
The algorithm:

    for each car in the list of tracked_cars:
        predict the position of car (using the RNN) and update the 
        region map such that region_map[x:x+width, y:y+height] = car_id, where x, y, width, and height
        are the bounding box information for that car
        add car to unmatched cars (for the next frame)
    
    On the next frame, use a car detector to identify cars in the image
    
    for each car identified, check what percentage of it is covering a predicted car position from the region map
        if percentage more 75%, then it is the car that that was predicted at that position.
            remove car from unmatched cars
            update its bounding boxes list
        elif greater than 30%
            give the ID to whichever has the highest matching score by the CNN
            remove car from unidentified cars    
        else: 
            stack car with bounding box image to unidentified cars
        
    for car in unidentified cars:
        if unmatched cars is not empty:
            if car is not in starting region:
                
    
"""

colors = [[random.randint(0, 255) for _ in range(3)]
        for _ in range(10)]

def label_car_on_frame(frame, track_id, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)
    label = track_id
    #Create a rectangle above the detected object and add label and confidence score
    t_size=cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)[0]
    c2=x1+t_size[0], y1-t_size[1]-3
    cv2.rectangle(frame, (x1, y1), c2, color=(85, 45, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(frame, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return frame

def run_tracking(filepaths):
    models = {'0':'center_of_intersection/5', '1':'outside_the_intersection/3', '2':'vaid-jy5xo/1'}
    model = get_roboflow_model(model_id=models['0'], api_key="UWNPpMrGPun13aPF9MLJ")
    print(filepaths)
    for filepath in filepaths:
        print(filepath)
        st = time.perf_counter()
        extension = 'mp4'
        base_path = os.path.join('.', 'data', filepath) 
        path_counter = 1
        # specific_filepath = os.path.join(base_path, f'{filepath}_{path_counter}')
        video_path = os.path.join(base_path, f'{filepath}_{path_counter}.{extension}')
        print(video_path)

        while os.path.exists(video_path): 
            video_out_path = os.path.join(base_path, f'{filepath}_{path_counter}_with_rnn_v1.{extension}')
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            frame_out = copy.deepcopy(frame)

            nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                                    (frame.shape[1], frame.shape[0]))
            
            
            print(f'There are {nframes} in video {video_path}')

            
            
            
            height, width = frame.shape[:2]
            tracker = Tracker(width, height)
            detection_threshold = 0.3 
            counter = 0
            while ret:
                # print("frame #", counter)
                percent = int((counter / nframes) * 100)
                if percent % 10 == 0:
                    print(f"{percent:.2g}% progress for {video_out_path}")
                counter += 1 
                result = model.infer(frame, confidence=detection_threshold, overlap=0)
                detections = []
                for r in result[0].predictions:
                    x = int(r.x -r.width / 2)
                    y = int(r.y -r.height / 2)
                    h = int(r.height)
                    w = int(r.width)
                    
                    detections.append([x, y, w, h])
                
                
                
                tracker.update(detections, frame)
                for track_id, car_dict in tracker.cars_info.items():
                    if car_dict["time_elapsed"] == 0:
                        x1, y1, a, h1 = car_dict["coords"][-1]
                        x2 = int(x1 + (a * h1))
                        y2 = y1 + h1
                        frame_out = label_car_on_frame(frame_out, track_id, x1, y1, x2, y2)                  
               
              
                
                """
                for det in detections:
                    frame_out = label_car_on_frame(frame_out, 0, det[0], det[1], det[0] + det[2], det[1] + det[3])                
                """

                


                cap_out.write(frame_out)
                ret, frame = cap.read()
                frame_out = copy.deepcopy(frame)
            cap.release()
            cap_out.release()
            et = time.perf_counter()
            total_time_mins = (et - st)/60
            print(f"took {total_time_mins} for {video_out_path}")
            path_counter += 1  
            video_path = os.path.join(base_path, f'{filepath}_{path_counter}.{extension}') # has to be after video_path
            # specific_filepath = os.path.join(base_path, f'{filepath}_{path_counter}')


if __name__ == '__main__':
    filepaths = ["valpo", "video_3", "Long_video_4", "Long_video_3", "Long_video_2", "Long_video_1"]
    # filepaths = ["valpo"]
    run_tracking(filepaths)