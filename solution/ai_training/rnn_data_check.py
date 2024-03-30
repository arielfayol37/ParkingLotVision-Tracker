import cv2, csv
import copy
import sys, random
# Get data points
data_points_filepath = '..\data\Long_video_4\Long_video_4_track.tsv' # with the tsv extension
coord_data = {}
with open(data_points_filepath, 'r', newline='') as coord_file:
    
    reader = csv.reader(coord_file, delimiter='\t')
    
    for row in reader:
        id = int(row[0])
        val = row[1:]
        # print(row)
        # sys.exit()
        slice_size = 5
        # print(len(val))
        num_strides = int(len(val) // slice_size)
        assert num_strides == len(val) / slice_size
        coord_data[id] = []
        
        for k in range(num_strides):
            coord_data[id].append(val[k * slice_size: (k + 1) * slice_size])
        # print(coord_data[id])
        # sys.exit()

filepath = '..\data\Long_video_4\Long_video_4_1.MP4' # Filepath to video
cap = cv2.VideoCapture(filepath)
ret, frame = cap.read()
sorted_data = sorted(coord_data.items())
start_id = 91
for car_id, dps in sorted_data:
    if int(car_id) > start_id:
        new_frame = copy.deepcopy(frame)
        print(car_id)
        car_color = tuple([random.randint(0, 255) for _ in range(3)])
        # print(dps)
        for dp in dps:
            width = float(dp[2]) * int(dp[3])
            height = int(dp[3])
            x1, y1, x2, y2 = int(dp[0]), int(dp[1]), int(float(dp[0]) + width), int(dp[1]) + height
            cv2.rectangle(new_frame, (x1, y1), (x2, y2), car_color, 3)

            cv2.imshow("Frame", new_frame)
            key = cv2.waitKey(1)
            if key == 9:
                break
