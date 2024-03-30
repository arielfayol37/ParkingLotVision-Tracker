import os, csv
import h5py
import numpy as np
import sys

"""

id, x, y, a, h, frame_count, x, y, a, h, frame_count, ..., x, y, a, h, frame_count

"""
CONTEXT_SIZE = 20
PREDICT_SIZE = 8

def write_car_tracks(filenames):
    # filename_ID.tsv contains the IDs
    # filename_1.tsv contains the coordinates
    # Reading the TSV file

    track_arrays = []

    for filename in filenames:
        print(f'processing file {filename}')
        id_filepath = filename + '_ID.tsv'
        big_array = []
        fake_big_array = []
        # Read coordinates into memory
        with open(filename + '_1.tsv', 'r', newline='') as coord_file:
            coord_data = {int(row[0]): row[1:] for row in csv.reader(coord_file, delimiter='\t')}

        with open(id_filepath, 'r', newline='') as tsvfile:
            id_reader = csv.reader(tsvfile, delimiter='\t')
            for ids in id_reader:
                # example of ids = [1, 200, 743]
                car_data = {}
                ids = [j for j in ids if j.strip()]
                if len(ids) != 1:
                    print(f"{ids}")
                    continue
                ids = ids[0].split(', ')
                for id in ids:
                    try:
                        id = int(float(id)) 
                    except:
                        continue
                    if id in coord_data:
                        car_data[id] = []
                        val = coord_data[id]
                        # print(val)
                        slice_size = 5 # (x, y, a, h, frame_count)
                        num_strides = int(len(val) // slice_size)
                        assert num_strides == len(val) / slice_size
                        for k in range(num_strides):
                            car_data[id].append(val[slice_size * k : slice_size * (k + 1)])
                # print(car_data)
                if len(car_data) == 0:
                    print('did not find ids', ids)
                else:
                    big_array, fake_big_array = get_arrays(car_data, big_array, fake_big_array)
        
        with open(filename + '_track.tsv', 'w', newline='') as savefile:
            writer = csv.writer(savefile, delimiter='\t')
            writer.writerows(fake_big_array)
        l = create_datasets(big_array, filename)
        track_arrays.extend(big_array)
    s = create_datasets(track_arrays, 'car_tracks_dataset_final_data')

def create_datasets(track_arrays, save_filename):
    array_data_x = []
    array_data_y = []
    sweep_start = PREDICT_SIZE - CONTEXT_SIZE
    for track_array in track_arrays:
        reversed_array = track_array[::-1]
        arrays = [track_array, reversed_array]
        for array in arrays:
            counter = 1
            for index in range(len(array) - (PREDICT_SIZE + CONTEXT_SIZE)):
                x = array[index: index + CONTEXT_SIZE]
                if len(x) != CONTEXT_SIZE:
                    counter += 1
                    print(f"x {len(x)}\n, {CONTEXT_SIZE}\n, {counter}\n, {y}\n, {x}\n, {array}\n")
                else:
                    array_data_x.append(x) # 0: 16, 1: 17, 2, 18
                
                y = array[index + CONTEXT_SIZE: index + CONTEXT_SIZE + PREDICT_SIZE]

                if len(y) != PREDICT_SIZE:
                    counter += 1
                    print(f"y {len(y)}\n, {PREDICT_SIZE}\n, {counter}\n, {y}\n, {x}\n, {array}\n")
                else:
                    array_data_y.append(y) # 16: 20, 17: 21, 18: 22 
            
            while sweep_start < 0:
                instance_array = [[0, 0, 0, 0] for _ in range(abs(sweep_start))]
                instance_array.extend(array[:CONTEXT_SIZE + sweep_start]) # : 4, : 5
                if len(instance_array) != CONTEXT_SIZE:
                    counter += 1
                    print(f"{len(instance_array)}\n, {CONTEXT_SIZE}\n, {instance_array}\n, {array}\n")
                else:
                    array_data_x.append(instance_array)
                i_y = array[CONTEXT_SIZE + sweep_start:CONTEXT_SIZE + sweep_start + PREDICT_SIZE] # 4: 8
                if len(i_y) != PREDICT_SIZE:
                    print(f"{len(i_y)}\n, {PREDICT_SIZE}\n, {i_y}\n, {array}\n")
                else:
                    array_data_y.append(i_y)
                sweep_start += 1
            
            sweep_start = PREDICT_SIZE - CONTEXT_SIZE
    
    assert len(array_data_x) == len(array_data_y)
    array_data_x = np.array(array_data_x)
    array_data_y = np.array(array_data_y)
    
    print(array_data_x.shape, array_data_y.shape)
        # creating a file
    with h5py.File(save_filename + ".hdf5", 'w') as f: 
        dsetx = f.create_dataset("data_x", data = array_data_x)
        dsety = f.create_dataset("data_y", data = array_data_y)
    
    return True


def get_arrays(car_data, big_array, fake_big_array):
    # print('getting arrays')
    last_frame_counts = []
    first_frame_counts = []
    # print(car_data)
    keys = []
    for key, val in car_data.items():
        keys.append(key)
        last_frame_counts.append(int(val[-1][4]))
        first_frame_counts.append(int(val[0][4]))
    # print(first_frame_counts, last_frame_counts)
    max_lfc = max(last_frame_counts)
    search_frame_count = min(first_frame_counts) - 1
    
    final_array = []
    fake_final_array = []
    # print(car_data)
    prev_coord = None
    while True:
        found = False
        search_frame_count += 1
        # print(search_frame_count)
        if search_frame_count > max_lfc:
            break
        for key, val in car_data.items():
            
            for idx, coord in enumerate(val):
                # print(coord)
                if str(search_frame_count) == coord[4]:
                    x, y = int(coord[0]), int(coord[1])
                    a, h = (eval(coord[2])), int(coord[3])
                    if prev_coord and (x-prev_coord[0])**2 + (y-prev_coord[1])**2 >= 10000:
                        print('Too much of a distance', prev_coord, x, y)
                        break
                    prev_coord = [x, y, a, h]
                    final_array.append([x, y, a, h])
                    fake_final_array.extend([x, y, a, h, int(coord[4])])
                    # print(fake_final_array)
                    found = True
                    break
            if found:
                break
        if not found:
            if len(final_array) >= CONTEXT_SIZE + PREDICT_SIZE:
                big_array.append(final_array)
            final_array = []
            prev_coord = None
    if final_array and len(final_array) >= CONTEXT_SIZE + PREDICT_SIZE:
        big_array.append(final_array)
    if fake_final_array:
        fake_final_array.insert(0, keys[0])
        fake_big_array.append(fake_final_array)

    return big_array, fake_big_array


write_car_tracks(["../data/Long_video_3/Long_video_3", "../data/Long_video_4/Long_video_4", "../data/Long_video_2/Long_video_2", "../data/Long_video_1/Long_video_1"])
