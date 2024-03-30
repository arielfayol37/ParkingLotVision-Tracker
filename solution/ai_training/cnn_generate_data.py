import tensorflow as tf
import numpy as np

import random
import csv
import h5py

import cv2


skips = 0
IMG_SIZE = (37, 37)

def scale_resize_image(image, img_size):
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tf.image.resize(image, img_size)
    return image.numpy()  # Convert back to numpy array if further numpy operations are needed

def extract_and_transform(frame, x, y, a, height, target_size=IMG_SIZE):
    width = int(float(a) * float(height))
    height = int(height)
    x, y = int(x), int(y)
    cropped_image = frame[y:y + height, x:x + width]
    # Check if cropped image is empty
    """
    if cropped_image.size == 0:
    print("Cropped image has zero size.")
    raise ValueError("Image has zero size")  # Return None or consider an alternative handling strategy  
    """
    resized_image = scale_resize_image(cropped_image, target_size)
    return resized_image



def read_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        return None  # Return None to signify failure
    return frame

def generate_true_datapoint(true_coords_array, fake_coords_array, cap, frame=None):
    global skips
    num_data_pts = 5
    data_pts_x = np.zeros((2 * num_data_pts, 2, *IMG_SIZE, 3))  # Assuming 3 color channels
    data_pts_y = np.zeros(2 * num_data_pts)
    
    # Randomly shuffle arrays to avoid repetitive selections
    shuffled_true_indices = np.random.permutation(len(true_coords_array))
    shuffled_fake_indices = np.random.permutation(len(fake_coords_array))
    
    for i in range(num_data_pts):
        true_index = shuffled_true_indices[i % len(true_coords_array)]  # Wrap around if i exceeds length
        fake_index = shuffled_fake_indices[i % len(fake_coords_array)]
        
        true_frame_num = int(float(true_coords_array[true_index][4]))
        fake_frame_num = int(float(fake_coords_array[fake_index][4]))
        if not(frame is None):
            frame_true = frame
            frame_fake = frame
        else:
            frame_true = read_frame(cap, true_frame_num)
            frame_fake = read_frame(cap, fake_frame_num)
        
        if frame_true is None or frame_fake is None:
            continue  # Skip this iteration if frames couldn't be read
        if int(true_coords_array[true_index][0]) <= 0 or int(true_coords_array[true_index][1])\
              <= 0 or int(fake_coords_array[fake_index][0]) <= 0 or int(fake_coords_array[fake_index][1]) <= 0:
            #print('negative position, skip')
            skips += 1
            continue
        # Extract and transform images
        img_true_1 = extract_and_transform(frame_true, *true_coords_array[true_index][:4])
        img_fake = extract_and_transform(frame_fake, *fake_coords_array[fake_index][:4])
        
        # For a positive pair, use the same true image twice
        data_pts_x[2 * i] = [img_true_1, img_true_1]
        data_pts_y[2 * i] = 1  # Positive label
        
        # For a negative pair, pair a true image with a fake one
        data_pts_x[2 * i + 1] = [img_true_1, img_fake]
        data_pts_y[2 * i + 1] = 0  # Negative label
    # print(data_pts_x)
    # print(data_pts_y)
    # sys.exit()
    return data_pts_x, data_pts_y


def get_cars_data(data_points_filepath):
    # Get data points
    # data_points_filepath = 'data\Long_video_4\Long_video_4_track.tsv' # with the tsv extension
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
    
    # print(coord_data[random.choice(list(coord_data.keys()))])
    
    return coord_data

def generate_data(filepath_list):
    block_size = 28
    small_stride = 3
    for filepath in filepath_list:
        X_train, Y_train = [], []
        print(f"processing {filepath}...")
        with open(filepath + '_1.MP4', 'rb') as f:
            # vr = VideoReader(f, ctx=cpu(0))
            cap = VideoReader(f, ctx=cpu(0))
        coord_data = get_cars_data(filepath + '_track.tsv')
        dict_keys = list(coord_data.keys())
        len_of_file = len(coord_data)
        print(f"number of cars data: {len_of_file}")
        counter = 0

        for key, value in coord_data.items():
            counter += 1

            if int((counter * 100)/len_of_file) % 10 == 0:
                print(f"file progress {int((counter * 100)/len_of_file)}%, counter, len of file = {counter}, {len_of_file}")
            count = len(value) // block_size
            
            for i in range(count):
                true_feed_array = value[block_size * i: block_size * (i + 1)]
                r = [random.randint(0, len(value)-1) for j in range(small_stride)]
                for idx in r:
                    true_feed_array.append(value[idx])
                # Now pick 31 random values for the fake array
                fake_feed_array = []

                for z in range(block_size + small_stride):
                    idx =  random.choice(dict_keys)
                    if int(idx) == int(key):
                        continue
                    fake_feed_array.append(random.choice(coord_data[idx]))
                tiny_x, tiny_y = generate_true_datapoint(true_feed_array, fake_feed_array, cap)
                X_train.append(tiny_x)
                Y_train.append(tiny_y)
        cap.release()        
        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.array(Y_train).flatten()

        with h5py.File("cnn_training_data_"+ filepath[-1] + "_.hdf5", 'w') as f: 
            dsetx = f.create_dataset("data_x", data = X_train)
            dsety = f.create_dataset("data_y", data = Y_train)
    
    print(X_train.shape, Y_train.shape)
    return None


def combine_data():
    # Step 1: Load all arrays
    arrays_X, arrays_Y = [], []
    for i in range(1, 5):  # Assuming your files are named '1.hdf5' through '4.hdf5'
        with h5py.File(f'cnn_training_data_{i}_.hdf5', 'r') as f:
            # Assuming each file contains one main dataset named 'data'
            # Adjust the dataset name if it's different
            arrays_X.append(f['data_x'][:])
            arrays_Y.append(f['data_y'][:])

    # Step 2: Stack the arrays
    # Replace 'vstack' with 'hstack' or 'dstack' if needed
    combined_array_X = np.vstack(arrays_X)
    combined_array_Y = np.concatenate(arrays_Y)

    # Step 3: Save the combined array to a new HDF5 file
    with h5py.File('cnn_training_data.hdf5', 'w') as f:
        f.create_dataset('data_x', data=combined_array_X)
        f.create_dataset('data_y', data=combined_array_Y)

    print("Combined array saved to 'cnn_training_data.hdf5'.", combined_array_Y.shape, combined_array_X.shape)


if __name__ == "__main__":
    filepaths = ["../data/Long_video_3/Long_video_3", "../data/Long_video_4/Long_video_4", "../data/Long_video_2/Long_video_2", "../data/Long_video_1/Long_video_1"]
    nothing = generate_data(filepaths)
    print(skips)