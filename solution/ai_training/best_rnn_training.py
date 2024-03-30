from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, TimeDistributed, Dropout, Flatten
import h5py
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import cv2
import random
import copy 
import time 

CONTEXT_SIZE = 20
PREDICT_SIZE = 8
best_save_filename = 'car_trajectory_best_validation_final_weights.h5'

def load_training_data():

    # Open the HDF5 file in read mode
    with h5py.File('car_tracks_dataset_final_data.hdf5', 'r') as f:
        # Access a specific dataset
        data_X = f['data_x']
        
        # Read the dataset into a NumPy array
        data_Y = f['data_y']

        assert data_X.shape[0] == data_Y.shape[0], f"x and y shapes do not match.{data_X.shape[0]}, {data_X.shape[0]}"
        size = data_X.shape[0]
        val_percent = 0.2 
        data_X = np.array(data_X)
        data_Y = np.array(data_Y)

        # Generate a list of indices from 0 to the length of your dataset
        indices = np.arange(data_X.shape[0])

        # Shuffle the indices
        np.random.shuffle(indices)

        # Use the shuffled indices to reorder your data
        shuffled_X = data_X[indices]
        shuffled_Y = data_Y[indices]

        # Now you can split the shuffled data into training and validation sets
        val_percent = 0.2  # Assuming you're using 20% of the data as validation
        num_val_samples = int(len(shuffled_X) * val_percent)

        train_X = shuffled_X[:-num_val_samples]
        train_Y = shuffled_Y[:-num_val_samples]
        val_X = shuffled_X[-num_val_samples:]
        val_Y = shuffled_Y[-num_val_samples:]
        # Do whatever you need with the data
        print("Shape of the dataset X :", data_X.shape)
        # print("First few elements of the dataset x:", data_X[:5])
        print("Shape of the dataset Y :", data_Y.shape)
        # print("First few elements of the dataset y:", data_Y[:5])
        assert val_Y.shape[0] == val_X.shape[0]
        assert train_Y.shape[0] == train_X.shape[0]

        return train_X, train_Y, val_X, val_Y


def get_model():
    # Define the model
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(units=64, input_shape=(CONTEXT_SIZE, 4), return_sequences=False))

    model.add(Dense(units=64, activation='relu'))
    
    # Assuming PREDICT_SIZE is defined and represents the number of time steps you want to predict
    # and each time step has 4 features
    model.add(Dense(PREDICT_SIZE * 4))
    model.add(Reshape((PREDICT_SIZE, 4)))

    # Specify a smaller learning rate for the Adam optimizer
    optimizer = Adam(learning_rate=0.00005)

    # Compile the model with the customized optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Print the model summary
    model.summary()

    return model 
def train():

    model = get_model()

    # ModelCheckpoint callback to save the weights
    checkpoint_callback = ModelCheckpoint(
        best_save_filename,          # Path where to save the model
        save_weights_only=True,     # Save only the weights
        monitor='val_loss',         # Monitor the validation loss
        mode='min',                 # The monitoring direction: 'min' means the less val_loss, the better
        save_best_only=True,        # Save only the best model
        verbose=1)                  # Log when models are being saved

    train_X, train_Y, val_X, val_Y = load_training_data()
    validation_data = (val_X, val_Y)

    # Assuming train_X, train_Y, and validation_data are defined
    history = model.fit(train_X, train_Y, epochs=5000, batch_size=1024, validation_data=validation_data,
                        callbacks=[checkpoint_callback])
    plot_losses(history=history)

def plot_losses(history):
    # Extracting the loss and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))  # Optional: Adjusts the size of the plot
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def base():
    def baseline_predictor(input_sequences):
        # Extract the last item from each sequence
        last_items = input_sequences[:, -1, :]  # Shape: (M, 4)
        # Repeat the last item 4 times to form the baseline predictions
        predictions = np.repeat(last_items[:, np.newaxis, :], PREDICT_SIZE, axis=1)  # Shape: (M, 8, 4)
        return predictions
    train_X, train_Y, val_X, val_Y = load_training_data()
    # Assuming val_X and val_Y are your validation inputs and targets, respectively
    # Generate baseline predictions for val_X
    baseline_predictions = baseline_predictor(val_X)

    # Correctly flatten the arrays for MSE calculation.
    # Since both predictions and actual values are sequences of shape (M, 8, 4), where M is the number of samples,
    # 8 is the sequence length, and 4 is the number of features, they should be flattened to (M*8*4,)
    # to ensure each element corresponds between prediction and actual arrays.
    predictions_flat = baseline_predictions.reshape(-1)
    actual_flat = val_Y.reshape(-1)

    # Calculate MSE
    mse = mean_squared_error(actual_flat, predictions_flat)

    print(f'Mean Squared Error: {mse}')

    
def visually_test_model(filename):
    # Open the HDF5 file in read mode
    with h5py.File(filename + '.hdf5', 'r') as f:
        # Access a specific dataset
        data_X = f['data_x']
        
        # Read the dataset into a NumPy array
        data_Y = f['data_y']

        assert data_X.shape[0] == data_Y.shape[0], f"x and y shapes do not match.{data_X.shape[0]}, {data_X.shape[0]}"
        data_X = np.array(data_X)
        data_Y = np.array(data_Y)

    def draw_on_frame(new_frame, coords, color):
        for dp in coords:
            width = float(dp[2]) * int(dp[3])
            height = int(dp[3])
            x1, y1, x2, y2 = int(dp[0]), int(dp[1]), int(float(dp[0]) + width), int(dp[1]) + height
            cv2.rectangle(new_frame, (x1, y1), (x2, y2), color, 3)

            cv2.imshow("Frame", new_frame)
            key = cv2.waitKey(5)        

    model = get_model()
    model.load_weights(best_save_filename)
    model.load_weights('car_trajectory_best_validation_final_weights.h5')

    # load video frame
    filepath = filename + '_1.MP4' # Filepath to video
    cap = cv2.VideoCapture(filepath)
    ret, frame = cap.read()

    # get random data points, 
    # show predicted path on frame
    # vs real path
    num_samples = 500
    sample_skip = 20
    start_sample = 0
    samples_X = data_X[start_sample: :sample_skip]
    samples_Y = data_Y[start_sample: :sample_skip]
    samples_predict = model.predict(samples_X)
    print(samples_X.shape)
    real_path_color = (0, 100, 255)
    predict_path_color = (255, 50, 50)

    for i in range(num_samples):
        sample_X, sample_Y = samples_X[i], samples_Y[i]
        new_frame = copy.deepcopy(frame)
        car_color = tuple([random.randint(0, 255) for _ in range(3)])
        
        draw_on_frame(new_frame, sample_X, car_color)
        draw_on_frame(new_frame, samples_predict[i], predict_path_color)
        draw_on_frame(new_frame, sample_Y, real_path_color)

        time.sleep(1)

if __name__ == "__main__":
    pass
# base()
# train()
# visually_test_model("../data/Long_video_4/Long_video_4")