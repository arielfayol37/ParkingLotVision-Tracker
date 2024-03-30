from keras import layers, models
import h5py
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
import matplotlib.pyplot as plt

def create_lightweight_branch(input_shape=(37, 37, 3)):
    """
    Create a lightweight branch for the Siamese network.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    
    return model

def get_model(input_shape=(37, 37, 3)):
    """
    Create a two-branch Siamese network for comparing two images.
    """
    # Define the inputs for each branch
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    # Lightweight branch model
    branch_model = create_lightweight_branch(input_shape)
    
    # Reuse the same branch model on both inputs
    processed_a = branch_model(input_a)
    processed_b = branch_model(input_b)
    
    # Combine the outputs from both branches
    concatenated = layers.concatenate([processed_a, processed_b])
    x = layers.Dense(64, activation='relu')(concatenated)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = models.Model(inputs=[input_a, input_b], outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    
    return model


class DataGenerator(Sequence):
    def __init__(self, file_path, indices, batch_size=32, is_validation=False):
        self.file_path = file_path
        self.indices = indices
        self.batch_size = batch_size
        self.is_validation = is_validation

    def __len__(self):
        # Compute the number of batches to produce
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Sorting the indices to comply with HDF5 requirements
        sorted_batch_indices = np.sort(batch_indices)
        with h5py.File(self.file_path, 'r') as f:
            if not self.is_validation:
                # Loading and processing data for training
                batch_x_a = np.array(f['data_x'][sorted_batch_indices, 0, :, :, :])
                batch_x_b = np.array(f['data_x'][sorted_batch_indices, 1, :, :, :])
                batch_y = np.array(f['data_y'][sorted_batch_indices])
                return [batch_x_a, batch_x_b], batch_y
            else:
                # Loading and processing data for validation
                batch_x_a = np.array(f['data_x'][sorted_batch_indices, 0, :, :, :])
                batch_x_b = np.array(f['data_x'][sorted_batch_indices, 1, :, :, :])
                return [batch_x_a, batch_x_b]

def train():
    best_save_filename = "cnn_reidentification_weights.h5"
    file_path = "cnn_training_data.hdf5"
    
    with h5py.File(file_path, 'r') as f: 
        data_X = f['data_x']
        data_Y = f['data_y']

        assert data_X.shape[0] == data_Y.shape[0], "X and Y shapes do not match."

        # Generate a list of indices from 0 to the length of your dataset
        indices = np.arange(data_X.shape[0])
        np.random.shuffle(indices)

        val_percent = 0.2
        num_val_samples = int(len(indices) * val_percent)
        train_indices = indices[:-num_val_samples]
        val_indices = indices[-num_val_samples:]

        # Creating the data generators for training and validation
        training_generator = DataGenerator(file_path, train_indices, is_validation=False)
        validation_generator = DataGenerator(file_path, val_indices, is_validation=False)

        # Creating the model
        model = get_model(input_shape=(37, 37, 3))
        model.summary()

        # ModelCheckpoint callback to save the weights
        checkpoint_callback = ModelCheckpoint(
            best_save_filename,          # Path where to save the model
            save_weights_only=True,     # Save only the weights
            monitor='val_loss',         # Monitor the validation loss
            mode='min',                 # The monitoring direction: 'min' means the less val_loss, the better
            save_best_only=True,        # Save only the best model
            verbose=1)                  # Log when models are being saved

        # Start training
        model.fit(training_generator, validation_data=validation_generator, epochs=100, callbacks=[checkpoint_callback])


def visualize_model():
    num_samples = 100  # Number of samples you want to visualize
    
    model = get_model(input_shape=(37, 37, 3))
    model.load_weights("cnn_reidentification_weights.h5")   
    indices = np.linspace(0, 905119, num_samples, dtype=int)
    with h5py.File("cnn_training_data.hdf5", 'r') as f:
        print('loading data...')
        data_X = f['data_x'][indices]
        data_Y = f['data_y'][indices]
        print(data_X.shape, data_Y.shape)
        print('done loading data...')
        assert data_X.shape[0] == data_Y.shape[0], "X and Y shapes do not match."
        
        # Select a random subset of indices for visualization
        # indices = np.random.choice(np.arange(data_X.shape[0]), size=num_samples, replace=False)

    for index in range(len(indices)):
        # Extract the pair of images
        image_pair = data_X[index]
        print(image_pair.shape)
        img_a, img_b = image_pair[0], image_pair[1]
        print(img_a.shape, img_b.shape)
        
        # Reshape if necessary or adjust dimensions as needed for your model
        img_a = np.expand_dims(img_a, axis=0)
        img_b = np.expand_dims(img_b, axis=0)
        
        # Predict similarity
        prediction = model.predict([img_a, img_b])
        
        # Plot the images side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_a[0])
        axs[0].set_title('Image A')
        axs[0].axis('off')
        
        axs[1].imshow(img_b[0])
        axs[1].set_title('Image B')
        axs[1].axis('off')
        
        # Show prediction
        plt.suptitle(f'PS, RS: {prediction[0][0]:.2f}, {data_Y[index]}')
        plt.show()

if __name__ == "__main__":

    visualize_model()

