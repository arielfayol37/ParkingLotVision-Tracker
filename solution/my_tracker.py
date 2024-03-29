import tensorflow as tf
import numpy as np
from ai_training.best_rnn_training import get_model as rnn_get_model
from ai_training.best_cnn_training import get_model as cnn_get_model

rnn_model = rnn_get_model()
cnn_model = cnn_get_model()

rnn_model.load_weights("ai_weights/car_trajectory_best_validation_final_weights.h5")
cnn_model.load_weights("ai_weights/cnn_reidentification_weights.h5")

context_size = 20
predict_size = 8


IMG_SIZE = (37, 37)

def scale_resize_image(image, img_size=IMG_SIZE):
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tf.image.resize(image, img_size)
    return image.numpy()  # Convert back to numpy array if further numpy operations are needed

"""
    time_elapsed: time (num of frames) since last prediction
"""
class Tracker:
    def __init__(self, W, H):
        self.next_id = 1
        self.cars_info = {}
        self.W, self.H = W, H # Video width, video height
        self.region_map = np.zeros((W, H))
        self.unmatched_cars = []
        self.unidentified_cars = []
        self.exited_cars = {}
        self.current_frame = None # this is a processed frame (numpy array)
        self.frame_count = -1

    def add_car(self, bounding_box):
        coord_pts = self.get_coord(bounding_box)
        last_seen_image = self.get_image(bounding_box)
        self.cars_info[self.next_id] = {"coords":[coord_pts for _ in range(context_size)], "time_elapsed":0, \
                                        "last_seen":last_seen_image, "first_bb": bounding_box, "last_bb":bounding_box,\
                                            "predicted":[coord_pts for _ in range(predict_size)]}
        self.update_region_map(bounding_box, self.next_id)
        self.next_id += 1
        

    def update_car(self, car_id, bounding_box=None):
        """
        Takes a car id and a bounding_box(optional)
        - if bounding_box, then the car has been identified in the image
        - Update the car's position coordinates and predictions
        - Update the region_map
        """
        
        self.cars_info[car_id]["coords"].pop(0)
        
        if bounding_box: # if car has been matched
            self.unmatched_cars.remove(car_id)
            coord_pts = self.get_coord(bounding_box)
            last_seen_image = self.get_image(bounding_box)
            self.cars_info[car_id]["last_bb"] = bounding_box
            if self.cars_info[car_id]["coords"][-1] == [0, 0, 0, 0]: # this happens when it loses the car for more than 8 frames
                self.cars_info[car_id]["coords"] = [coord_pts for i in range(context_size)]
                self.cars_info[car_id]["predicted"] = [coord_pts for i in range(predict_size)]
            else:
                self.cars_info[car_id]["coords"].append(coord_pts)
                # TODO: this can be optimized by not having to predict at every timestep
                self.predict_car_position(car_id)
            
            pred_0 = self.cars_info[car_id]["predicted"][0]
            self.update_region_map(self.get_bounding_box(pred_0), car_id)
            
            self.cars_info[car_id]["time_elapsed"] = 0
            self.cars_info[car_id]["last_seen"] = last_seen_image
            
            
        else: # in case car has not been matched
            time_elapsed = self.cars_info[car_id]["time_elapsed"]
            if  time_elapsed >= predict_size:
                self.cars_info[car_id]["coords"].append([0, 0, 0, 0])
                # Wanted to clear the predicted positions but
                # those (the last one) might be used by the CNN for later comparisons
            else:
                p = self.cars_info[car_id]["predicted"][time_elapsed]
                self.cars_info[car_id]["coords"].append(p)
            
            self.cars_info[car_id]["time_elapsed"] += 1

    def predict_car_position(self, car_id):
        prev_20_steps = np.array(self.cars_info[car_id]["coords"])
        preds = rnn_model(prev_20_steps).tolist()
        self.cars_info[car_id]["predicted"] = preds
        
    
    def get_car_id_with_highest_matching_score(self, car_ids, car_image):
        """
        Given a list of car_ids and an image, return the id with the closest match
        or None if zero match for each.
        """
        # Assuming self.cars_info[car_id]["last_seen"] is a numpy array of shape (37, 37, 3)
        # and car_image is also of shape (37, 37, 3)
        car_image = scale_resize_image(car_image)
        cars_images = np.array([
            np.stack([scale_resize_image(self.cars_info[car_id]["last_seen"]), car_image])
            for car_id in car_ids
        ])
        
        # Correct shape assertion for debugging
        assert cars_images.shape == (len(car_ids), 2, 37, 37, 3), f"Expected shape {(len(car_ids), 2, 37, 37, 3)}, but got {cars_images.shape}"
        
        # Reshape to fit the model's expected input shape if necessary
        cars_images = cars_images.reshape(-1, 37, 37, 3)  # Only if your model expects all images at once without the pair dimension

        # Predict matching scores
        matching_scores = cnn_model.predict(cars_images)
        
        # Find the index with the highest matching score
        max_arg = np.argmax(matching_scores)
        
        # Get the value of the highest matching score
        val = matching_scores[max_arg]

        if val == 0:
            return None, matching_scores
        else:
            return car_ids[max_arg], matching_scores 
    
    def update(self, detections, current_frame):
        self.current_frame = current_frame
        self.frame_count += 1
        self.unidentified_cars = []

        for bounding_box in detections:

            max_car_id, max_car_coverage, car_ids_covered = self.process_coverages(bounding_box)

            if max_car_coverage >= 0.75: # count this as a match
                self.update_car(max_car_id, bounding_box)
            
            elif max_car_coverage >= 0.3: # use CNN to identify which car is it
                car_image = self.get_image(bounding_box)
                car_id, matching_scores = self.get_car_id_with_highest_matching_score(car_ids_covered, car_image)
                # if car_id here is None, then the the matching_score was zero (which is not supposed to be the case)
                # because the car must be visible, hence the CNN shouldn't give 0.0 in principle(unless it is not
                # well trained)
                self.update_car(car_id, bounding_box)    
            else: # car unidentifiable based on position only
                self.unidentified_cars.append(bounding_box)

        temp_unidentified_cars = []    
        for bounding_box in self.unidentified_cars:
            car_image = self.get_image(bounding_box)
            # TODO: update this to take into consideration the distance
            car_id, matching_scores = self.get_car_id_with_highest_matching_score(self.unmatched_cars, car_image)
            if car_id is None:
                temp_unidentified_cars.append(bounding_box)
            else:
                self.update_car(car_id, bounding_box)
        self.unidentified_cars = temp_unidentified_cars

        next_unmatched_cars = []
        for car_id in self.unmatched_cars:
            # check if the RNN is predicting the car to go out of bounds
            last_pred = self.cars_info[car_id]["predicted"][-1]
            x, y, a, h = last_pred[0], last_pred[1], last_pred[2], last_pred[3]
            if (x + a*h) >= self.W or (y + h) >= self.H:
                # is exiting the video
                self.exited_cars[car_id] = self.cars_info.pop(car_id)
            else:
                # is not close to exiting the video yet
                next_unmatched_cars.append(car_id)
        self.unmatched_cars = next_unmatched_cars

        for bounding_box in self.unidentified_cars:
            # Probably new cars.
            self.add_car(bounding_box)

        self.unmatched_cars = list(self.cars_info.keys())

    def process_coverages(self, bounding_box):
        x, y, width, height = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        x_c, y_c, width_c, height_c = self.clip_coords(x, y, width, height)
        car_ids_covered = list(set(self.region_map[x_c: x_c + width_c, y_c: y_c + height_c].reshape(-1).tolist()))
        coverages =[]

        if sum(car_ids_covered) == 0: # because the region map is initialized with zeroes and ids 
                                      # are only positive, we can just check whether
                                      # the sum of checked ids is zero
            return -1, 0, []

        for car_id in car_ids_covered:
            time_elapsed = self.cars_info[car_id]["time_elapsed"]
            preds = self.cars_info[car_id]["predicted"]

            if time_elapsed >= predict_size:
                time_elapsed = -1
            coverages.append(self.get_coverage(preds[time_elapsed], bounding_box))

        max_idx = np.argmax(coverages)
        max_car_id = car_ids_covered[max_idx]
        max_car_coverage = coverages[max_idx]

        return max_car_id, max_car_coverage, car_ids_covered

    def update_region_map(self, bounding_box, car_id):
        """
        Takes a bounding box with a car id and updates the region map at those positions
        with the car id
        """
        x, y, width, height = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        x, y, width, height = self.clip_coords(x, y, width, height)
        self.region_map[x: x + width, y: y + height] = car_id
        
    def exit_car(self, car_id):
        exiting_car = self.cars_info.pop(car_id)
        self.exited_cars.append(exiting_car)


    def clip_coords(self, x, y, width, height):
        if x + width > self.W:
            width = x + width - self.W
        if y + height > self.H:
            height = y + height - self.H
        return [x, y, width, height]
    
    def get_image(self, bounding_box):
        x, y, w, h = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        x, y, w, h = self.clip_coords(x, y, w, h)
        return self.current_frame[x: x + w, y: y + h]
    
    def get_coord(self, bounding_box):
        a = bounding_box[2] / bounding_box[3]
        coord_pts = bounding_box
        coord_pts[2] = a
        return coord_pts
    
    def get_bounding_box(self, coord):
        w = int(coord[2] * coord[3])
        return [coord[0], coord[1], w, coord[3]]
    
    def get_coverage(self, coord_1, bounding_box):
        """
        Given two coordinates coord_1==(x1, y1, a1, h1) and bounding_box==(x2, y2, w2, h2), get the percentage of coord_1 that 
        is covered by bounding_box
        """
        x1, y1, a1, h1 = coord_1[0], coord_1[1], coord_1[2], coord_1[3]
        x2, y2, w2, h2 = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        w1 = int(a1 * h1)

        if x1 < x2:
            width = (x1 + w1) - x2
        else:
            width = (x2 + w2) - x1
        
        if y1 > y2:
            height = y1 - (y2 + h2)
        else:
            height = y2 - (y1 + h1)
        
        area = width * height
        coverage = area / (w1 * h1)

        return coverage