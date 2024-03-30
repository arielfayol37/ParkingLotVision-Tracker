import tensorflow as tf
import numpy as np
from ai_training.best_cnn_training import get_model as cnn_get_model
from ai_training.best_rnn_training import get_model as rnn_get_model
import sys 
import matplotlib.pyplot as plt
import cv2
import copy

cnn_model = cnn_get_model()
rnn_model = rnn_get_model()


rnn_model.load_weights("ai_weights/car_trajectory_best_validation_final_weights.h5")
cnn_model.load_weights("ai_weights/cnn_reidentification_weights.h5")

context_size = 20
predict_size = 8



IMG_SIZE = (37, 37)

def scale_resize_image(image, img_size=IMG_SIZE):
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tf.image.resize(image, img_size)
    return image  # Convert back to numpy array if further numpy operations are needed

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
        """
        Adds a new car to the cars_info dictionary.
        This should only be called when the bounding box could not be identified as a previously present car.
        """
        
        #  print("bounding box", bounding_box)
        last_seen_image = self.get_image(bounding_box)        
        coord_pts = self.get_coord(bounding_box)
        self.cars_info[int(self.next_id)] = {"coords":[coord_pts for _ in range(context_size)], "time_elapsed":0, \
                                        "last_seen":last_seen_image, "first_bb": bounding_box, "last_bb":bounding_box,\
                                            "predicted":[coord_pts for _ in range(predict_size)], "time_update_pred":0}
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
        self.cars_info[car_id]["time_update_pred"] += 1
        if bounding_box: # if car has been matched
            try:
                self.unmatched_cars.remove(int(car_id))
            except:
                print("did not find id to remove: unmatched cars, car_id", self.unmatched_cars, car_id)
            coord_pts = self.get_coord(bounding_box)
            last_seen_image = self.get_image(bounding_box)
            self.cars_info[car_id]["last_bb"] = bounding_box # will be used at the end for the analysis
            if self.cars_info[car_id]["time_elapsed"] >= predict_size: # this should when it loses the car for more than 8 frames
                # so this as though it is seeing the car for the first time
                self.cars_info[car_id]["coords"] = [coord_pts for i in range(context_size)]
                self.cars_info[car_id]["predicted"] = [coord_pts for i in range(predict_size)]
            else:
                self.cars_info[car_id]["coords"].append(coord_pts)
                # TODO: this can be optimized by not having to predict at every timestep
                if self.cars_info[car_id]["time_update_pred"] >= predict_size:
                    self.cars_info[car_id]["time_update_pred"] = 0
                    self.predict_car_position(car_id) # let us make sure that this is taking the right input before predicting by visualizing
                # self.cars_info[car_id]["predicted"] = [coord_pts for i in range(predict_size)]
            pred_0 = self.cars_info[car_id]["predicted"][0]
            self.update_region_map(self.get_bounding_box(pred_0), car_id)
            
            self.cars_info[car_id]["time_elapsed"] = 0
            self.cars_info[car_id]["last_seen"] = last_seen_image
            
            
        else: # in case car has not been matched
            time_elapsed = self.cars_info[car_id]["time_elapsed"]
            if  time_elapsed >= predict_size:
                copy_last_coord = copy.deepcopy(self.cars_info[car_id]["coords"][-1])
                self.cars_info[car_id]["coords"].append(copy_last_coord)
                # Wanted to clear the predicted positions but
                # those (the last one) might be used by the CNN for later comparisons
            else:
                p = self.cars_info[car_id]["predicted"][time_elapsed]
                self.cars_info[car_id]["coords"].append(p)
            
            self.cars_info[car_id]["time_elapsed"] += 1

    def predict_car_position(self, car_id):

 
        # frame_copy = copy.deepcopy(self.current_frame)         
        prev_20_steps = np.array(self.cars_info[car_id]["coords"]).reshape(-1, context_size, 4)
        preds = rnn_model(prev_20_steps).numpy().reshape(predict_size, 4).tolist()
        self.cars_info[car_id]["predicted"] = preds
        # prev_path_color = (0, 100, 255)
        # predict_path_color = (255, 50, 50)
        # draw_on_frame(frame_copy, coords=self.cars_info[car_id]["coords"], color=prev_path_color)
        # draw_on_frame(frame_copy, coords=preds, color=predict_path_color)

    def get_car_id_with_highest_matching_score(self, car_ids, car_image, x_y_coord):
        """
        Given a list of car_ids and an image, return the id with the closest match
        or None if zero match for each.
        """
        # Assuming self.cars_info[car_id]["last_seen"] is a numpy array of shape (37, 37, 3)
        # and car_image is also of shape (37, 37, 3)
        car_image = scale_resize_image(car_image)
        cars_images_a = np.array([
            np.stack(car_image)
            for _ in car_ids
        ])
        cars_images_b = np.array([
            np.stack(scale_resize_image(self.cars_info[car_id]["last_seen"])) for car_id in car_ids
        ])
        assert cars_images_a.shape == cars_images_b.shape, f"Expected a and b to have the same shapes but got \
                                                            a.shape = {cars_images_a.shape}, b.shape = {cars_images_b.shape}"
        # Correct shape assertion for debugging
        assert cars_images_a.shape == (len(car_ids), 37, 37, 3), f"Expected shape {(len(car_ids), 37, 37, 3)}, but got {cars_images_a.shape}"
        
        # Reshape to fit the model's expected input shape if necessary
        # cars_images = cars_images.reshape(-1, 37, 37, 3)  # Only if your model expects all images at once without the pair dimension

        # Predict matching scores
        cnn_matching_scores = cnn_model.predict([cars_images_a, cars_images_b], verbose=0)
        # car_positions = [[self.cars_info[car_id]["coords"][0], self.cars_info[car_id]["coords"][1]] for car_id in car_ids]
        # dist_matching_scores = list(map(lambda pos: 1 - ((pos[0] - x_y_coord[0]) ** 2 + (pos[1] - x_y_coord[1]) **2)/(self.W * self.H), car_positions))
        # matching_scores = [0.5 * cnn_matching_scores[i] + 0.5 * dist_matching_scores[i] for i in range(len(cnn_matching_scores))]

        # Find the index with the highest matching score
        max_arg = np.argmax(cnn_matching_scores)
        
        # Get the value of the highest matching score
        val = cnn_matching_scores[max_arg]

        if val == 0:
            """
            print(matching_scores)
            
            for i in range(len(car_ids)):
                        # Plot the images side by side
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(cars_images_a[i])
                axs[0].set_title('Image A')
                axs[0].axis('off')
                
                axs[1].imshow(cars_images_b[i])
                axs[1].set_title('Image B')
                axs[1].axis('off')
                
                # Show prediction
                # plt.suptitle(f'PS, RS: {prediction[0][0]:.2f}, {data_Y[index]}')
                plt.show()            
            """
            return None, cnn_matching_scores
        else:
            return car_ids[max_arg], cnn_matching_scores 
    
    def update(self, detections, current_frame):
        """
        The most important method of the entire class. 
        Here is kinda the main loop of the algorithm
        """
        self.current_frame = current_frame
        self.frame_count += 1
        self.unidentified_cars = []
        # print("car ids present ",self.cars_info.keys())
        # print("number of cars detected by yolo: ",len(detections))
        for bounding_box in detections:

            max_car_id, max_car_coverage, car_ids_covered = self.process_coverages(bounding_box)
            # print("car ids covered", car_ids_covered)

            if len(car_ids_covered) == 1:
                car_id = car_ids_covered[0]
                self.update_car(car_id, bounding_box)
            else:
                if max_car_coverage >= 0.75: # count this as a match
                    # print(max_car_id, "a")
                    self.update_car(max_car_id, bounding_box)
                
                elif max_car_coverage >= 0.3: # use CNN to identify which car is it
                    car_image = self.get_image(bounding_box)
                    x_y_coord = bounding_box[0], bounding_box[1]
                    car_id, matching_scores = self.get_car_id_with_highest_matching_score(car_ids_covered, car_image, x_y_coord)
                    # if car_id here is None, then the the matching_score was zero (which is not supposed to be the case)
                    # because the car must be visible, hence the CNN shouldn't give 0.0 in principle(unless it is not
                    # well trained)
                    # TODO: the cnn is not well trained and returns None sometimes when it shouldn't
                    # print(car_id, "b")
                    if car_id is None:
                        print("unexpected None in car matching")
                        self.unidentified_cars.append(bounding_box)
                    else:
                        self.update_car(car_id, bounding_box)    
                else: # car unidentifiable based on position only
                    self.unidentified_cars.append(bounding_box)

        temp_unidentified_cars = []    
        for bounding_box in self.unidentified_cars:
            """
            TODO: improve this clause because some unmatched cars are just still unmatched because
            the detector did not detect the car in the first place. So basing ourselves just on the looks
            may force an id on a bounding box which already belongs to another car which has not just been 
            detected on this particular frame

            -Make use of the distance from last seen and the time elapsed too. How far a car should be from the 
             last time should also depend on how much time has passed.
            """
            car_image = self.get_image(bounding_box)
            x_y_coord = bounding_box[0], bounding_box[1]
            # TODO: update this to take into consideration the distance
            if len(self.unmatched_cars) == 0:
                car_id = None
            else:
                car_id, matching_scores = self.get_car_id_with_highest_matching_score(self.unmatched_cars, car_image, x_y_coord)

            if car_id is None:
                temp_unidentified_cars.append(bounding_box)
            else:
                self.update_car(car_id, bounding_box)
        self.unidentified_cars = temp_unidentified_cars

        for car_id in self.unmatched_cars:
            # check if the RNN is predicting the car to go out of bounds
            last_pred = self.cars_info[car_id]["predicted"][-1]
            x, y, a, h = last_pred[0], last_pred[1], last_pred[2], last_pred[3]
            if (x + a*h) >= self.W or (y + h) >= self.H:
                # is exiting the video
                self.exit_car(car_id)
            else:
                # car has not yet exited
                pass

        for bounding_box in self.unidentified_cars:
            # Probably new cars.
            self.add_car(bounding_box)

        self.unmatched_cars = list(self.cars_info.keys())

    def process_coverages(self, bounding_box):
        x, y, width, height = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        x_c, y_c, width_c, height_c = self.clip_coords(x, y, width, height)
        ids = set(self.region_map[x_c: x_c + width_c, y_c: y_c + height_c].reshape(-1).tolist())
        ids.discard(0)
        if len(ids) != 0:
            car_ids_covered = list(ids)
        else:
            car_ids_covered = []

        coverages =[]

        if len(car_ids_covered) == 0: # because the region map is initialized with zeroes and ids start with 1
            return -1, 0, []

        for car_id in car_ids_covered:
            time_elapsed = self.cars_info[int(car_id)]["time_elapsed"]
            preds = self.cars_info[car_id]["predicted"]

            if time_elapsed >= predict_size:
                time_elapsed = -1 # used to get the last predicted coord
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
        self.region_map[self.region_map==car_id] = 0
        self.region_map[x: x + width, y: y + height] = car_id
        
    def exit_car(self, car_id):
        self.exited_cars[car_id] = self.cars_info.pop(car_id)
        self.region_map[self.region_map==car_id] = 0


    def clip_coords(self, x, y, width, height):
        # Ensure x and y are not less than 0
        x = max(x, 0)
        y = max(y, 0)

        # Adjust width and height to not exceed image boundaries
        if x + width > self.W:
            width = self.W - x
        if y + height > self.H:
            height = self.H - y

        return [int(x), int(y), int(width), int(height)]
    
    def get_image(self, bounding_box):
        x, y, w, h = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        x, y, w, h = self.clip_coords(x, y, w, h)
        img = self.current_frame[y: y + h, x: x + w]
        """
        if img.shape[0] == 0 or img.shape[1] == 0:
            print("gotcha", img.shape, x, y, w, h, self.W, self.H, self.current_frame.shape)
            sys.exit()        
        """

        return img
    
    def get_coord(self, bounding_box):
        a = bounding_box[2] / bounding_box[3]
        coord_pts = [bounding_box[0], bounding_box[1], a, bounding_box[3]]
        coord_pts[2] = a
        return coord_pts
    
    def get_bounding_box(self, coord):
        w = int(coord[2] * coord[3])
        return [coord[0], coord[1], w, coord[3]]
    
    def get_coverage(self, coord_1, bounding_box):
        """
        Given two coordinates coord_1==(x1, y1, a1, h1) and bounding_box==(x2, y2, w2, h2), get the percentage of coord_1 that 
        is covered by bounding_box
        # could return negative values
        """
        x1, y1, a1, h1 = coord_1[0], coord_1[1], coord_1[2], coord_1[3]
        x2, y2, w2, h2 = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        w1 = int(a1 * h1)

        if x1 < x2:
            width = (x1 + w1) - x2
        else:
            width = (x2 + w2) - x1
        
        if y1 < y2:
            height = y1 + h1 - y2
        else:
            height = y2 + h2 - y1
        
        area = width * height
        try:
            coverage = area / (w1 * h1)
            return coverage
        except:
            print(coord_1, bounding_box)
            return 0
        


def draw_on_frame(new_frame, coords, color):
    for dp in coords:
        width = float(dp[2]) * int(dp[3])
        height = int(dp[3])
        x1, y1, x2, y2 = int(dp[0]), int(dp[1]), int(float(dp[0]) + width), int(dp[1]) + height
        cv2.rectangle(new_frame, (x1, y1), (x2, y2), color, 3)

        cv2.imshow("Frame", new_frame)
        key = cv2.waitKey(200)  