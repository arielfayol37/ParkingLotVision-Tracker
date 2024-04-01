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

sigma = 4
critical_snn_score = 0.5
max_percent_fame_move = 0.15    # the 0.15 is arbirtarily chosen by intuition
                                # for each frame, car should not move more than
                                # 0.15 of its width I think


IMG_SIZE = (37, 37)

def scale_resize_image(image, img_size=IMG_SIZE):
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tf.image.resize(image, img_size)
    return image  

"""
    time_elapsed: time (num of frames) since last prediction
"""
class Tracker:
    def __init__(self, W, H):
        self.next_id = [i+1 for i in range(100)]
        self.id_add_next = 101
        self.cars_info = {}
        self.W, self.H = W, H # Video width, video height
        self.region_map = np.zeros((W, H))
        self.road_map = np.zeros((W, H))
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
        car_id = self.next_id.pop(0)
        self.cars_info[int(car_id)] = {"coords":[coord_pts for _ in range(context_size)], "time_elapsed":0, \
                                        "last_seen":last_seen_image, "first_bb": bounding_box, "last_bb":bounding_box,\
                                            "predicted":[coord_pts for _ in range(predict_size)], "time_update_pred":0,\
                                                "iterate_count":0, "exiting_count":0}
        self.update_region_map(bounding_box, car_id)
        self.next_id.append(self.id_add_next)
        self.id_add_next += 1
        

    def update_car(self, car_id, bounding_box=None):
        """
        Takes a car id and a bounding_box(optional)
        - if bounding_box, then the car has been identified in the image
        - Update the car's position coordinates and predictions
        - Update the region_map
        """

        car_dict = self.cars_info[car_id]
        car_dict["coords"].pop(0)
        car_dict["time_update_pred"] += 1
        car_dict["iterate_count"] += 1

        if bounding_box: # if car has been matched
            # check if the car is exiting
            x, y, w, h = bounding_box
            if car_dict["iterate_count"] >= 30 and \
                (x - sigma < 0 or x + w + sigma > self.W or y - sigma < 0 or y + h + sigma > self.H):
                # car is exiting
                car_dict["exiting_count"] += 1
                last_seen_image = self.get_image(bounding_box)
                car_dict["last_seen"] = last_seen_image
                self.update_region_map(bounding_box, car_id)
                car_dict["coords"].append(self.get_coord(bounding_box))


            else:
                try:
                    self.unmatched_cars.remove(int(car_id)) # TODO: check why this happens
                except:
                    pass 
                    # print("did not find id to remove: unmatched cars, car_id, cars", self.unmatched_cars, car_id)
                coord_pts = self.get_coord(bounding_box)
                last_seen_image = self.get_image(bounding_box)
                car_dict["last_bb"] = bounding_box # will be used at the end for the analysis
                car_dict["last_seen"] = last_seen_image

                if car_dict["time_elapsed"] >= predict_size: # this should when it loses the car for more than 8 frames
                    # so this as though it is seeing the car for the first time
                    car_dict["coords"] = [coord_pts for i in range(context_size)]
                    car_dict["predicted"] = [coord_pts for i in range(predict_size)]
                else:
                    car_dict["coords"].append(coord_pts)
                    # TODO: this can be optimized by not having to predict at every timestep
                    # the brief idea is in the commented code below. Check before running
                    """
                    if car_dict["time_update_pred"] >= predict_size:
                        self.predict_car_position(car_id)

                pred_current = car_dict["predicted"][car_dict["time_update_pred"]]
                self.update_region_map(self.get_bounding_box(pred_current), car_id)                    
                    """

                
                self.predict_car_position(car_id) # let us make sure that this is taking the right input before predicting by visualizing
                # car_dict["predicted"] = [coord_pts for i in range(predict_size)]


                pred_0 = car_dict["predicted"][0]
                self.update_region_map(self.get_bounding_box(pred_0), car_id)
                
            car_dict["time_elapsed"] = 0
                
                
            
        else: # Car has not been matched

            time_elapsed = car_dict["time_elapsed"]
            car_dict["time_elapsed"] += 1

            if  time_elapsed >= predict_size:
                copy_last_coord = copy.deepcopy(car_dict["coords"][-1])
                car_dict["coords"].append(copy_last_coord)
                # Wanted to clear the predicted positions but
                # those (the last one) might be used by the CNN for later comparisons
            else:
                p = car_dict["predicted"][time_elapsed]
                car_dict["coords"].append(p)
            
            
            exiting_count = car_dict["exiting_count"]
            deleted = None

            if exiting_count != 0: # if car already started exiting
                if time_elapsed >= 8:# car has not been seen for 8 frames
                    if car_dict["iterate_count"] >= 30:
                        # this means that the car has been seen 
                        # for at least 30 frames (hence likely an actual car)
                        self.exit_car(car_id)
                        # print("exiting car...", car_id)
                    else:
                        # the car has not been seen up to 30 frames
                        # so it probably was just an exiting car that was mistook for a new car
                        deleted = True
                        self.recycle_id(car_id)
                        # print("deleted id ", car_id)

            if time_elapsed >= 10 and 10 <= car_dict["iterate_count"] <= 30: # TODO: rethink about this recycling logic
                deleted = True
                self.recycle_id(car_id) 
                
            if not deleted:
                car_dict["exiting_count"] += 1

    def recycle_id(self, car_id):

        self.next_id.insert(0, int(car_id))
        deleted = self.cars_info.pop(car_id)
        self.region_map[self.region_map==car_id] = 0


    def predict_car_position(self, car_id):

 
        # frame_copy = copy.deepcopy(self.current_frame) 
        car_dict = self.cars_info[car_id]
        prev_20_steps = np.array(car_dict["coords"]).reshape(-1, context_size, 4)
        preds = rnn_model(prev_20_steps).numpy().reshape(predict_size, 4).tolist()
        car_dict["predicted"] = preds
        car_dict["time_update_pred"] = 0
        # prev_path_color = (0, 100, 255)
        # predict_path_color = (255, 50, 50)
        # draw_on_frame(frame_copy, coords=self.cars_info[car_id]["coords"], color=prev_path_color)
        # draw_on_frame(frame_copy, coords=preds, color=predict_path_color)

    def get_car_id_with_highest_matching_score(self, car_ids, car_image, x_y_coord = None):
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
        cnn_matching_scores = cnn_model.predict([cars_images_a, cars_images_b], verbose=0).reshape(-1)
        # car_positions = [[self.cars_info[car_id]["coords"][0], self.cars_info[car_id]["coords"][1]] for car_id in car_ids]
        # dist_matching_scores = list(map(lambda pos: 1 - ((pos[0] - x_y_coord[0]) ** 2 + (pos[1] - x_y_coord[1]) **2)/(self.W * self.H), car_positions))
        # matching_scores = [0.5 * cnn_matching_scores[i] + 0.5 * dist_matching_scores[i] for i in range(len(cnn_matching_scores))]

        # Find the index with the highest matching score
        max_arg = np.argmax(cnn_matching_scores)
        
        # Get the value of the highest matching score
        val = cnn_matching_scores[max_arg]

        if x_y_coord is None: # This should be none unless unidentified cars are tried to be matched to
                              # to unmatched cars
            
            if val <= critical_snn_score :
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
                return None
            else:
                return car_ids[max_arg]
        else:
            if val <= critical_snn_score:
                # doesn't look like any of the cars
                return None
            else:
                # looks like one of the cars but is it the same car ?
                # how far is this car(s) though ?
                # print(cnn_matching_scores)
                indices = cnn_matching_scores > critical_snn_score
                # print(indices, car_ids)
                non_zero_look_alike_car_ids = list(np.array(car_ids)[indices])

                within_range_ids = []

                for idx, id in enumerate(non_zero_look_alike_car_ids):
                    car_dict = self.cars_info[id]
                    coords = car_dict["coords"][-1]
                    time_elapsed = car_dict["time_elapsed"]
                    x, y, a, h = coords[0], coords[1], coords[2], coords[3]
                    w = a * h
                    dist_squared = (x - x_y_coord[0]) ** 2 + (y - x_y_coord[1]) ** 2
                    allowed_dist_squared = (max_percent_fame_move * w * time_elapsed) ** 2 
                    if dist_squared < allowed_dist_squared:
                        # within acceptable range
                        within_range_ids.append([idx, id])
                
                # now return the acceptable id with the highest cnn match
                # TODO: may be you should take into account both the cnn matching score
                # and dist_squared
                return_id = None
                max_match_score = 0
                scores = cnn_matching_scores[indices]
                for idx, id in within_range_ids:
                    if scores[idx] > max_match_score:
                        return_id = id
                        max_match_score = scores[idx]
                
                return return_id


                

    
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
      
            if max_car_coverage >= 0.75: # count this as a match
                # print(max_car_id, "a")
                
                self.update_car(max_car_id, bounding_box)
            
            elif max_car_coverage >= 0.3: # use CNN to identify which car is it
                
                if len(car_ids_covered) == 1: # TODO: may be you should remove this. Though it saves time, it can lead
                                              # to an ID transfer (not good) like the bicycle and car in video_3
                    car_id = car_ids_covered[0]
                    self.update_car(car_id, bounding_box)

                else:
                
                # indent block below if you uncomment this three quotes comment
               

                    car_image = self.get_image(bounding_box)
                    # x_y_coord = bounding_box[0], bounding_box[1]
                    car_id = self.get_car_id_with_highest_matching_score(car_ids_covered, car_image)
                    # if car_id here is None, then the the matching_score was zero (which is not supposed to be the case)
                    # because the car must be visible, hence the CNN shouldn't give 0.0 in principle(unless it is not
                    # well trained)
                    # TODO: the cnn is not well trained and returns None sometimes when it shouldn't
                    #       - response to TODO above. Well, it may happen that it covers a good portion
                    #       - of another car_id from the region map, yet it is not the same car! (ID transfer)
                    #       - An ID transfer can happen easily if a vehicle (bicycle for example)
                    #          is identified once (hence updating the region map), but not identied later
                    #          As such, when an actual car will cover that region, it might take that ID.
                    #          so it is Indeed possible that if the CNN checks, it may realize that it is a different car.
                    # print(car_id, "b")
                    if car_id is None:
                        # print("car not identified using cnn though coverage was more than 30%")
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

            - Also, remember that new cars should usually(besides the first z << n[n=number of frames]) 
              appear at the borders of the screen
            """
            car_image = self.get_image(bounding_box)
            x_y_coord = bounding_box[0], bounding_box[1]
            # TODO: update this to take into consideration the distance
            if len(self.unmatched_cars) == 0:
                car_id = None
            else:
                car_id = self.get_car_id_with_highest_matching_score(self.unmatched_cars, car_image, x_y_coord)

            if car_id is None:
                temp_unidentified_cars.append(bounding_box)
            else:
                # print(car_id, "c")
                self.update_car(car_id, bounding_box)
        self.unidentified_cars = temp_unidentified_cars

        for unmatched_id in self.unmatched_cars:
            self.update_car(unmatched_id)

        for bounding_box in self.unidentified_cars:
            # Probably new cars.
            self.add_car(bounding_box)

        self.unmatched_cars = list(self.cars_info.keys())

    def process_coverages(self, bounding_box):
        x, y, width, height = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        x_c, y_c, width_c, height_c = self.clip_coords(x, y, width, height)
        ids = set(self.region_map[x_c: x_c + width_c, y_c: y_c + height_c].reshape(-1).tolist())
        ids.discard(0) # because the region map is initialized with zeroes and ids start with 1
        if len(ids) != 0:
            car_ids_covered = list(ids)
        else:
            car_ids_covered = []

        coverages =[]

        if len(car_ids_covered) == 0: 
            return -1, 0, []

        for car_id in car_ids_covered:
            car_dict = self.cars_info[car_id]
            time_elapsed = car_dict["time_elapsed"]
            preds = car_dict["predicted"]

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
        self.road_map[x: x + width, y: y + height] = 1
        
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
        divisor = w1 * h1
        
        if divisor == 0:
            return 0
        
        coverage = area / (w1 * h1)
        return coverage

    def get_coord(self, bounding_box):
        a = bounding_box[2] / bounding_box[3]
        coord_pts = [bounding_box[0], bounding_box[1], a, bounding_box[3]]
        coord_pts[2] = a
        return coord_pts
    
    def get_bounding_box(self, coord):
        w = int(coord[2] * coord[3])
        return [coord[0], coord[1], w, coord[3]]
      
        


def draw_on_frame(new_frame, coords, color):
    for dp in coords:
        width = float(dp[2]) * int(dp[3])
        height = int(dp[3])
        x1, y1, x2, y2 = int(dp[0]), int(dp[1]), int(float(dp[0]) + width), int(dp[1]) + height
        cv2.rectangle(new_frame, (x1, y1), (x2, y2), color, 3)

        cv2.imshow("Frame", new_frame)
        key = cv2.waitKey(200)  