from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os
from tensorflow.keras.models import model_from_json
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from sort import Sort
from scipy.interpolate import interp1d

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

MARGIN = 40  # pixels
ROW_SIZE = 3  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)  # red
POINT_RADIUS = 5
def get_rectified(text, num_frame):
    if 'rectified' in text.lower():
        with open('rectified.txt', 'a') as file:
            file.write(f"{num_frame} {text}\n")

def is_segment_intersect(segment_line, trapezoid):
    # Extract coordinates of bounding box corners
    xmin, ymin, xmax, ymax = segment_line

    # Extract coordinates of trapezoid vertices
    x1, y1, x2, y2, x3, y3, x4, y4 = trapezoid

    def is_point_on_line(p1, p2, p):
        return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])

    def intersect(a, b, c, d):
        denominator = (b[1] - a[1]) * (d[0] - c[0]) - (a[0] - b[0]) * (c[1] - d[1])
        if denominator == 0:
            return None  # Lines are parallel or coincident

        intersect_x = ((b[0] - a[0]) * (d[0] - c[0]) * (c[1] - a[1]) + (b[1] - a[1]) * (d[0] - c[0]) * a[0] - (d[1] - c[1]) * (b[0] - a[0]) * c[0]) / denominator
        intersect_y = -((b[1] - a[1]) * (d[1] - c[1]) * (c[0] - a[0]) + (b[0] - a[0]) * (d[1] - c[1]) * a[1] - (d[0] - c[0]) * (b[1] - a[1]) * c[1]) / denominator

        intersection = (intersect_x, intersect_y)
        if is_point_on_line(a, b, intersection) and is_point_on_line(c, d, intersection):
            return intersection
        else:
            return None

    # Check for intersection between the segment and each of the four segments of the trapezoid
    trapezoid_edges = [(x1, y1, x2, y2), (x2, y2, x3, y3), (x3, y3, x4, y4), (x4, y4, x1, y1)]
    for edge in trapezoid_edges:
        intersection = intersect(edge[:2], edge[2:], (xmin, ymin), (xmax, ymax))
        if intersection is not None:
            return True

    return False

def predict_bbox_coords_future(futureX, futureY,bbox_coords_current):
    xmin,ymin,xmax,ymax=bbox_coords_current
    xmin=int(futureX-(xmax-xmin)/2)
    xmax=int(futureX+(xmax-xmin)/2)
    ymin=int(futureY-(ymax-ymin)/2)
    ymax=int(futureY+(ymax-ymin)/2)
    
    bbox_coords_future=(xmin,ymin,xmax,ymax)
    
    return bbox_coords_future

def set_trapezoid_coords(ratiox1,ratioy1,ratiox2,ratioy2,ratiox3,ratioy3,ratiox4,ratioy4,w_scale,ratio_h_w):
    h_scale=int(w_scale*ratio_h_w)
    x1=ratiox1*w_scale
    y1=ratioy1*h_scale
    
    x2=ratiox2*w_scale
    y2=ratioy2*h_scale
    
    x3=ratiox3*w_scale
    y3=ratioy3*h_scale
    
    x4=ratiox4*w_scale
    y4=ratioy4*h_scale
    return x1, y1, x2, y2, x3, y3, x4, y4,w_scale,h_scale

def set_dataset(choice):
    if (choice=="pie"):
        ratiox1=353/768
        ratioy1=333/432
        
        ratiox2=410/768
        ratioy2=333/432
        
        ratiox3=593/768
        ratioy3=432/432
        
        ratiox4=170/768
        ratioy4=432/432
        path = cv2.imread('path_pie.png')
    else:
        ratiox1=342/768
        ratioy1=288/432
        
        ratiox2=426/768
        ratioy2=288/432
        
        ratiox3=620/768
        ratioy3=432/432
        
        ratiox4=150/768
        ratioy4=432/432
        path = cv2.imread('path_jaad.png')        
    
    return ratiox1,ratioy1,ratiox2,ratioy2,ratiox3,ratioy3,ratiox4,ratioy4,path
def get_delta(pose_piles,id,n):
    if id not in pose_piles:
        return None

    delta= pose_piles[id]["time"][-1]- pose_piles[id]["time"][-1-n]
    return delta
def get_last_pile(pose_piles, id):
    if id not in pose_piles:
        return None, None, None, None, None

    last_pose = pose_piles[id]["pile"][-1] if pose_piles[id]["pile"] else None
    last_bbx = pose_piles[id]["bbx"][-1] if pose_piles[id]["bbx"] else None
    last_time = pose_piles[id]["time"][-1] if pose_piles[id]["time"] else None
    last_frame = pose_piles[id]["frame"][-1] if pose_piles[id]["frame"] else None

    return last_pose, last_bbx, last_time, last_frame

def get_data_before_last(pose_piles, id):
    if id not in pose_piles:
        return None

    # Retrieve the data just before the last entry
    pose_before_last = pose_piles[id]["pile"][-2] if len(pose_piles[id]["pile"]) >= 2 else None
    bbx_before_last = pose_piles[id]["bbx"][-2] if len(pose_piles[id]["bbx"]) >= 2 else None
    time_before_last = pose_piles[id]["time"][-2] if len(pose_piles[id]["time"]) >= 2 else None
    frame_before_last = pose_piles[id]["frame"][-2] if len(pose_piles[id]["frame"]) >= 2 else None

    return pose_before_last, bbx_before_last, time_before_last, frame_before_last


def management_pose_pile(pose_piles, id, pose_landmarks, bbox_coords, time, frame):
    id = str(int(id))
    if id not in pose_piles:
        pose_piles[id] = {"pile": [], "bbx": [], "time": [], "frame": [], "new": True}
    
    pose_data = pose_piles[id]["pile"]
    bbx_data = pose_piles[id]["bbx"]
    time_data = pose_piles[id]["time"]
    frame_data = pose_piles[id]["frame"]
    

    #print (f"management_pose_pile received id={id}")
    # Add new record 
    if pose_landmarks is not None:
        pose_data.append(pose_landmarks)    
    bbx_data.append(bbox_coords)
    time_data.append(time)
    frame_data.append(frame)
    
    # Ensure data has 16 records
    if pose_landmarks is not None:
        while len(pose_data) < 16:
            pose_data.insert(0, pose_landmarks)        

    # Ensure data has 16 records
    while len(pose_data) > 16:
        pose_data.pop(0)
    while len(bbx_data) > 16:
        bbx_data.pop(0)
    while len(time_data) > 16:
        time_data.pop(0)
    while len(frame_data) > 16:
        frame_data.pop(0)

    return pose_piles



def get_state_pedestrian(pose_piles, id, loaded_model, threshold):
    pose_data = pose_piles[id]["pile"]
    num_elements = len(pose_data)
    print("Number of elements in pose_data:", num_elements)
    
    # Initialize an empty list to store the tensor data (X) 
    tensor_data = []
    
    # Iterate through each pose estimation in pose_data
    for pose_estimation in pose_data:
        pose_coords_arr = []

        # Extract the x and y coordinates for each landmark (considering 14 landmarks)
        for i in range(33):
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 23, 24]:
                continue

            x, y, _ = get_landmark_coords_ratio(pose_estimation, i)
            
            # Append the (x, y) coordinate to pose_coords_arr
            pose_coords_arr.append([x, y])

        # Convert pose_coords_arr to a NumPy array to create the 3D tensor (14, 2) for the pose estimation
        pose_tensor = np.array(pose_coords_arr)

        # Append the pose_tensor to the list
        tensor_data.append(pose_tensor)

    # Convert the list of tensors to a NumPy array to create the 4D tensor (num_elements, 14, 2)
    X = np.array(tensor_data)
    print ("Shape of X befor: ",X.shape)

    X = tf.reshape(X, shape=(-1, 16, 28))


    # Now X will have the shape (num_elements, 14, 2), where num_elements is the number of pose estimations in pose_data
    # You can feed this X to your trained model to make predictions
    print ("Shape of X after: ",X.shape)
    
    
    # Now, you can use the loaded_model to make predictions on this padded_tensor
    predicted_probabilities = loaded_model.predict(X)
    
    # Assuming binary classification, get the predicted class (0 or 1) based on the threshold
    predicted_class = 1 if predicted_probabilities > threshold else 0
    predicted_probabilities = predicted_probabilities[0][0] if predicted_probabilities > threshold else 1-predicted_probabilities[0][0]
    print("Predicted Class:", predicted_probabilities)
    return predicted_class, predicted_probabilities




def prediction_direction(R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x,R_Ankle_x,L_Ankle_x):
    score = 0
    if R_Foot_Index_x > R_Ankle_x:
        score += 1
    else:
        score -= 1
        
    if R_Foot_Index_x > R_Heel_x:
        score += 1
    else:
        score -= 1
        
    if L_Foot_Index_x > L_Ankle_x:
        score += 1
    else:
        score -= 1
        
    if L_Foot_Index_x > L_Heel_x:
        score += 1
    else:
        score -= 1
    
    if score > 0:
        direction = 'LR'
    elif (score<0):
        direction = 'RL'
    else:
        direction = 'N'
            
    
    return direction
# function to return the magnitude of a vector
def vec_length(v: np.array):
    return np.sqrt(sum(i ** 2 for i in v))

# function to process a vector parameter and return a normalized vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# function to calculate and return a rotation matrix for quaternion generation
def look_at(eye: np.array, target: np.array):
    axis_z = normalize((eye - target))
    if vec_length(axis_z) == 0:
        axis_z = np.array((0, -1, 0))

    axis_x = np.cross(np.array((0, 0, 1)), axis_z)
    if vec_length(axis_x) == 0:
        axis_x = np.array((1, 0, 0))

    axis_y = np.cross(axis_z, axis_x)
    rot_matrix = np.matrix([axis_x, axis_y, axis_z]).transpose()
    return rot_matrix


def get_angle(image, sh_right, sh_left, bbox_coords):
    
    x_l, y_l, z_l=sh_left
    x_r, y_r, z_r=sh_right
    
    z_l,_ =  relative_to_absolute(image, z_l, 0, bbox_coords)
    z_r,_ =  relative_to_absolute(image, z_r, 0, bbox_coords)
    
    orient = look_at(
                np.array( [x_l, y_l, z_l] ),
                np.array( [x_r, y_r, z_r] ) )
    
    vec1 = np.array(orient[0], dtype=float)
    vec3 = np.array(orient[1], dtype=float)
    vec4 = np.array(orient[2], dtype=float)
    # normalize to unit length
    vec1 = vec1 / np.linalg.norm(vec1)
    vec3 = vec3 / np.linalg.norm(vec3)
    vec4 = vec4 / np.linalg.norm(vec4)

    M1 = np.zeros((3, 3), dtype=float)  # rotation matrix

      # rotation matrix setup
    M1[:, 0] = vec1
    M1[:, 1] = vec3
    M1[:, 2] = vec4

    # obtaining the quaternion in cartesian form
    a = np.math.sqrt(np.math.sqrt((float(1) + M1[0, 0] + M1[1, 1] + M1[2, 2]) ** 2)) * 0.5
    b1 = (M1[2, 1] - M1[1, 2]) / (4 * a)
    b2 = (M1[0, 2] - M1[2, 0]) / (4 * a)
    b3 = (M1[1, 0] - M1[0, 1]) / (4 * a)

    # converting quaternion to polar form
    A = np.math.sqrt((a ** 2) + (b1 ** 2) + (b2 ** 2) + (b3 ** 2))
    theta = np.math.acos(a / A)
#0.1359234550542877 1.46782358337831 1.478061429738543
    realAngle = ((np.rad2deg(theta) / 45) - 1) * 180

    return realAngle    

def polynomial_interpolation(pose_piles, id, n):
    def get_past_point(pose_piles, id, n):
        if id not in pose_piles:
            return None
        points = []
        for i in range(n + 1):
            bbx = pose_piles[id]["bbx"][-1 - i]
            x, y = get_center(bbx)
            points.append((x, y))

        return points

    # Extract x and y coordinates from the points
    points = get_past_point(pose_piles, id, n)
    t_values = np.arange(n + 1)  # Time values for observed points
    x_coords, y_coords = zip(*points)   
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # Create interpolation functions for x and y coordinates based on time (t)
    x_interp_func = interp1d(t_values, x_coords, kind='cubic')
    y_interp_func = interp1d(t_values, y_coords, kind='cubic')

    return x_interp_func, y_interp_func




def futureXYP(image,timeToFuture,pose_piles,id,n):
    x_interp_func, y_interp_func = polynomial_interpolation(pose_piles, id, n)
    print("timeToFuture:", timeToFuture)
    future_position_x = x_interp_func(timeToFuture)
    future_position_y = y_interp_func(timeToFuture)
    return future_position_x, future_position_y

def futureXYD(image, initial, angle, velocity_x, velocity_y, timeToFuture, err, draw, x_text, y_text,pose_piles,id,direction,loaded_model,text,threshold):
    x_c, y_c, _ = initial
    x_c = int(x_c)
    y_c = int(y_c)
    
    delta_x = np.math.sqrt(((velocity_x * timeToFuture) * np.math.cos(angle)) ** 2)
    delta_y = np.math.sqrt(((velocity_y * timeToFuture) * np.math.cos(angle)) ** 2)

    state_pedestrian, probab = get_state_pedestrian(pose_piles, id,loaded_model,threshold)
    text_state=""
    if(state_pedestrian==1):
        text_state="Walking:"+" "+str(int(probab*100))+"%"
    else:
        text_state="Standing:"+" "+str(int(probab*100))+"%"
        
    cv2.putText(image, text_state, (x_text, (y_text-30)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 0, 255), 1,cv2.LINE_AA)


    if direction=="LR":
        cv2.arrowedLine(image, (x_text, y_text-60), (x_text +100, y_text-60), (0, 0, 255), 2, tipLength=0.3)
    elif direction=="RL":
        cv2.arrowedLine(image, (x_text+100, y_text-60), (x_text, y_text-60), (0, 0, 255), 2, tipLength=0.3)
    else:
        cv2.line(image, (x_text, y_text-60),  (x_text+100, y_text-60) , (0, 0, 255), 2)

    case = "No case detected"

    if (angle > 0) and (angle < 90) and (velocity_x > 0) and (velocity_y < 0):
        case = "case1"
        if(direction=="LR"):
            futureX = x_c + delta_x
        else:
            futureX = x_c - delta_x
           
            case = "case1 rectified"
        futureY = y_c - delta_y
    elif (angle > 0) and (angle < 90) and (velocity_x < 0) and (velocity_y > 0):
        case = "case2"
        if(direction=="RL"):
            futureX = x_c - delta_x
        else:
            futureX = x_c + delta_x
            
            case = "case2 rectified"
        futureY = y_c + delta_y
    elif (angle > 90) and (angle < 180) and (velocity_x > 0) and (velocity_y > 0):
        case = "case3"
        if(direction=="LR"):
            futureX = x_c + delta_x
        else:
            futureX = x_c - delta_x
           
            case = "case3 rectified"

        futureY = y_c + delta_y
        case = "case3"
    elif (angle > 90) and (angle < 180) and (velocity_x < 0) and (velocity_y < 0):
        case = "case4"
        if(direction=="RL"):
            futureX = x_c - delta_x
        else:
            futureX = x_c + delta_x
            
            case = "case4 rectified"
        futureY = y_c - delta_y
    elif (((angle > 0 - err) and (angle < 0 + err)) or ((angle > 180 - err) and (angle < 180 + err))) and (
            velocity_x > 0) and ((velocity_y > 0 - err) and (velocity_y < 0 + err)):
        case = "case5"
        if(direction=="LR"):
            futureX = x_c + delta_x
        else:
            futureX = x_c - delta_x
            
            case = "case5 rectified"
        futureY = y_c
    elif (((angle > 0 - err) and (angle < 0 + err)) or ((angle > 180 - err) and (angle < 190 - err))) and (
            velocity_x < 0) and ((velocity_y > 0 - err) and (velocity_y < 0 + err)):
        case = "case6"
        if(direction=="RL"):
            futureX = x_c - delta_x
        else:
            futureX = x_c + delta_x
            
            case = "case6 rectified"

        futureY = y_c
        case = "case6"
    elif ((angle > 90 - err) and (angle < 90 + err)) and ((velocity_x > 0 - err) and (velocity_x < 0 + err)) and (
            velocity_y > 0):
        futureX = x_c
        futureY = y_c - delta_y
        case = "case7"
    elif ((angle > 90 - err) and (angle < 90 + err)) and ((velocity_x > 0 - err) and (velocity_x < 0 + err)) and (
            velocity_y < 0):
        futureX = x_c
        futureY = y_c + delta_y
        case = "case8"
    else:
        if state_pedestrian==0:
            futureX = x_c
            futureY = y_c
            case = "case9"
        else:
            delta_x2=np.math.sqrt((velocity_x * timeToFuture))
            if direction=="LR":
                futureX = x_c+ delta_x2
            elif direction=="RL":
                futureX = x_c-delta_x2
            else:
                futureX=x_c
            futureY = y_c
            case = "case9 rectified"
           
    # visualizing the direction of movement
    if draw:
        print(f"Drawing path for id {id}")
        cv2.drawMarker(image, (x_c, y_c), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                       thickness=2)
        
        cv2.drawMarker(image, (int(futureX), int(futureY)), color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                       thickness=2)

        cv2.line(image, (x_c, y_c), (int(futureX), int(futureY)), (255, 255, 255), 2)

        
        cv2.putText(image, str(int(id)), (int(x_c+3), int(y_c-3)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        
        cv2.putText(image, str(int(id)), (int(futureX-10), int(futureY-3)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        text=text+case
        cv2.putText(image, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return futureX, futureY, text,image

def futureXY(image, initial, angle, velocity_x, velocity_y, timeToFuture, err, draw, x_text, y_text,pose_piles,id,direction,loaded_model,text,threshold):
    x_c, y_c, _ = initial
    x_c = int(x_c)
    y_c = int(y_c)

    delta_x = np.math.sqrt(((velocity_x * timeToFuture) * np.math.cos(angle)) ** 2)
    delta_y = np.math.sqrt(((velocity_y * timeToFuture) * np.math.cos(angle)) ** 2)
        

    case = "No case detected"

    if (angle > 0) and (angle < 90) and (velocity_x > 0) and (velocity_y < 0):
        futureX = x_c + delta_x
        futureY = y_c - delta_y
        case = "case1"
    elif (angle > 0) and (angle < 90) and (velocity_x < 0) and (velocity_y > 0):
        futureX = x_c - delta_x
        futureY = y_c + delta_y
        case = "case2"
    elif (angle > 90) and (angle < 180) and (velocity_x > 0) and (velocity_y > 0):
        futureX = x_c + delta_x
        futureY = y_c + delta_y
        case = "case3"
    elif (angle > 90) and (angle < 180) and (velocity_x < 0) and (velocity_y < 0):
        futureX = x_c - delta_x
        futureY = y_c - delta_y
        case = "case4"
    elif (((angle > 0 - err) and (angle < 0 + err)) or ((angle > 180 - err) and (angle < 180 + err))) and (
            velocity_x > 0) and ((velocity_y > 0 - err) and (velocity_y < 0 + err)):
        futureX = x_c + delta_x
        futureY = y_c
        case = "case5"
    elif (((angle > 0 - err) and (angle < 0 + err)) or ((angle > 180 - err) and (angle < 190 - err))) and (
            velocity_x < 0) and ((velocity_y > 0 - err) and (velocity_y < 0 + err)):
        futureX = x_c - delta_x
        futureY = y_c
        case = "case6"
    elif ((angle > 90 - err) and (angle < 90 + err)) and ((velocity_x > 0 - err) and (velocity_x < 0 + err)) and (
            velocity_y > 0):
        futureX = x_c
        futureY = y_c - delta_y
        case = "case7"
    elif ((angle > 90 - err) and (angle < 90 + err)) and ((velocity_x > 0 - err) and (velocity_x < 0 + err)) and (
            velocity_y < 0):
        futureX = x_c
        futureY = y_c + delta_y
        case = "case8"
    else:

            futureX = x_c
            futureY = y_c
            case = "case9"

            
    # visualizing the direction of movement
    if draw:
        print(f"Drawing path for id {id}")
        cv2.drawMarker(image, (x_c, y_c), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                       thickness=2)
        cv2.drawMarker(image, (int(futureX), int(futureY)), color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                       thickness=2)
        cv2.line(image, (x_c, y_c), (int(futureX), int(futureY)), (255, 255, 255), 2)
        cv2.putText(image, str(int(id)), (int(x_c+3), int(y_c-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(image, str(int(id)), (int(futureX-10), int(futureY-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        text=text+case
        cv2.putText(image, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return futureX, futureY, text, image

def get_features_foot(annotated_image,pose_landmarks,bbox_coords):
    
    R_Foot_Index_x , _,_ = get_landmark_coords_ratio(pose_landmarks, 32)
    L_Foot_Index_x , _,_ = get_landmark_coords_ratio(pose_landmarks, 31)
    R_Heel_x , _,_ = get_landmark_coords_ratio(pose_landmarks, 30)
    L_Heel_x , _,_ = get_landmark_coords_ratio(pose_landmarks, 29)
    R_Ankle_x , _,_ = get_landmark_coords_ratio(pose_landmarks, 28)
    L_Ankle_x , _,_ = get_landmark_coords_ratio(pose_landmarks, 27)
    Nose_x , Nose_y,_ = get_landmark_coords_ratio(pose_landmarks, 0)
     
    R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x,R_Ankle_x,L_Ankle_x, Nose_x , Nose_y=relative_to_absolute_pixel_8 (annotated_image, R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x,R_Ankle_x,L_Ankle_x,Nose_x, Nose_y, bbox_coords)
    
    return R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x,R_Ankle_x,L_Ankle_x, Nose_x, Nose_y
        
    
def get_features_ID3(image,pose_landmarks_current, pose_landmarks_previous,bbox_coords_current,bbox_coords_previous,t1,t2):
    # Extract the coordinates of the bounding box
     L_Hip_x1 , L_Hip_y1,_ = get_landmark_coords(pose_landmarks_previous, bbox_coords_previous, 23)
     R_Hip_x1 , R_Hip_y1,_ = get_landmark_coords(pose_landmarks_previous, bbox_coords_previous, 24)
     
     L_Hip_x2 , L_Hip_y2,_ = get_landmark_coords(pose_landmarks_current, bbox_coords_current, 23)
     R_Hip_x2 , R_Hip_y2,_ = get_landmark_coords(pose_landmarks_current, bbox_coords_current, 24)
     
     velocity_x_L=(abs(L_Hip_x2-L_Hip_x1))/(t2-t1)
     velocity_y_L=(abs(L_Hip_y2-L_Hip_y1))/(t2-t1)
     
     velocity_x_R=(abs(R_Hip_x2-R_Hip_x1))/(t2-t1)
     velocity_y_R=(abs(R_Hip_y2-R_Hip_y1))/(t2-t1)
     
     velocity_x=( velocity_x_R+ velocity_x_L)/2
     velocity_y=( velocity_y_R+ velocity_y_L)/2
     
     sh_left=sh_right=(0,0,0)
     
     x_l, y_l, _ = get_landmark_coords_ratio(pose_landmarks_current, 11)
     x_r, y_r, _ = get_landmark_coords_ratio(pose_landmarks_current, 12)
     

     x_l,y_l =  relative_to_absolute_pixel(image, x_l, y_l, bbox_coords_current)
     x_r,y_r =  relative_to_absolute_pixel(image, x_r, y_r, bbox_coords_current)
    
     
     sh_left = ( x_l, y_l,0)
     sh_right = ( x_r, y_r,0)
     
     angle=get_angle(image, sh_right, sh_left, bbox_coords_current)
     return velocity_x, velocity_y, angle

def get_center(bbox_coords):
    xmin,ymin,xmax,ymax=bbox_coords
    
    
    x= (xmin+xmax)/2
    y= (ymin+ymax)/2
    

    return x, y


def get_landmark_coords_ratio(pose_landmarks, landmark_id):

    # Check if pose landmarks are detected
    if pose_landmarks is not None:
        # Get the landmark object for the specified ID
        landmark = pose_landmarks.landmark[landmark_id]

        # Calculate the x and y pixel coordinates of the landmark
        x = landmark.x
        y = landmark.y
        z = landmark.z

        return x, y, z
    else:
        # Return None if no landmarks are detected
        return None, None,None


def get_landmark_coords(pose_landmarks, bbox_coords, landmark_id):
    # Get the dimensions of the input image
    xmin,ymin,xmax,ymax=bbox_coords
    width=xmax-xmin
    height=ymax-ymin

    # Check if pose landmarks are detected
    if pose_landmarks is not None:
        # Get the landmark object for the specified ID
        landmark = pose_landmarks.landmark[landmark_id]

        # Calculate the x and y pixel coordinates of the landmark
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        z = int(landmark.z * width)

        return x, y, z
    else:
        # Return None if no landmarks are detected
        return None, None,None


def color_bounding_box(image, bbox_coords, color):
    # Extract the coordinates of the bounding box
    xmin, ymin, xmax, ymax = bbox_coords
    
    # Create a mask for the bounding box
    mask = np.zeros_like(image)
    mask[ymin:ymax, xmin:xmax] = color
    
    # Apply the mask to the original image
    colored_image = cv2.add(image, mask)
    
    return colored_image


def is_corner_inside_trapezoid(bounding_box, trapezoid):
    # Extract coordinates of bounding box corners
    xmin, ymin, xmax, ymax = bounding_box

    # Extract coordinates of trapezoid vertices
    x1, y1, x2, y2, x3, y3, x4, y4 = trapezoid

    # Check if any corner of the bounding box is inside the trapezoid
    if is_point_inside_polygon(xmin, ymin, x1, y1, x2, y2, x3, y3, x4, y4) or \
       is_point_inside_polygon(xmin, ymax, x1, y1, x2, y2, x3, y3, x4, y4) or \
       is_point_inside_polygon(xmax, ymin, x1, y1, x2, y2, x3, y3, x4, y4) or \
       is_point_inside_polygon(xmax, ymax, x1, y1, x2, y2, x3, y3, x4, y4):
        return True
    
    return False

def is_point_inside_polygon(x, y, *vertices):
    n = len(vertices) // 2
    inside = False
    p1x, p1y = vertices[0], vertices[1]
    for i in range(1, n + 1):
        p2x, p2y = vertices[2*i % (2*n)], vertices[2*i % (2*n) + 1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def relative_to_absolute_pixel_8 (image, x1, x2, x3, x4, x5, x6, x7, y, bbox_coords_original):
    x_min,y_min , x_max, y_max = bbox_coords_original
    x1 = (x_min + (x_max - x_min) * x1) 
    x2 = (x_min + (x_max - x_min) * x2) 
    x3 = (x_min + (x_max - x_min) * x3) 
    x4 = (x_min + (x_max - x_min) * x4) 
    x5 = (x_min + (x_max - x_min) * x5) 
    x6 = (x_min + (x_max - x_min) * x6) 
    x7 = (x_min + (x_max - x_min) * x7) 
    y = (y_min + (y_max - y_min) * y) 
    return x1, x2, x3, x4, x5, x6, x7, y

def relative_to_absolute_pixel (image, x, y, bbox_coords_original):
    x_min, y_min, x_max, y_max = bbox_coords_original
    x = (x_min + (x_max - x_min) * x) 
    y = (y_min + (y_max - y_min) * y)
    return x,y

def relative_to_absolute(image, x, y, bbox_coords_original):
    x_min, y_min, x_max, y_max = bbox_coords_original
    x = (x_min + (x_max - x_min) * x) / image.shape[1]
    y = (y_min + (y_max - y_min) * y) / image.shape[0]
    return x,y


def visualize_person(image, results, bbox_coords_original):
    liste = results
    x_min_crop, y_min_crop, x_max_crop, y_max_crop = bbox_coords_original
    if liste is not None:
        for landmark in liste.landmark:
            landmark.x,  landmark.y = relative_to_absolute(image,  landmark.x ,  landmark.y , bbox_coords_original)
        mp_drawing.draw_landmarks(image, liste, mp_pose.POSE_CONNECTIONS)

    return image

def visualize(image, xmin, ymin, xmax, ymax, category_name, t):
    if (t==0):
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), ROW_SIZE)  # Change color to (255, 0, 0)
    else:
        if(t==1):
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), ROW_SIZE)  # Change color to (255, 0, 0)
    cv2.putText(
        image,
        category_name,
        (xmin, ymin - ROW_SIZE),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SIZE,
        TEXT_COLOR,
        FONT_THICKNESS
    )

    return image



def convert_detections_list(detection_result):
    dets = []
    for detection in detection_result.detections:
        category = detection.categories[0]
        xmin = int(detection.bounding_box.origin_x)
        ymin = int(detection.bounding_box.origin_y)
        bbox_width = detection.bounding_box.width
        bbox_height = detection.bounding_box.height
        aspect_ratio = bbox_width / bbox_height
        xmax = xmin + int(bbox_height * aspect_ratio)  # adjust xmax based on aspect ratio
        ymax = int(detection.bounding_box.origin_y) + bbox_height  # keep ymax same as before    
        
        if category.category_name == 'person':    
            bbox_coords = [xmin, ymin, xmax, ymax, 0.5]
            dets.append(bbox_coords)
    
    return np.array(dets)


def convert_detections_np(detection_result):
    dets = []
    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name == 'person':    
            xmin = int(detection.bounding_box.origin_x)
            ymin = int(detection.bounding_box.origin_y)
            bbox_width = detection.bounding_box.width
            bbox_height = detection.bounding_box.height
            aspect_ratio = bbox_width / bbox_height
            xmax = xmin + int(bbox_height * aspect_ratio)  # adjust xmax based on aspect ratio
            ymax = int(detection.bounding_box.origin_y) + bbox_height  # keep ymax same as before

            # Skip detections with NaN coordinates
            if any(np.isnan([xmin, ymin, xmax, ymax])):
                continue

            bbox_coords = [xmin, ymin, xmax, ymax, 0.5]
            dets.append(bbox_coords)

    return np.array(dets)
    
print('Processing started.......')



cap = cv2.VideoCapture("Jaad/video_0125.mp4")#Single detection zebra crossing
cap = cv2.VideoCapture("Jaad/video_0080.mp4")#Multiple detection zebra crossing


future=30
correction=True
skip=False


frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

# Create historical of pose estimation, bbox_coords and times 
pose_piles={}
filtered_pose_piles={}
# setting red area
ratiox1,ratioy1,ratiox2,ratioy2,ratiox3,ratioy3,ratiox4,ratioy4,path=set_dataset("jaad")
x1, y1, x2, y2, x3, y3, x4, y4,w_scale,h_scale=set_trapezoid_coords( ratiox1,ratioy1,ratiox2,ratioy2,ratiox3,ratioy3,ratiox4,ratioy4,w_scale=1200,ratio_h_w=(432/768))
trapezoid_coords= [x1, y1, x2, y2, x3, y3, x4, y4]


# Load the trained model pedestrian detector 
MODEL_PATH = 'efficientdet_lite0_uint8.tflite'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.3
    )
detector = vision.ObjectDetector.create_from_options(options)

# Load the trained model walking-standing
json_file = open('model_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("lstm_best_weights.h5")


# Create instance of SORT
mot_tracker = Sort()


# Create a MediaPipe Pose object
with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:


    pedestrians = []
    num_frame = 0
    desired_frame=1
    inference_frame=0.01477253516515096
    delay=0.0138941315015158
    with open("classification.txt", "w"):
        pass
    while True:
        start_time = time.time()
        num_frame += 1
        _, img = cap.read()
        if not(num_frame==desired_frame):
            continue
        
        if img is None:
            break
        # resizing the image to fit the frame
        img = cv2.resize(img, (w_scale, h_scale))    
        # flipping the image to get a real depiction of the scene
        img = cv2.flip(img, 1)    
        # resizing & adding a standard path overlay
        path = cv2.resize(path, (w_scale, h_scale))
        img_np2 = np.array(img, dtype=np.uint8)  # Convert to uint8 data type
        img = cv2.addWeighted(img,0.7,path,0.3,0)
        img = Image.fromarray(img)
    
        # Convert the image to numpy array
        img_np = np.array(img, dtype=np.uint8)
    
        # Create a MediaPipe Image object from the frame_rgb.
        image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np2)
            
        #Define location of alert message.
        stop_x=int(w_scale/2)
        stop_y=int(0.9*h_scale)
            
            
        detection_result = detector.detect(image2)
        
        annotated_image = np.copy(img_np)
        cv2.putText(annotated_image, f"Frame {num_frame} ", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),FONT_THICKNESS )

            # Convert detections to numpy array
        dets = convert_detections_np(detection_result)

        # Handle empty detections
        if len(dets) == 0:
            trackers = []
        else:
            # Update the tracker with detections
            trackers = mot_tracker.update(dets)
            
            
        # Get all IDs saved by the SORT mot_tracker
        all_ids = [d[-1] for d in trackers]
        all_ids_str = [str(int(id)) for id in all_ids]

        #print(f"All stored keys in all_ids_str: {all_ids_str}")
        # Filter the pose_piles dictionary to keep only the items with matching IDs
        filtered_pose_piles = {id: pose_piles[id] for id in all_ids_str if id in pose_piles}


        print(f"Calculation of frame number: {num_frame}")
        #print(f"All stored keys in filtered_pose_piles: {filtered_pose_piles.keys()}")
        #Draw bounding boxes and IDs, and pose estimation for each pedestrian and managmement pile 
        for d in trackers:
            pedestrian = {}
            xmin, ymin, xmax, ymax, id = d
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            B = (xmin, ymin, xmax, ymax)
            id =str(int(id))
            pedestrian['id'] = id
            pedestrian["pedestrian_class"] = "safe"
            pedestrian['bbox_coords'] = B
            pedestrians.append(pedestrian)
            collision = is_corner_inside_trapezoid(B, trapezoid_coords)
            if id in filtered_pose_piles:
                annotated_image = visualize(annotated_image, xmin, ymin, xmax, ymax, '', 1)

                if collision:
                    pedestrian["pedestrian_class"] = "danger"
                    annotated_image = color_bounding_box(annotated_image, [xmin, ymin, xmax, ymax], (0, 0, 255))
                    cv2.putText(annotated_image, "STOP", (stop_x - 50, stop_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                    print(f"The id {id}  exists in the previous data and is danger")
                else:
                    print(f"The id {id}  exists in the previous data- is not danger")  # t=1
            
            else:
                annotated_image = visualize(annotated_image, xmin, ymin, xmax, ymax, '', 0)
                if collision:
                    annotated_image = color_bounding_box(annotated_image, [xmin, ymin, xmax, ymax], (0, 0, 255))
                    print(f"The id {id} is danger and  not found in previous data")
                else:
                    print(f"The id {id} is safe and  not found in previous data")
            # Printing the id above bbx
            text=str(int(id))+":"
            # Pose estimation for each pedestrian detected
            width = xmax - xmin
            height = ymax - ymin

            #Expand the bounding box
            width_expansion = int(0.2 * width)
            height_expansion = int(0.2 * height)

            xmin2 = max(0, xmin - width_expansion)
            ymin2 = max(0, ymin - height_expansion)
            xmax2 = min(img_np.shape[1], xmax + width_expansion)
            ymax2 = min(img_np.shape[0], ymax + height_expansion)

            # Extract the sub-image +20% bbx
            sub_image = img_np[ymin2:ymax2, xmin2:xmax2]
            
            #Pose estimation
            if sub_image.size > 0:
                sub_image_rgb = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)
                results = pose.process(sub_image_rgb)
                # if results.pose_landmarks is not None:
                annotated_image = visualize_person(annotated_image, results.pose_landmarks, [xmin2, ymin2, xmax2, ymax2])
                filtered_pose_piles=management_pose_pile(filtered_pose_piles, id, results.pose_landmarks, B, time.time(), num_frame)
                pose_piles[id] = filtered_pose_piles[id].copy()
        # End Draw bounding boxes and IDs, and pose estimation and management pile
        # End for d in trackers
        
        
        
        
        future_danger = False
        # Predict pedestrian path of each pedestrian:
        #print(f"All stored keys in filtered_pose_piles: {filtered_pose_piles.keys()}")
        for id, info in filtered_pose_piles.items():
            if not info.get("new"):
                last_pose, last_bbx, last_time, last_frame = get_last_pile(filtered_pose_piles, id)
                print(f"We are trying to predict the path of id={id}")
                   
                last_before_pose, last_before_bbx, last_before_time, last_before_frame = get_data_before_last(filtered_pose_piles, id)
                velocity_x, velocity_y, angle=0,0,0
                missed_data=True
                if last_pose:
                    if last_before_pose:
                        if not (last_before_time == last_time):
                            velocity_x, velocity_y, angle = get_features_ID3(annotated_image, last_pose, last_before_pose, last_bbx, last_before_bbx, last_before_time, last_time)
                            missed_data=False
                        else:
                            print("last_before_time = last_time)")
                    else:
                        print("last before pose not available")
                    
                else:
                    print("last pose not available")
                    
                center_x, center_y = get_center(last_bbx)
                center_x, center_y = int(center_x), int(center_y)
                center = (center_x, center_y, 0)
                direction = ""
                if missed_data==False:
                    R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x, R_Ankle_x, L_Ankle_x, Nose_x, Nose_y = get_features_foot(annotated_image, last_pose, last_bbx)
                    direction = prediction_direction(R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x, R_Ankle_x, L_Ankle_x)
                x_text, y_text,_,_= last_bbx
                x_text, y_text = x_text, y_text - 10
                # Printing the id above bbx
                text=str(int(id))+":"
                if missed_data==False:
                    print("We can predict the path")
                    threshold=0.5
                    if correction==True:
                        futureX, futureY, text, annotated_image= futureXYD(annotated_image, center, angle, velocity_x, velocity_y, future, 5, True, x_text, y_text, pose_piles, id, direction, loaded_model,text,threshold)
                        get_rectified(text, num_frame)
                    else:
                        futureX, futureY, text, annotated_image= futureXY(annotated_image, center, angle, velocity_x, velocity_y, future, 5, True, x_text, y_text, pose_piles, id, direction, loaded_model,text,threshold)
                    #delta=get_delta(pose_piles,int(id),3)
                    #futureX, futureY= futureXYP(delta,pose_piles,int(id),3)
                else:
                    print("We can not predict the path beacause of missed data")
                    cv2.putText(annotated_image, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    futureX, futureY = center_x, center_y
                bbox_coords_future = predict_bbox_coords_future(futureX, futureY, last_bbx)
                predict_collusion = False
                for pedestrian in pedestrians:
                    if pedestrian['id'] == id:
                        if pedestrian['pedestrian_class'] == "safe":
                            segment_line = (center_x, center_y, futureX, futureY)
                            predict_collusion = is_segment_intersect(segment_line, trapezoid_coords)
                            if predict_collusion:
                                pedestrian['pedestrian_class'] = "future danger"
                                annotated_image = color_bounding_box(annotated_image, [xmin, ymin, xmax, ymax], (0, 242, 255))
                                future_danger = True
                                print(f"The id {id} is checked and we found it future danger (segment)")
                            else:
                                predict_collusion = is_corner_inside_trapezoid(bbox_coords_future, trapezoid_coords)
                                if predict_collusion:
                                    pedestrian['pedestrian_class'] = "future danger"
                                    annotated_image = color_bounding_box(annotated_image, [xmin, ymin, xmax, ymax], (0, 242, 255))
                                    future_danger = True
                                    print(f"The id {id} is checked and we found it future danger (bbx)")
            
                        break  # Exit the loop once the matching ID is found        
        
        #Marking all new detections as not new detections
        for id, info in pose_piles.items():
            if  info.get("new"):
                pose_piles[id]["new"]=False
         # Display the annotated image
        print(f"Showing frame number: {num_frame}")
        cv2.imshow('output', annotated_image)
        #input ("press enter to continue...")

        with open("classification.txt", "a") as f:
            f.write(f"Number of frame: {num_frame}\n")
            # Loop through each dictionary in my_dicts and write its content to the file
            for pedestrian in pedestrians:
                f.write(f"ID: {pedestrian['id']}\n")
                f.write(f"Pedestrian class: {pedestrian['pedestrian_class']}\n")
                f.write(f"Bbox coordinates: {pedestrian['bbox_coords']}\n")
                f.write("\n")  # Add an empty line between each entry
              
        pedestrians = []
        
       # input("Hello")
        key = cv2.waitKey(1)
        if key == 27:  # Check if 'Esc' key is pressed
            break
        end_time = time.time()
        delta=end_time-start_time
        skip_frames=int(delta/(inference_frame+delay))
        if skip==False:
            skip_frames=1
        desired_frame=num_frame+skip_frames
        print("delta: ",delta)
        print("skip_frames: ",skip_frames)
        print("desired_frame: ",desired_frame)
        time.sleep(delay)
      #  input('Hello')

with open('rectified.txt', 'a') as file:
    file.write(f"Toatle frame: {num_frame}\n")

cap.release()
cv2.destroyAllWindows()
detector.close()
print('\nProcessing completed.......!!!')







