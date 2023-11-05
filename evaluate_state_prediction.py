import os
import cv2
import xml.etree.ElementTree as ET
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import time
import math
from scipy.interpolate import interp1d

def get_time_of_frame(pose_piles,n, id):
    last_frame_time=0
    frame_time = pose_piles[id]["time"][-1-n] if pose_piles[id]["time"] else None
    last_frame_time= pose_piles[id]["time"][-1-n-1] if pose_piles[id]["time"] else None
    delta=frame_time-last_frame_time
    return delta

def get_step_to_past(pose_piles,period , id):
    n=1
    delta=0
    print ("len(pose_piles[id][time])=",len(pose_piles[id]["time"]))
    while (delta<=period) and n<16:
        print ("n=",n)
        if len(pose_piles[id]["time"])>n+1 :
            delta+=get_time_of_frame(pose_piles,n , id)
            
        n+=1
        
    if(n>16):
        n-=1
    print (delta)
    return n,delta
def squared_error(predicted_point, ground_truth_point):
    # Extract x and y coordinates of the predicted point
    pred_x, pred_y = predicted_point[0], predicted_point[1]
    
    # Extract x and y coordinates of the ground truth point
    gt_x, gt_y = ground_truth_point[0], ground_truth_point[1]
    
    # Calculate squared differences of x and y coordinates
    se_x = (pred_x - gt_x) ** 2
    se_y = (pred_y - gt_y) ** 2
    
    # Calculate Squared Error
    squared_error = se_x + se_y
    
    return squared_error

def euclidean_distance(point1, point2):
    x1, y1,_ = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_last_pile(pose_piles, id):
    if id not in pose_piles:
        return None, None, None, None, None



    last_pose = pose_piles[id]["pile"][-1] if pose_piles[id]["pile"] else None
    last_bbx = pose_piles[id]["bbx"][-1] if pose_piles[id]["bbx"] else None
    last_time = pose_piles[id]["time"][-1] if pose_piles[id]["time"] else None
    last_frame = pose_piles[id]["frame"][-1] if pose_piles[id]["frame"] else None

    return last_pose, last_bbx, last_time, last_frame

def get_data_before_last(pose_piles, id,step):
    if id not in pose_piles:
        return None

    # Retrieve the data just before the last entry
    pose_before_last = pose_piles[id]["pile"][-1-step] if len(pose_piles[id]["pile"]) >= step else None
    bbx_before_last = pose_piles[id]["bbx"][-1-step] if len(pose_piles[id]["bbx"]) >= step else None
    time_before_last = pose_piles[id]["time"][-1-step] if len(pose_piles[id]["time"]) >= step else None
    frame_before_last = pose_piles[id]["frame"][-1-step] if len(pose_piles[id]["frame"]) >= step else None
    state_before_last = pose_piles[id]["state"][-1-step] if len(pose_piles[id]["state"]) >= step else None
    return pose_before_last, bbx_before_last, time_before_last, frame_before_last, state_before_last

def management_pose_pile(pose_piles, id, pose_landmarks, bbox_coords, time, frame,action):
    if id not in pose_piles:
        pose_piles[id] = {"state": [],"pile": [], "bbx": [], "time": [], "frame": [], "new": True}
    
    pose_data = pose_piles[id]["pile"]
    bbx_data = pose_piles[id]["bbx"]
    time_data = pose_piles[id]["time"]
    frame_data = pose_piles[id]["frame"]
    state_data = pose_piles[id]["state"]
    
    # Add new record 
    if pose_landmarks is not None:
        pose_data.append(pose_landmarks)    
    bbx_data.append(bbox_coords)
    time_data.append(time)
    frame_data.append(frame)
    state_data.append(action)
    
    # Ensure data has 16 records
    if pose_landmarks is not None:
        while len(pose_data) < 32:
            pose_data.insert(0, pose_landmarks)        
    while len(bbx_data) < 32:
        bbx_data.insert(0, bbox_coords)  
    while len(time_data) < 32:
        time_data.insert(0, time)  
    while len(frame_data) < 32:
        frame_data.insert(0, frame)  
    while len(state_data) < 32:
        state_data.insert(0, action)  
        
        # Ensure data has 16 records
    while len(pose_data) > 32:
        pose_data.pop(0)
    while len(bbx_data) > 32:
        bbx_data.pop(0)
    while len(time_data) > 32:
        time_data.pop(0)
    while len(frame_data) > 32:
        frame_data.pop(0)
    while len(state_data) > 32:
        state_data.pop(0)
        
    return pose_piles


def prediction_direction(R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x,R_Ankle_x,L_Ankle_x):
    score = 0
    R_Foot_Index_xx,R_Foot_Index_v=R_Foot_Index_x
    L_Foot_Index_xx,L_Foot_Index_v=L_Foot_Index_x
    R_Heel_xx,R_Heel_v=R_Heel_x
    R_Heel_xx,R_Heel_v=R_Heel_x
    L_Heel_xx,L_Heel_v=L_Heel_x
    R_Ankle_xx,R_Ankle_v=R_Ankle_x
    L_Ankle_xx,L_Ankle_v=L_Ankle_x
    
    if(R_Foot_Index_v<0.5) or (L_Foot_Index_v<0.5) or (R_Heel_v<0.5) or (L_Heel_v<0.5) or (R_Ankle_v<0.5) or (L_Ankle_v<0.5):
        return 'NV'
    if R_Foot_Index_xx > R_Ankle_xx:
        score += 1
    else:
        score -= 1
        
    if R_Foot_Index_xx > R_Heel_xx:
        score += 1
    else:
        score -= 1
        
    if L_Foot_Index_xx > L_Ankle_xx:
        score += 1
    else:
        score -= 1
        
    if L_Foot_Index_xx > L_Heel_xx:
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
    kind=""
    if (n==1):
        kind="linear"
    elif (n==2):
        kind="quadratic"
    elif (n==3):
        kind="cubic"
        
    # Create interpolation functions for x and y coordinates based on time (t)
    x_interp_func = interp1d(t_values, x_coords, kind=kind)
    y_interp_func = interp1d(t_values, y_coords, kind=kind)

    return x_interp_func, y_interp_func

def futureXYP(timeToFuture,pose_piles,id,n):
    x_interp_func, y_interp_func = polynomial_interpolation(pose_piles, id, n)
    # Ensure that timeToFuture is within the interpolation range
    if timeToFuture < 0:
        timeToFuture = 0
    elif timeToFuture > n:
        timeToFuture = n
    future_position_x = x_interp_func(timeToFuture)
    future_position_y = y_interp_func(timeToFuture)
    return future_position_x, future_position_y
    
def futureXY_GT_State(image, initial, angle, velocity_x, velocity_y, timeToFuture, err, draw, x_text, y_text,pose_piles,id,direction,loaded_model,text,threshold,step,state):
    x_c, y_c, _ = initial
    x_c = int(x_c)
    y_c = int(y_c)
    
    delta_x = np.math.sqrt(((velocity_x * timeToFuture) * np.math.cos(angle)) ** 2)
    delta_y = np.math.sqrt(((velocity_y * timeToFuture) * np.math.cos(angle)) ** 2)

    text_state=""
        
    cv2.putText(image, text_state, (x_text, (y_text-23)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,cv2.LINE_AA)    
    case = "No case detected"

    if (angle > 0) and (angle < 90) and (velocity_x > 0) and (velocity_y < 0):
        case = "case1"
        futureX = x_c + delta_x
        futureY = y_c - delta_y
    elif (angle > 0) and (angle < 90) and (velocity_x < 0) and (velocity_y > 0):
        case = "case2"
        futureX = x_c - delta_x
        futureY = y_c + delta_y
    elif (angle > 90) and (angle < 180) and (velocity_x > 0) and (velocity_y > 0):
        case = "case3"
        futureX = x_c + delta_x
        futureY = y_c + delta_y
        case = "case3"
    elif (angle > 90) and (angle < 180) and (velocity_x < 0) and (velocity_y < 0):
        case = "case4"
        futureX = x_c - delta_x
        futureY = y_c - delta_y
    elif (((angle > 0 - err) and (angle < 0 + err)) or ((angle > 180 - err) and (angle < 180 + err))) and (
            velocity_x > 0) and ((velocity_y > 0 - err) and (velocity_y < 0 + err)):
        case = "case5"
        futureX = x_c + delta_x
        futureY = y_c
    elif (((angle > 0 - err) and (angle < 0 + err)) or ((angle > 180 - err) and (angle < 190 - err))) and (
            velocity_x < 0) and ((velocity_y > 0 - err) and (velocity_y < 0 + err)):
        case = "case6"
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
        if state==0:
            futureX = x_c
            futureY = y_c
            case = "case9"
        else:
            case = "case9 rectified"
            if direction=="LR":
                futureX = x_c+ delta_x
            elif direction=="RL":
                futureX = x_c-delta_x
            elif direction=="N":  
                case = "case9 not rectified (not coherent orientation of foot)"
                futureX=x_c
            elif direction=="NV":     
                case = "case9 not rectified (not visibility of foot)"
                futureX=x_c
                                
            futureY = y_c
           
    # visualizing the direction of movement
    if draw:
        cv2.drawMarker(image, (x_c, y_c), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                       thickness=2)
        
        cv2.drawMarker(image, (int(futureX), int(futureY)), color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                       thickness=2)

        cv2.line(image, (x_c, y_c), (int(futureX), int(futureY)), (255, 255, 255), 2)

        
        text=text+case
        cv2.putText(image, text, (x_text, y_text-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return futureX, futureY, text


def futureXY(image, initial, angle, velocity_x, velocity_y, timeToFuture, err, draw, x_text, y_text,pose_piles,id,text):
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
            case=case
            
    # visualizing the direction of movement
    if draw:
        cv2.drawMarker(image, (x_c, y_c), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                       thickness=2)
        cv2.drawMarker(image, (int(futureX), int(futureY)), color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                       thickness=2)
        cv2.line(image, (x_c, y_c), (int(futureX), int(futureY)), (255, 255, 255), 2)

    return futureX, futureY, text

def futureXYD(image, initial, angle, velocity_x, velocity_y, timeToFuture, err, draw, x_text, y_text,pose_piles,id,direction,loaded_model,text,threshold,step):
    x_c, y_c, _ = initial
    x_c = int(x_c)
    y_c = int(y_c)
    
    delta_x = np.math.sqrt(((velocity_x * timeToFuture) * np.math.cos(angle)) ** 2)
    delta_y = np.math.sqrt(((velocity_y * timeToFuture) * np.math.cos(angle)) ** 2)

    state_pedestrian, probab = get_state_pedestrian_before(pose_piles, id,loaded_model,threshold,step)
    text_state=""
    if(state_pedestrian==1):
        text_state="Walking:"+" "+str(int(probab*100))+"%"
    else:
        text_state="Standing:"+" "+str(int(probab*100))+"%"
        
    cv2.putText(image, text_state, (x_text, (y_text-23)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,cv2.LINE_AA)    
    case = "No case detected"

    if (angle > 0) and (angle < 90) and (velocity_x > 0) and (velocity_y < 0):
        case = "case1"
        futureX = x_c + delta_x
        futureY = y_c - delta_y
    elif (angle > 0) and (angle < 90) and (velocity_x < 0) and (velocity_y > 0):
        case = "case2"
        futureX = x_c - delta_x
        futureY = y_c + delta_y
    elif (angle > 90) and (angle < 180) and (velocity_x > 0) and (velocity_y > 0):
        case = "case3"
        futureX = x_c + delta_x
        futureY = y_c + delta_y
        case = "case3"
    elif (angle > 90) and (angle < 180) and (velocity_x < 0) and (velocity_y < 0):
        case = "case4"
        futureX = x_c - delta_x
        futureY = y_c - delta_y
    elif (((angle > 0 - err) and (angle < 0 + err)) or ((angle > 180 - err) and (angle < 180 + err))) and (
            velocity_x > 0) and ((velocity_y > 0 - err) and (velocity_y < 0 + err)):
        case = "case5"
        futureX = x_c + delta_x
        futureY = y_c
    elif (((angle > 0 - err) and (angle < 0 + err)) or ((angle > 180 - err) and (angle < 190 - err))) and (
            velocity_x < 0) and ((velocity_y > 0 - err) and (velocity_y < 0 + err)):
        case = "case6"
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
        if state_pedestrian==0:
            futureX = x_c
            futureY = y_c
            case = "case9"
        else:
            case = "case9 rectified"
            if direction=="LR":
                futureX = x_c+ delta_x
            elif direction=="RL":
                futureX = x_c-delta_x
            elif direction=="N":  
                case = "case9 not rectified (not coherent orientation of foot)"
                futureX=x_c
            elif direction=="NV":     
                case = "case9 not rectified (not visibility of foot)"
                futureX=x_c
                                
            futureY = y_c
           
    # visualizing the direction of movement
    if draw:
        cv2.drawMarker(image, (x_c, y_c), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                       thickness=2)
        
        cv2.drawMarker(image, (int(futureX), int(futureY)), color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                       thickness=2)

        cv2.line(image, (x_c, y_c), (int(futureX), int(futureY)), (255, 255, 255), 2)

        
        text=text+case
        cv2.putText(image, text, (x_text, y_text-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return futureX, futureY, text
def get_features_foot(annotated_image,pose_landmarks,bbox_coords):
    
    R_Foot_Index_x , _,_, R_Foot_Index_v  = get_landmark_coords_ratio(pose_landmarks, 32)
    L_Foot_Index_x , _,_, L_Foot_Index_v  = get_landmark_coords_ratio(pose_landmarks, 31)
    R_Heel_x , _,_, R_Heel_v = get_landmark_coords_ratio(pose_landmarks, 30)
    L_Heel_x , _,_, L_Heel_v  = get_landmark_coords_ratio(pose_landmarks, 29)
    R_Ankle_x , _,_, R_Ankle_v  = get_landmark_coords_ratio(pose_landmarks, 28)
    L_Ankle_x , _,_, L_Ankle_v = get_landmark_coords_ratio(pose_landmarks, 27)
    Nose_x , Nose_y,_, _  = get_landmark_coords_ratio(pose_landmarks, 0)
     
    R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x,R_Ankle_x,L_Ankle_x, Nose_x , Nose_y=relative_to_absolute_pixel_8 (annotated_image, R_Foot_Index_x, L_Foot_Index_x, R_Heel_x, L_Heel_x,R_Ankle_x,L_Ankle_x,Nose_x, Nose_y, bbox_coords)
    
    return (R_Foot_Index_x,R_Foot_Index_v) , (L_Foot_Index_x,L_Foot_Index_v), (R_Heel_x,R_Heel_v), (L_Heel_x,L_Heel_v),(R_Ankle_x,R_Ankle_v),(L_Ankle_x,L_Ankle_v), Nose_x, Nose_y
  
    
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
     
     x_l, y_l, _ , _= get_landmark_coords_ratio(pose_landmarks_current, 11)
     x_r, y_r, _ , _= get_landmark_coords_ratio(pose_landmarks_current, 12)
     

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
        v = landmark.visibility

        return x, y, z, v
    else:
        # Return None if no landmarks are detected
        return None, None,None, None


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

def get_state_pedestrian_before(pose_piles, id, loaded_model, threshold,step):
    pose_data = pose_piles[id]["pile"]
    
    # Initialize an empty list to store the tensor data (X) 
    tensor_data = []
    
    # Iterate through each pose estimation in pose_data
    k=j=0
    for pose_estimation in pose_data:
        
        if(k<step):
            k+=1
            continue
        pose_coords_arr = []

        # Extract the x and y coordinates for each landmark (considering 14 landmarks)
        for i in range(33):
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 23, 24]:
                continue

            x, y, _, _ = get_landmark_coords_ratio(pose_estimation, i)
            
            # Append the (x, y) coordinate to pose_coords_arr
            pose_coords_arr.append([x, y])
        j+=1
        # Convert pose_coords_arr to a NumPy array to create the 3D tensor (14, 2) for the pose estimation
        pose_tensor = np.array(pose_coords_arr)

        # Append the pose_tensor to the list
        tensor_data.append(pose_tensor)
        if(j==16):
            break;

    # Convert the list of tensors to a NumPy array to create the 4D tensor (num_elements, 14, 2)
    X = np.array(tensor_data)

    X = tf.reshape(X, shape=(-1, 16, 28))


    predicted_probabilities = loaded_model.predict(X)
    
    predicted_class = 1 if predicted_probabilities > threshold else 0
    predicted_probabilities = predicted_probabilities[0][0] if predicted_probabilities > threshold else 1-predicted_probabilities[0][0]
    return predicted_class, predicted_probabilities



def get_state_pedestrian(pose_data, loaded_model, threshold):
    # Initialize an empty list to store the tensor data (X)
    tensor_data = []
    # Iterate through each pose estimation in pose_data
    j=0
    for pose_estimation in pose_data:
        j+=1
        pose_coords_arr = []
        
        # Extract the x and y coordinates for each landmark (considering 14 landmarks)
        for i in range(14):
            if len(pose_estimation) < (i + 1) * 2:
                # If the length is less than (i + 1) * 2, skip this pose estimation
                break

            x = pose_estimation[i * 2]
            y = pose_estimation[i * 2 + 1]
            pose_coords_arr.extend([x, y])

        # Check if the pose_coords_arr has the expected length (28 = 14 landmarks * 2 coordinates)
        if len(pose_coords_arr) == 28:
            # Append the pose_coords_arr to the list
            tensor_data.append(pose_coords_arr)

    # Convert the list of tensors to a NumPy array to create the 4D tensor (num_elements, 14, 2)
    X = np.array(tensor_data)

    X = tf.reshape(X, shape=(-1, 16, 28))
    predicted_probabilities = loaded_model.predict(X)

    predicted_class = 1 if predicted_probabilities > threshold else 0
    predicted_probabilities = predicted_probabilities[0][0] if predicted_probabilities > threshold else 1-predicted_probabilities[0][0]
    return predicted_class, predicted_probabilities



def relative_to_absolute(image, x, y, bbox_coords_original):
    x_min, y_min, x_max, y_max = bbox_coords_original
    x = (x_min + (x_max - x_min) * x) / image.shape[1]
    y = (y_min + (y_max - y_min) * y) / image.shape[0]
    return x,y


def visualize_person(image, results, bbox_coords_original,mp_drawing,mp_pose):
    liste = results
    x_min_crop, y_min_crop, x_max_crop, y_max_crop = bbox_coords_original
    if liste is not None:
        for landmark in liste.landmark:
            landmark.x,  landmark.y = relative_to_absolute(image,  landmark.x ,  landmark.y , bbox_coords_original)
        mp_drawing.draw_landmarks(image, liste, mp_pose.POSE_CONNECTIONS)

    return image


video_folder = "/content/drive/MyDrive/JAAD"

video_folder = "M:\Memoire master\PCPTree\Object Detector\MobileNet\JAAD"


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
    
    # Load the trained model walking-standing
json_file = open('model_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("lstm_best_weights.h5")
    
   # mp_holistic = mp.solutions.holistic
with mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
    # Open the text file for reading
    with open('check_point.txt', 'r') as file:
        # Read each line and assign values to variables
        nb_occlusion = int(file.readline().strip())
        sum_tp_occ = int(file.readline().strip())
        sum_fp_occ = int(file.readline().strip())
        sum_fn_occ = int(file.readline().strip())
        sum_tn_occ = int(file.readline().strip())
        
        
        sum_tp = int(file.readline().strip())
        sum_fp = int(file.readline().strip())
        sum_fn = int(file.readline().strip())
        sum_tn = int(file.readline().strip())
        
        nb_prediction = int(file.readline().strip())
        nb_prediction2 = int(file.readline().strip())
        nb_prediction3 = int(file.readline().strip()) 
        
        total_distance = float(file.readline().strip())
        total_distance2 = float(file.readline().strip())
        total_distance3 = float(file.readline().strip())
        
        total_distanceD = float(file.readline().strip())
        total_distance2D = float(file.readline().strip())
        total_distance3D = float(file.readline().strip())
        
        total_distanceD_GT = float(file.readline().strip())
        total_distanceD2_GT = float(file.readline().strip())
        total_distanceD3_GT = float(file.readline().strip())
        
        total_distanceSE = float(file.readline().strip())
        total_distanceSE2 = float(file.readline().strip())
        total_distanceSE3 = float(file.readline().strip())
        
        total_distanceSED = float(file.readline().strip())
        total_distanceSE2D = float(file.readline().strip())
        total_distanceSE3D = float(file.readline().strip())
        
        
        total_distanceSED_GT = float(file.readline().strip())
        total_distanceSED2_GT = float(file.readline().strip())
        total_distanceSED3_GT = float(file.readline().strip())
        
        v = int(file.readline().strip())

        
        num_v=0
        for video_file in os.listdir(video_folder):
            num_v+=1
            if(num_v<v):
                continue
            if video_file.endswith(".mp4") and not(video_file=="fus1.mp4"):
                video_path = os.path.join(video_folder, video_file)
                annotation_path = os.path.join(video_folder, video_file.replace(".mp4", ".xml"))
                print (video_path)
                # Parse the XML annotation file
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                track_elements = root.findall(".//track[@label='pedestrian']")
                
                # We will work only with the first 'pedestrian' track
                if track_elements:
                    i=0
                    for pedestrian_track in track_elements:
                        i+=1
                        pose_piles={}
                        boxes = pedestrian_track.findall(".//box")
                        # Open the video to iterate through frames
                        cap = cv2.VideoCapture(video_path)
        
                        # Get original video resolution
                        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
                        # Define the desired resized width and height
                        desired_width = 1200
                        desired_height = 600
        
                        # Calculate the scale factors for resizing
                        scale_x = desired_width / original_width
                        scale_y = desired_height / original_height
                        num_frame=0
                        total_inference_time=0
                        while cap.isOpened():
                            initial_t=time.time()
                            num_frame+=1
                            ret, frame = cap.read()
                            if not ret:
                                break
        
                            # Resize the frame
                            #frame = cv2.resize(frame, (desired_width, desired_height))
        
                            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Get current frame number (0-based index)
        
                            # Check if the frame number has corresponding annotation
                            matching_boxes = [box for box in boxes if int(box.get("frame")) == frame_number]
                            scale_x=scale_y=1
                            for box in matching_boxes:
                                xtl = int(float(box.get("xtl")) * scale_x)
                                ytl = int(float(box.get("ytl")) * scale_y)
                                xbr = int(float(box.get("xbr")) * scale_x)
                                ybr = int(float(box.get("ybr")) * scale_y)
                                occluded = int(float(box.get("occluded")))
        
                                # Process the 'action' attribute to get the state of pedestrian  
                                action = box.find(".//attribute[@name='action']").text.strip().lower()
        
                                # Filtering the extracted state (e.g., only consider 'walking' or 'standing')
                                    # Display the bounding box
                                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
                                cv2.putText(frame, f"G.Truth: {action}", (xtl, ytl - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                # Pose estimation for each pedestrian detected
                                xmin,ymin,xmax,ymax=xtl,ytl,xbr,ybr
                                B = (xmin, ymin, xmax, ymax)
                                img_np = np.array(frame, dtype=np.uint8)
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
                                    frame = visualize_person(frame, results.pose_landmarks, [xmin2, ymin2, xmax2, ymax2],mp_drawing,mp_pose)
                                    pose_piles=management_pose_pile(pose_piles, 1.0 , results.pose_landmarks, B, time.time(), num_frame,action)
                                    # Check if pose landmarks are detected
                                    if results.pose_landmarks is not None:

                                            threshold=0.5
                                            state_pedestrian, probab = get_state_pedestrian_before(pose_piles,1.0,loaded_model,threshold,0)
    
                                            if(action=="standing"):#actual positive
                                                if (state_pedestrian==0):
                                                    sum_tp += 1
                                                else:                    
                                                    sum_fn += 1
                                                            
                                            if(action=="walking"):#actual negative
                                                if (state_pedestrian==0):#predicted positive
                                                    sum_fp += 1
                                                else:                   #predicted negative
                                                    sum_tn += 1    
                                                    
                                            if(occluded==0):
                                                if(action=="standing"):#actual positive
                                                    if (state_pedestrian==0):
                                                        sum_tp_occ += 1
                                                    else:                    
                                                        sum_fn_occ += 1
                                                                
                                                if(action=="walking"):#actual negative
                                                    if (state_pedestrian==0):#predicted positive
                                                        sum_fp_occ += 1
                                                    else:                   #predicted negative
                                                        sum_tn_occ += 1                                          
                                        #prediction path
    
                                    period=0.6
                                    print("Frame number: ",frame_number)
                                    step,_=get_step_to_past(pose_piles,period, 1)
                                    period2=1
                                    step2,_=get_step_to_past(pose_piles,period2, 1)
                                    period3=1.5
                                    step3,_=get_step_to_past(pose_piles,period3, 1)
                                        
                                    if (step>16):
                                        step=16
                                        
                                    if (step2>16):
                                        step2=16
                                        
                                    if (step3>16):
                                        step3=16
    
    
                                    last_pose, last_bbx, last_time, last_frame = get_last_pile(pose_piles, 1)                                      
        
                                                                        
                                    last_before_pose, last_before_bbx, last_before_time, last_before_frame,state = get_data_before_last(pose_piles, 1,step)
                                    last_before_pose2, last_before_bbx2, last_before_time2, last_before_frame2,state2 = get_data_before_last(pose_piles, 1,step2)
                                    last_before_pose3, last_before_bbx3, last_before_time3, last_before_frame3,state3 = get_data_before_last(pose_piles, 1,step3)
    
    
    
                                    velocity_x, velocity_y, angle=0,0,0
                                    missed_data=True
                                    if last_pose:
                                        print ("last_pose")
                                        if last_before_pose:
                                            print ("last_before_poseeeeeeeeeeeeeeeeeeeeeee")
                                            if not (last_before_time == last_time):
                                                velocity_x, velocity_y, angle = get_features_ID3(frame, last_pose, last_before_pose, last_bbx, last_before_bbx, last_before_time, last_time)
                                                missed_data=False
                                                print ("time")
    
                                    velocity_x2, velocity_y2, angle2=0,0,0
                                    missed_data2=True
                                    if last_before_pose2:
                                        if not (last_before_time2 == last_time):
                                            velocity_x2, velocity_y2, angle2 = get_features_ID3(frame, last_pose, last_before_pose2, last_bbx, last_before_bbx2, last_before_time2, last_time)
                                            missed_data2=False
    
                                    velocity_x3, velocity_y3, angle3=0,0,0
                                    missed_data3=True
                                    if last_before_pose3:
                                        if not (last_before_time3 == last_time):
                                            velocity_x3, velocity_y3, angle3 = get_features_ID3(frame, last_pose, last_before_pose3, last_bbx, last_before_bbx3, last_before_time3, last_time)
                                            missed_data3=False
    
                                                    
                                    center_x, center_y = get_center(last_before_bbx)
                                    center_x, center_y = int(center_x), int(center_y)
                                    center = (center_x, center_y, 0)
                                        
                                                    
                                    center_x2, center_y2 = get_center(last_before_bbx2)
                                    center_x2, center_y2 = int(center_x2), int(center_y2)
                                    center2 = (center_x2, center_y2, 0)
    
                                        
                                    center_x3, center_y3 = get_center(last_before_bbx3)
                                    center_x3, center_y3 = int(center_x3), int(center_y3)
                                    center3 = (center_x3, center_y3, 0)
    
    
                                    direction = ""
                                    if missed_data==False:
                                        R_F_I_x, L_F_I_x, R_H_x, L_H_x, R_A_x, L_A_x, Nose_x, Nose_y = get_features_foot(frame, last_pose, last_bbx)
                                        direction = prediction_direction(R_F_I_x, L_F_I_x, R_H_x, L_H_x, R_A_x, L_A_x)
                                        if not(direction=="N") and  not(direction=="NV"):
                                            xx=int(xmin+(xmax-xmin)/2)
                                            yy=ymin-0
                                            start_point = (xx, yy)
                                            if(direction=="LR"):
                                                end_point = (xmax, yy)   
                                            else:
                                                end_point = (xmin, yy)   
    
                                            colorA = (0, 0, 255)
                                            thicknessA = 2
                                            tip_lengthA = 0.5
                                            frame = cv2.arrowedLine(frame, start_point, end_point, colorA, thicknessA,tip_length =tip_lengthA)
                                                                                        
                                    x_text, y_text = xmin, ymin - 10
                                    text="1:"                                    
                                        
                                direction2 = ""
                                if missed_data2==False:
                                    R_F_I_x2, L_F_I_x2, R_H_x2, L_H_x2, R_A_x2, L_A_x2, _, _ = get_features_foot(frame, last_pose, last_bbx)
                                    direction2 = prediction_direction(R_F_I_x2, L_F_I_x2, R_H_x2, L_H_x2, R_A_x2, L_A_x2)
                                            
                                direction3 = ""
                                if missed_data3==False:
                                    R_F_I_x3, L_F_I_x3, R_H_x3, L_H_x3, R_A_x3, L_A_x3, _, _ = get_features_foot(frame, last_pose, last_bbx)
                                    direction3 = prediction_direction(R_F_I_x3, L_F_I_x3, R_H_x3, L_H_x3, R_A_x3, L_A_x3)
    
                                threshold=0.5
                                if missed_data==False: 
                                    futureX1, futureY1, text = futureXY(frame, center, angle, velocity_x, velocity_y, period, 5, True, x_text, y_text, pose_piles, 1.0 ,text)
                                    futureXD1, futureYD1, text = futureXYD(frame, center, angle, velocity_x, velocity_y,  period, 5, True, x_text, y_text, pose_piles, 1.0, direction, loaded_model,text,threshold,step)
                                    futureXD_GT1, futureYD_GT1, text = futureXY_GT_State(frame, center, angle, velocity_x, velocity_y,  period, 5, True, x_text, y_text, pose_piles, 1.0, direction, loaded_model,text,threshold,step,state)


    
                                if missed_data2==False:
                                    futureX2, futureY2, text = futureXY(frame, center2, angle2, velocity_x2, velocity_y2, period2, 5, True, x_text, y_text, pose_piles, 1.0 ,text)
                                    futureXD2, futureYD2, text = futureXYD(frame, center2, angle2, velocity_x2, velocity_y2,  period2, 5, True, x_text, y_text, pose_piles, 1.0, direction, loaded_model,text,threshold,step2)
                                    futureXD_GT2, futureYD_GT2, text = futureXY_GT_State(frame, center2, angle2, velocity_x2, velocity_y2,  period2, 5, True, x_text, y_text, pose_piles, 1.0, direction, loaded_model,text,threshold,step2,state2)
    
                                if missed_data3==False:
                                    futureX3, futureY3, text = futureXY(frame, center3, angle3, velocity_x3, velocity_y3, period3, 5, True, x_text, y_text, pose_piles, 1.0 ,text)
                                    futureXD3, futureYD3, text = futureXYD(frame, center3, angle3, velocity_x3, velocity_y3,  period3, 5, True, x_text, y_text, pose_piles, 1.0, direction, loaded_model,text,threshold,step3)
                                    futureXD_GT3, futureYD_GT3, text = futureXY_GT_State(frame, center3, angle3, velocity_x3, velocity_y3,  period3, 5, True, x_text, y_text, pose_piles, 1.0, direction, loaded_model,text,threshold,step3,state3)
    
                                #Get the ground truth point
                                gt_x,gt_y = get_center(last_bbx)
                                gt_x, gt_y = int(gt_x), int(gt_y)
                                ground_truth=( gt_x, gt_y)
                              
                                if missed_data==False: 
                                    future=(futureX1, futureY1)
                                    futureD=(futureXD1, futureYD1)
                                    futureDGT1=(futureXD_GT1, futureYD_GT1)
        
                                if missed_data2==False:             
                                    future2=(futureX2, futureY2)
                                    futureD2=(futureXD2, futureYD2)
                                    futureDGT2=(futureXD_GT2, futureYD_GT2)
    
                                if missed_data3==False: 
                                    future3=(futureX3, futureY3)
                                    futureD3=(futureXD3, futureYD3)
                                    futureDGT3=(futureXD_GT3, futureYD_GT3)
    
    
                           #0.5 sec                 
                                   #ID3 0.5 sec
                                if missed_data==False: 
                                    distance1 = euclidean_distance(center, future)
                                    distance2 = euclidean_distance(center, ground_truth)
                                    distance = (distance1 + distance2) / 2
                                    total_distance += distance
                                    se=squared_error(future, ground_truth)
                                    total_distanceSE+=se
                                                
                                            #ID3 Direction 0.5
                                    distance1D = euclidean_distance(center, futureD)
                                    distanceD = (distance1D + distance2) / 2
                                    total_distanceD += distanceD
                                    se=squared_error(futureD, ground_truth)
                                    total_distanceSED+=se
                                                
                                                #ID3 Direction GT 0.5
                                    distance1D_GT = euclidean_distance(center, futureDGT1)
                                    distanceD_GT = (distance1D_GT + distance2) / 2
                                    total_distanceD_GT += distanceD_GT
                                    se=squared_error(futureDGT1, ground_truth)
                                    total_distanceSED_GT+=se  
                                    nb_prediction+=1

                                            
                           #1 sec                 
                                        #<ID3 1 sec
                                if missed_data==False: 
                                    distance12 = euclidean_distance(center2, future2)
                                    distance22 = euclidean_distance(center2, ground_truth)
                                    distance = (distance12 + distance22) / 2
                                    total_distance2 += distance
                                    se=squared_error(future2, ground_truth)
                                    total_distanceSE2+=se
                                                
                                            #ID3 Direction 1 sec
                                    distance12D = euclidean_distance(center2, futureD2)
                                    distance2D = (distance12D + distance22) / 2
                                    total_distance2D += distance2D
                                    se=squared_error(futureD2, ground_truth)
                                    total_distanceSE2D+=se   
                                                
                                            #ID3 Direction GT 1
                                    distance2D_GT = euclidean_distance(center2, futureDGT2)
                                    distance2D_GT = (distance2D_GT + distance22) / 2
                                    total_distanceD2_GT += distance2D_GT
                                    se=squared_error(futureDGT2, ground_truth)
                                    total_distanceSED2_GT+=se  
                                    nb_prediction2+=1
                                            
                           #1.5 sec                 
                                       #ID3 1.5 sec
                                if missed_data==False: 
                                    distance13 = euclidean_distance(center3, future3)
                                    distance23 = euclidean_distance(center3, ground_truth)
                                    distance = (distance13 + distance23) / 2
                                    total_distance3 += distance
                                    se=squared_error(future3, ground_truth)
                                    total_distanceSE3+=se
                                                
                                            #ID3 Direction 1.5 sec
                                    distance13D = euclidean_distance(center3, futureD2)
                                    distance2D = (distance13D + distance23) / 2
                                    total_distance3D += distance2D
                                    se=squared_error(futureD3, ground_truth)
                                    total_distanceSE3D+=se    
                                                
                                            #ID3 Direction GT 1.5
                                    distance3D_GT = euclidean_distance(center3, futureDGT3)
                                    distance3D_GT = (distance3D_GT + distance23) / 2
                                    total_distanceD3_GT += distance3D_GT
                                    se=squared_error(futureDGT3, ground_truth)
                                    total_distanceSED3_GT+=se                                          
                                                                                    
                                    nb_prediction3+=1
                                precision_occ=0            
                                if not((sum_tp_occ + sum_fp_occ)==0):
                                    precision_occ = sum_tp_occ / float(sum_tp_occ + sum_fp_occ)
                                recall_occ=0  
                                if not((sum_tp_occ + sum_fn_occ)==0):
                                    recall_occ = sum_tp_occ / float(sum_tp_occ + sum_fn_occ)
                                accuracy_occ=0
                                if not((sum_tp_occ + sum_fp_occ + sum_fn_occ)==0):
                                    accuracy_occ = sum_tp_occ / float(sum_tp_occ + sum_fp_occ + sum_fn_occ)
                                precision=0
                                if not((sum_tp + sum_fp)==0):
                                    precision = sum_tp / float(sum_tp + sum_fp)
                                recall=0
                                if not((sum_tp + sum_fn)==0):
                                    recall = sum_tp / float(sum_tp + sum_fn)
                                accuracy=0
                                if not((sum_tp + sum_fp + sum_fn)==0):
                                    accuracy = sum_tp / float(sum_tp + sum_fp + sum_fn)
                                if not(nb_prediction==0):
                                    mean_distance = total_distance / nb_prediction
                                if not(nb_prediction2==0):
                                    mean_distance2 = total_distance2 / nb_prediction2
                                if not(nb_prediction3==0):
                                    mean_distance3 = total_distance3 / nb_prediction3
                                if not(nb_prediction==0):
                                    mean_distanceD = total_distanceD / nb_prediction
                                if not(nb_prediction2==0):
                                    mean_distance2D = total_distance2D / nb_prediction2
                                if not(nb_prediction3==0):
                                    mean_distance3D = total_distance3D / nb_prediction3
                                if not(nb_prediction==0):
                                    mean_distanceD_GT = total_distanceD_GT / nb_prediction
                                if not(nb_prediction2==0):
                                    mean_distance2D_GT = total_distanceD2_GT / nb_prediction2
                                if not(nb_prediction3==0):
                                    mean_distance3D_GT = total_distanceD3_GT / nb_prediction3
                                SE=SE2=SE3=SED=SE2D=SE3D=SED_GT=SED2_GT=SED3_GT=0
                                if not(nb_prediction==0):
                                    SE = total_distanceSE / nb_prediction
                                if not(nb_prediction2==0):
                                    SE2 = total_distanceSE2 / nb_prediction2
                                if not(nb_prediction3==0):
                                    SE3 = total_distanceSE3 / nb_prediction3
                                if not(nb_prediction==0):
                                    SED = total_distanceSED / nb_prediction
                                if not(nb_prediction2==0):
                                    SE2D = total_distanceSE2D / nb_prediction2
                                if not(nb_prediction2==0):
                                    SE3D = total_distanceSE3D / nb_prediction3
                                if not(nb_prediction==0):
                                    SED_GT = total_distanceSED_GT / nb_prediction
                                if not(nb_prediction2==0):
                                    SED2_GT = total_distanceSED2_GT / nb_prediction2
                                if not(nb_prediction3==0):
                                    SED3_GT = total_distanceSED3_GT / nb_prediction3                                            
                                            
                                final_t=time.time()
                                inference_time=final_t-initial_t
                                total_inference_time+=inference_time
                                fps=1/(total_inference_time/num_frame)
                                video_name = video_path[-14:]

                                cv2.putText(frame, f"Video : {video_name}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Pedestrian : {str(i)}", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Frame : {num_frame}", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Inference Time Frame : {inference_time:.4f}", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Frame Per Second : {fps:.4f}", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                cv2.putText(frame, f"Precision without occlusion : {precision_occ:.4f}", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Recall without occlusion : {recall_occ:.4f}", (400, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Accuracy without occlusion : {accuracy_occ:.4f}", (400, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                               
                                cv2.putText(frame, f"Precision : {precision:.4f}", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Recall : {recall:.4f}", (700, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"Accuracy : {accuracy:.4f}", (700, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                                               
                                cv2.putText(frame, f"MSE 0.5s  : {SE:.4f}", (700, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"MSE   1s  : {SE2:.4f}", (700, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"MSE 1.5s  : {SE3:.4f}", (700, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                cv2.putText(frame, f"MSE 0.5s  (Correction): {SED:.4f}", (700, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"MSE   1s  (Correction): {SE2D:.4f}", (700, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"MSE 1.5s  (Correction): {SE3D:.4f}", (700, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                cv2.putText(frame, f"MSE 0.5s  (Correction State JAAD): {SED_GT:.4f}", (700, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"MSE   1s  (Correction State JAAD): {SED2_GT:.4f}", (700, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, f"MSE 1.5s  (Correction State JAAD): {SED3_GT:.4f}", (700, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                # Display the frameid
                                
                            
                            cv2.imshow('Frame', frame)
                            frame=None
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            del frame

                        cap.release()
                        cv2.destroyAllWindows()
                        del cap
                        del pose_piles
                        #End While cap
                    #End if track element
                # Save progress data
                with open('check_point.txt', 'w') as f:
                                    f.write(f"{nb_occlusion}\n")
                                    f.write(f"{sum_tp_occ}\n")
                                    f.write(f"{sum_fp_occ}\n")
                                    f.write(f"{sum_fn_occ}\n")
                                    f.write(f"{sum_tn_occ}\n")
                                    
                                    f.write(f"{sum_tp}\n")
                                    f.write(f"{sum_fp}\n")
                                    f.write(f"{sum_fn}\n")
                                    f.write(f"{sum_tn}\n")

                                    f.write(f"{nb_prediction}\n")
                                    f.write(f"{nb_prediction2}\n")
                                    f.write(f"{nb_prediction3}\n")
                                    f.write(f"{total_distance}\n")
                                    f.write(f"{total_distance2}\n")
                                    f.write(f"{total_distance3}\n")
                                    f.write(f"{total_distanceD}\n")
                                    f.write(f"{total_distance2D}\n")
                                    f.write(f"{total_distance3D}\n")
                                    f.write(f"{total_distanceD_GT}\n")
                                    f.write(f"{total_distanceD2_GT}\n")
                                    f.write(f"{total_distanceD3_GT}\n")
                                    f.write(f"{total_distanceSE}\n")
                                    f.write(f"{total_distanceSE2}\n")
                                    f.write(f"{total_distanceSE3}\n")
                                    f.write(f"{total_distanceSED}\n")
                                    f.write(f"{total_distanceSE2D}\n")
                                    f.write(f"{total_distanceSE3D}\n")
                                    f.write(f"{total_distanceSED_GT}\n")
                                    f.write(f"{total_distanceSED2_GT}\n")
                                    f.write(f"{total_distanceSED3_GT}\n")
                                    f.write(f"{num_v+1}\n")
            #End if video mp4
        #End for video
    #End Whith pose
    # Calculate precision, recall, and accuracy.

precision_occ = sum_tp_occ / float(sum_tp_occ + sum_fp_occ)
recall_occ = sum_tp_occ / float(sum_tp_occ + sum_fn_occ)
accuracy_occ = sum_tp_occ / float(sum_tp_occ + sum_fp_occ + sum_fn_occ)

precision = sum_tp / float(sum_tp + sum_fp)
recall = sum_tp / float(sum_tp + sum_fn)
accuracy = sum_tp / float(sum_tp + sum_fp + sum_fn)

mean_distance = total_distance / nb_prediction
mean_distance2 = total_distance2 / nb_prediction2
mean_distance3 = total_distance3 / nb_prediction3
mean_distanceD = total_distanceD / nb_prediction
mean_distance2D = total_distance2D / nb_prediction2
mean_distance3D = total_distance3D / nb_prediction3
mean_distanceD_GT = total_distanceD_GT / nb_prediction
mean_distance2D_GT = total_distanceD2_GT / nb_prediction2
mean_distance3D_GT = total_distanceD3_GT / nb_prediction3




print('**********State Prediction Evaluation**********')

print('******Number of skeleton occluded:', 0)
print('TP:', sum_tp_occ)
print('FP:', sum_fp_occ)
print('FN:', sum_fn_occ)
print('TN:', sum_tn_occ)
print('Precision:', precision_occ)
print('Recall:', recall_occ)
print('Accuracy:', accuracy_occ)

print('******Number of skeleton occluded:', nb_occlusion)
print('TP:', sum_tp)
print('FP:', sum_fp)
print('FN:', sum_fn)
print('TN:', sum_tn)
print('Precision:', precision)
print('Recall:', recall)
print('Accuracy:', accuracy)

print('**********Path Prediction Evaluation**********')
print('Total number of path predicted:', nb_prediction)
print('****MED:')
print(f'Mean Euclidean Distance: {mean_distance} , {mean_distance2} , {mean_distance3}')
print(f'Mean Euclidean Distance with correction case 9:{mean_distanceD} , {mean_distance2D} , {mean_distance3D}')
print(f'Mean Euclidean Distance with correction case 9 and state provided:{mean_distanceD_GT} , {mean_distance2D_GT} , {mean_distance3D_GT}')

print('****MSE:')
print(f'Mean Squared Error: {SE} , {SE2} , {SE3}')
print(f'Mean Squared Error with correction case 9: {SED} , {SE2D} , {SE2D}')
print(f'Mean Squared Error with correction case 9 and state provided: {SED_GT} , {SED2_GT} , {SED3_GT}')

    

