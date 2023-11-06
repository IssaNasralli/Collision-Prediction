import os
import cv2
import xml.etree.ElementTree as ET
import mediapipe as mp
import numpy as np


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


def get_landmark_coords_ratio(pose_landmarks, landmark_id):
    # Check if pose landmarks are detected
    if pose_landmarks is not None:
        # Get the landmark object for the specified ID
        landmark = pose_landmarks.landmark[landmark_id]

        # Calculate the x and y pixel coordinates of the landmark
        x = landmark.x
        y = landmark.y

        return x, y
    else:
        # Return None if no landmarks are detected
        return None, None
    
def process_video_and_annotations(video_folder):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
   # mp_holistic = mp.solutions.holistic
    with mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5) as pose:
        pose_data = []
        frame_count = 0
        current_state=""
        for video_file in os.listdir(video_folder):
            
            if video_file.endswith(".mp4"):
                video_path = os.path.join(video_folder, video_file)
                annotation_path = os.path.join(video_folder, video_file.replace(".mp4", ".xml"))
                print (video_path)
                # Parse the XML annotation file
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                track_elements = root.findall(".//track[@label='pedestrian']")
                
                # We will work only with the first 'pedestrian' track
                if track_elements:
                    pedestrian_track = track_elements[0]
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
    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
    
                        # Resize the frame
                        frame = cv2.resize(frame, (desired_width, desired_height))
    
                        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Get current frame number (0-based index)
    
                        # Check if the frame number has corresponding annotation
                        matching_boxes = [box for box in boxes if int(box.get("frame")) == frame_number]
                        for box in matching_boxes:
                            xtl = int(float(box.get("xtl")) * scale_x)
                            ytl = int(float(box.get("ytl")) * scale_y)
                            xbr = int(float(box.get("xbr")) * scale_x)
                            ybr = int(float(box.get("ybr")) * scale_y)
                            occluded = int(float(box.get("occluded")))
    
                            # Process the 'action' attribute to get the state of pedestrian  
                            action = box.find(".//attribute[@name='action']").text.strip().lower()
                            if(current_state==""):
                                current_state=action
    
                            # Filtering the extracted state (e.g., only consider 'walking' or 'standing')
                                # Display the bounding box
                            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
                            cv2.putText(frame, f"Action: {action}", (xtl, ytl - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # Pose estimation for each pedestrian detected
                            xmin,ymin,xmax,ymax=xtl,ytl,xbr,ybr
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
                            
                                # Check if pose landmarks are detected
                                if results.pose_landmarks is not None:
                    
                                    # Collect pose data
                                    if frame_count < 16:
                                        if(current_state==action):
                                            if(occluded==0):
                                                pose_landmarks = results.pose_landmarks
                                                pose_coords = []
                            
                                                for i in range(33):
                                                    if i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,18,19,20,23,24]:
                                                        continue
                                                    x, y = get_landmark_coords_ratio(pose_landmarks, i)
                                                    pose_coords.extend([x, y])
                            
                                                pose_data.append(pose_coords)
                                                frame_count += 1
                                        else:
                                            frame_count = 0
                                            pose_data = []
                                            current_state=""                    
                                    if frame_count == 16:
                                        # Write pose data to file based on user input
                                        if current_state=="standing":
                                            jaad_input=0
                                        else:
                                            jaad_input=1
                                        if jaad_input in [0, 1]:
                                            #user_input = input("Press any key then enter to cancel, just enter to validate:")
                                            user_input=""
                                            if  (user_input==""):
                                                with open("D4.csv", "a") as f:
                                                    # Create a single line of comma-separated values for the 16 consecutive poses
                                                    line_data = []
                                                    for pose_coords in pose_data:
                                                        line_data.extend(pose_coords)
                                                    line_data.append(jaad_input)
                                                    line = ",".join(map(str, line_data)) + "\n"
                                            
                                                    # Write the line to the file
                                                    with open("D4.csv", "a") as f:
                                                        f.write(line)
                    
                                        # Reset frame count and pose data
                                        frame_count =frame_count-1
                                        pose_data.pop(0)
                                       

                        # Display the frame
                        cv2.imshow('Frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
    
                    cap.release()
                    cv2.destroyAllWindows()

# Provide the path to the folder containing the videos and annotations
video_folder_path = "/path/to/your/video/folder"

video_folder_path = "M:\Memoire master\PCPTree\Object Detector\MobileNet\JAAD"
process_video_and_annotations(video_folder_path)
