import os
import cv2
import numpy as np
import tensorflow as tf

def get_bounding_boxes(image, bounding_boxes, class_labels, threshold, nms_threshold):
    # Filter bounding boxes based on threshold
    selected_boxes = []
    selected_class_labels = []
    for i, bbox in enumerate(bounding_boxes):
        confidence = class_labels[i][0]
        if confidence > threshold:
            selected_boxes.append(bbox)
            selected_class_labels.append(class_labels[i])

    # Apply non-maximum suppression (NMS)
    selected_boxes = np.array(selected_boxes)
    selected_class_labels = np.array(selected_class_labels)
    indices = cv2.dnn.NMSBoxes(selected_boxes.tolist(), selected_class_labels[:, 0].tolist(), threshold, nms_threshold)
    indices = indices.flatten()

    # Create a list to store the bounding boxes
    bounding_box_list = []

    # Extract bounding boxes for the selected indices
    for idx in indices:
        bbox = selected_boxes[idx]
        class_label = selected_class_labels[idx]

        # Convert class_label to a scalar value or list
        if isinstance(class_label, np.ndarray) and class_label.size == 1:
            class_label = class_label.item()
        elif isinstance(class_label, np.ndarray):
            class_label = class_label.tolist()

        # Create a dictionary for the bounding box
        bounding_box = {
            'xmin': int(bbox[0]),
            'ymin': int(bbox[1]),
            'xmax': int(bbox[2]),
            'ymax': int(bbox[3]),
            'class_label': class_label
        }

        bounding_box_list.append(bounding_box)

    return bounding_box_list

def calculate_iou(box1, box2):
    print (box1[0])
    print (box2[0])
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

IMAGE_PATH = 'Dataset/'
ANNOTATION_PATH = 'Dataset/'

sum_tp = 0  # Sum of True Positives
sum_fp = 0  # Sum of False Positives
sum_fn = 0  # Sum of False Negatives

# STEP 2: Load the input images and annotations.
image_files = sorted([file for file in os.listdir(IMAGE_PATH) if file.endswith('.jpg')])

# Load TFLite model
model_path = 'efficientdet_lite0_uint8.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']  # Get the input shape

for image_file in image_files:
    image_path = os.path.join(IMAGE_PATH, image_file)
    annotation_path = os.path.join(ANNOTATION_PATH, image_file.replace('.jpg', '.txt'))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']  # Get the input shape
    input_image = cv2.imread(image_path)
    resized_image = tf.image.resize(input_image, (input_shape[1], input_shape[2]))
    normalized_image = np.array(resized_image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(normalized_image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    bounding_boxes = output_data[:, :, :4]
    class_labels = output_data[:, :, 4:]
    detected_bounding_boxes = get_bounding_boxes(input_image, bounding_boxes[0], class_labels[0], threshold=0.001, nms_threshold=0.001)


    # Load the ground truth bounding boxes from the annotation file.
    with open(annotation_path, 'r') as file:
        
        gt_bounding_boxes = []
        content = file.read()
        num_chars = len(content.replace('\n', ''))
        gt = 0
        if num_chars == 0:
            print("Empty")
            continue  # Skip this iteration if the file is empty
    
        for line in content.split('\n'):
            if line.strip() != '':
                print ("Not empty")
                gt+=1        
                class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                x_min = int((x_center - box_width / 2) * 640)
                y_min = int((y_center - box_height / 2) * 640)
                x_max = int((x_center + box_width / 2) * 640)
                y_max = int((y_center + box_height / 2) * 640)
                gt_bounding_boxes.append((x_min, y_min, x_max, y_max))

    # Initialize counters for TP, FP, FN for each image.
    tp = 0
    fp = 0
    fn = 0

    # Process the detection result and calculate TP, FP, FN.
    for gt_box in gt_bounding_boxes:
        matched = False
        for i, detected_box in enumerate(detected_bounding_boxes):
            iou = calculate_iou(gt_box, detected_box)
            if iou >= 0.5:
                tp += 1
                matched = True
                detected_bounding_boxes.pop(i)
                break
        if not matched:
            fn += 1

    fp = len(detected_bounding_boxes)

    # Accumulate TP, FP, FN for all images.
    sum_tp += tp
    sum_fp += fp
    sum_fn += fn
    print(f'{image_file}')
    print(f'gt={gt}, TP={tp}, FP={fp} and FN= {fn}')

# Calculate precision, recall, and accuracy.
precision = sum_tp / float(sum_tp + sum_fp)
recall = sum_tp / float(sum_tp + sum_fn)
accuracy = sum_tp / float(sum_tp + sum_fp + sum_fn)

print('TP:', sum_tp)
print('FP:', sum_fp)
print('FN:', sum_fn)

print('Precision:', precision)
print('Recall:', recall)
print('Accuracy:', accuracy)

    