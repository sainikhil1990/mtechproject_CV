'''
Motion detection and motion tracking using Background substraction method. 

'''

import cv2
import numpy as np
import time

def process_frame(frame, background_subtractor, min_contour_area=500):
    # Apply background subtractor to get the foreground mask
    fg_mask = background_subtractor.apply(frame)

    # Remove shadows and noise using morphological operations
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    


    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2
                    # Calculate orientation (angle) using PCA
            data = np.array(contour[:, 0, :], dtype=np.float64)
            mean, eigenvectors = cv2.PCACompute(data, mean=np.empty((0)))
            angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

            bbox = (x, y, w, h)
            coordinates = (center_x, center_y)
            orientation = angle
            print(f"Bounding Box: {bbox}")
            print(f"Coordinates: {coordinates}")
            print(f"Orientation: {orientation}")
            

            return center_x, center_y, (x, y, w, h)
    return None, None, None

def calculate_speed(position1, position2, time_elapsed, scale=1):
    if position1 is None or position2 is None:
        return 0.0
    distance = np.linalg.norm(np.array(position1) - np.array(position2))
    speed = (distance * scale) / time_elapsed  # speed in units per second
    print(distance) 
    return speed

def main():
    # Initialize video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture('/home/eiiv-nn1-l3t04/conv/image_file/out.mp4')
    speed_list = []
    # Initialize background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    previous_position = None
    previous_time = None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count = frame_count+1
        print("Processing frame no." + str(frame_count)+".....")
        print("Object info...")
        # Process the frame to detect and get the position of the moving object
        current_position, _, bbox = process_frame(frame, background_subtractor)
        print(f"Current Position: {current_position}")


        # Calculate speed if there was a previous position
        if previous_position is not None and current_position is not None:
            current_time = time.time()
            time_elapsed = current_time - previous_time
            speed = calculate_speed(previous_position, current_position, time_elapsed, scale=1)  # scale is used to convert pixels to real-world units
            speed_list.append(speed)
            previous_time = current_time
        else:
            speed = 0.0
            previous_time = time.time()

        # Update previous position
        previous_position = current_position

        # Draw bounding box and speed on the frame
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Speed: {speed:.2f} units/s', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imwrite('Motion_track.png', frame)
        print("Processing Complete for frame: ", frame_count)
        print("*********************************************")



    avg_speed = sum(speed_list)/len(speed_list)
    print("Average speed of moving object is: ", avg_speed)
    # Release video capture and close windows


if __name__ == '__main__':
    main()
