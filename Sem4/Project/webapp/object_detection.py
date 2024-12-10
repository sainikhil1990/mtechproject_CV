import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect an object and return its bounding box, coordinates, and orientation
def detect_object(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Shape of the image", gray.shape) 
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the object
    contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate the center of the bounding box
    cx, cy = x + w // 2, y + h // 2
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0
    angle = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])

    # Calculate orientation (angle) using PCA
    data = np.array(contour[:, 0, :], dtype=np.float64)
    mean, eigenvectors = cv2.PCACompute(data, mean=np.empty((0)))
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

    annotated_image = image.copy()
    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 255), -1)
    cv2.putText(annotated_image, f"Center: "+str(cx)+","+str(cy), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #plt.imshow(annotated_image)
    resized_image = cv2.resize(annotated_image, (500, 500))
    #cv2.imshow('Annotated Image', resized_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    filename = 'savedImage.jpg'
  
    # Using cv2.imwrite() method 
    # Saving the image 
    cv2.imwrite(filename, resized_image)
        # Crop the image around the object
    output_size = (800, 800)
    #cropped_image = annotated_image[y:y+h, x:x+w]
    cropped_image = annotated_image[y:y+h+20, x:x+w+20]
    # Resize the cropped image to the desired output size
    #resized_image = cv2.resize(cropped_image, output_size)
    #plt.imshow(resized_image)
    cv2.imwrite('cropped_image.jpg', cropped_image)
    return (x, y, w, h), (cx, cy), angle

# Example usage
image = cv2.imread("/home/eiiv-nn1-l3t04/Project/dataset/training/image1.jpg")
bbox, coordinates, orientation = detect_object(image)



print(f"Bounding Box: {bbox}")
print(f"Coordinates: {coordinates}")
print(f"Orientation: {orientation}")
