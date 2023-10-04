# Import the libraries
import cv2
import numpy as np
import os # Import os library to check file existence

# Define a function to draw circles on mouse clicks
def draw_circle(event, x, y, flags, param):
    global img, l # Use global variables for the image and the list of points
    if event == cv2.EVENT_LBUTTONDBLCLK: # Check if the left button is double clicked
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1) # Draw a blue circle on the image
        p = (x, y) # Store the coordinates of the point
        l.append(p) # Append the point to the list
        print(l) # Print the list

# Define a function to drag circles on mouse moves
def drag_circle(event, x, y, flags, param):
    global img_copy, l_copy, dragging, index # Use global variables for the copied image and list, the dragging flag and the index of the selected point
    if event == cv2.EVENT_LBUTTONDOWN: # Check if the left button is pressed down
        for i in range(len(l_copy)): # Loop through the list of points
            if np.linalg.norm(np.array([x, y]) - np.array(l_copy[i])) < 10: # Check if the mouse position is close to any point
                dragging = True # Set the dragging flag to True
                index = i # Store the index of the selected point
                break # Break the loop
    elif event == cv2.EVENT_MOUSEMOVE: # Check if the mouse is moving
        if dragging: # Check if the dragging flag is True
            l_copy[index] = (x, y) # Update the coordinates of the selected point
            img_copy = img.copy() # Make a copy of the original image
            for p in l_copy: # Loop through the list of points
                cv2.circle(img_copy, p, 5, (255, 0, 0), -1) # Draw a blue circle on the copied image
            update_output() # Call a function to update the output image based on the new points
    elif event == cv2.EVENT_LBUTTONUP: # Check if the left button is released
        dragging = False # Set the dragging flag to False

# Define a function to update the output image based on the new points
def update_output():
    global img_copy, l_copy, dst # Use global variables for the copied image and list and the output image
    pts1 = np.float32(l_copy) # Store the new points as a numpy array
    pts2 = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]) # Define the output points as a numpy array (same as before)
    M = cv2.getPerspectiveTransform(pts1, pts2) # Get the perspective transform matrix from OpenCV 
    dst = cv2.warpPerspective(img_copy, M, (int(width), int(height))) # Apply the perspective transform to the copied image using OpenCV 
    cv2.imshow('output', dst) # Show the output image

# Ask the user to enter the file name
file_name = input("Enter the file name: ")

# Check if the file exists
if os.path.exists(file_name):
    # Load the image
    img = cv2.imread(file_name)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "FSRCNN_x4.pb"
    sr.readModel(path) 
    sr.setModel("fsrcnn",4) # set the model by passing the value and the upsampling ratio
    img = sr.upsample(img) # upscale the input image
else:
    # Print an error message and exit
    print("File not found")
    exit()
    
# Calculate the margin size as 5% of the original image size
margin = int(0.05 * min(img.shape[:2])) # Use min to get the smaller dimension of the image 

# Create a new image with a white border and copy the original image to the center of it
img_border = np.ones((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin, 3), dtype=np.uint8) * 255 # Create a white image with 2 * margin extra pixels in each dimension 
img_border[margin:-margin, margin:-margin] = img # Copy the original image to the center of the new image 
img = img_border # Assign the new image to img

# Create a list to store the selected points
l = []

# Create a window to display the image
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)

# Set the mouse callback function to draw circles
cv2.setMouseCallback('image', draw_circle)

# Show the image until ESC key is pressed or 4 points are selected
while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(20) & 0xFF 
    if key == 27 or len(l) == 4:
        break

# Destroy all windows
cv2.destroyAllWindows()

# Check if exactly 4 points are selected
if len(l) == 4:
    # Order the points as top-left, top-right, bottom-left, bottom-right
    l = sorted(l, key=lambda x: x[0]) # Sort by x coordinate
    l1 = sorted(l[:2], key=lambda x: x[1]) # Sort the left two points by y coordinate
    l2 = sorted(l[2:], key=lambda x: x[1]) # Sort the right two points by y coordinate
    factor = 1
    top_left = (int(l1[0][0] - margin*factor), int(l1[0][1] - margin*factor))
    top_right = (int(l2[0][0] + margin*factor), int(l2[0][1] - margin*factor))
    bottom_left = (int(l1[1][0] - margin*factor), int(l1[1][1] + margin*factor))
    bottom_right = (int(l2[1][0] + margin*factor), int(l2[1][1] + margin*factor))
    l = [top_left, top_right, bottom_left, bottom_right] # Store the ordered points as a list
    l = [l1[0], l2[0], l1[1], l2[1]] # Store the ordered points as a list

    # Ask the user to enter the aspect ratio of the output image as a fraction
    aspect_ratio = input("Enter the aspect ratio of the output image as a fraction (e.g. 4/3): ")

    # Convert the aspect ratio to a float
    aspect_ratio = eval(aspect_ratio)

    # Define the output size and points
    height = max(np.linalg.norm(np.array(l[0]) - np.array(l[2])), np.linalg.norm(np.array(l[1]) - np.array(l[3]))) # Calculate the height of the output image based on input points
    width = height * aspect_ratio # Calculate the width of the output image based on aspect ratio
    pts2 = np.float32([[0+ margin*factor, 0+ margin*factor] , [width-1 - margin*factor, 0 + margin*factor], [0 + margin*factor, height-1 - margin*factor], [width-1 -margin*factor, height-1-margin*factor]]) # Define the output points as a numpy array

    # Calculate the perspective transform matrix and apply it to the image
    print(l)
    M = cv2.getPerspectiveTransform(np.float32(l), pts2) # Get the perspective transform matrix from OpenCV 
    dsize = (int(width), int(height))
    dst = cv2.warpPerspective(img, M, dsize, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)) # Apply the perspective transform to the image using OpenCV 

    # Make a copy of the original image and list
    img_copy = img.copy()
    l_copy = l.copy()

    # Initialize the dragging flag and index
    dragging = False
    index = -1

    # Create a new window to display the copied image with circles
    cv2.namedWindow('adjust', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('adjust', 600, 600)
    
    # Create a new window to display the final image
    cv2.namedWindow('final', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('final', 1024, 768)

    # Set the mouse callback function to drag circles
    cv2.setMouseCallback('adjust', drag_circle)

    # Show the copied image and the output image until ESC key is pressed
    while True:
        cv2.imshow('adjust', img_copy)
        cv2.imshow('final', dst)
        if cv2.waitKey(20) & 0xFF == 27:
            break
            
    
    cv2.imwrite(file_name.split('.')[0]+'_reshape.jpg', dst)
    
    # Destroy all windows
    cv2.destroyAllWindows()
else:
    print("Please select exactly 4 points")
