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

# Ask the user to enter the file name
file_name = input("Enter the file name: ")

# Check if the file exists
if os.path.exists(file_name):
    # Load the image
    img = cv2.imread(file_name)
else:
    # Print an error message and exit
    print("File not found")
    exit()

# Create a list to store the selected points
l = []

# Create a window to display the image
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)

# Set the mouse callback function to draw circles
cv2.setMouseCallback('image', draw_circle)

# Show the image until ESC key is pressed
while True:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

# Destroy all windows
cv2.destroyAllWindows()

# Check if exactly 4 points are selected
if len(l) == 4:
    # Order the points as top-left, top-right, bottom-left, bottom-right
    l = sorted(l, key=lambda x: x[0]) # Sort by x coordinate
    l1 = sorted(l[:2], key=lambda x: x[1]) # Sort the left two points by y coordinate
    l2 = sorted(l[2:], key=lambda x: x[1]) # Sort the right two points by y coordinate
    pts1 = np.float32([l1[0], l2[0], l1[1], l2[1]]) # Store the ordered points as a numpy array

    # Define the output size and points
    width = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3])) # Calculate the width of the output image
    height = max(np.linalg.norm(pts1[0] - pts1[2]), np.linalg.norm(pts1[1] - pts1[3])) # Calculate the height of the output image
    pts2 = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]) # Define the output points as a numpy array

    # Calculate the perspective transform matrix and apply it to the image
    M = cv2.getPerspectiveTransform(pts1, pts2) # Get the perspective transform matrix from OpenCV 
    dst = cv2.warpPerspective(img, M, (int(width), int(height))) # Apply the perspective transform to the image using OpenCV 

    # Show the output image until ESC key is pressed
    cv2.imshow('output', dst)
    cv2.imwrite('file_name'+'_reshape.jpg', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Please select exactly 4 points")