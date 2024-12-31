######## Simple Picture Taking Script #########
# source: https://github.com/EdjeElectronics/Image-Dataset-Tools/blob/main/PictureTaker/PictureTaker.py
#
# Author: Evan Juras, EJ Technology Consultants
# Date: 8/8/21
# Description: 
# This program takes pictures (in .jpg format) from a connected webcam and saves
# them in the specified directory. The default directory is 'Pics' and the
# default resolution is 1280x720.
#
# Example usage to save images in a directory named Sparrow at 1920x1080 resolution:
# python3 PictureTaker.py --imgdir=Sparrow --resolution=1920x1080

import cv2
import os
import argparse
import sys

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--imgdir', help='Root directory for dataset',
                   default='SignLanguageDataset')
parser.add_argument('--letter', help='Letter/sign being captured',
                   required=True)
parser.add_argument('--resolution', help='Desired camera resolution in WxH.',
                   default='64x64')

args = parser.parse_args()

# Create class-specific directory
dirname = os.path.join(args.imgdir, args.letter)
if not os.path.exists(dirname):
    os.makedirs(dirname)

# Parse resolution
imW = int(args.resolution.split('x')[0])
imH = int(args.resolution.split('x')[1])

# Initialize image counter
imnum = 1
while os.path.exists(os.path.join(dirname, f'{args.letter}_{imnum}.jpg')):
    imnum += 1

# Initialize webcam
cap = cv2.VideoCapture(0)
ret = cap.set(3, 640)  # Set to webcam's native resolution
ret = cap.set(4, 480)

# Initialize display window
winname = 'Press "p" to take a picture, "q" to quit'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 50, 30)

print(f'Capturing images for letter {args.letter}')
print('Press p to take a picture. Pictures will be saved in the', dirname, 'folder')
print('Press q to quit')

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
        
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    processed = cv2.resize(frame, (imW, imH))
    
    # Show both original and processed frames
    cv2.imshow('Original', frame)
    cv2.imshow('Processed (64x64)', processed)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        # Save both original and processed images
        filename = f'{args.letter}_{imnum}.jpg'
        savepath = os.path.join(dirname, filename)
        cv2.imwrite(savepath, processed)
        print(f'Picture taken and saved as {filename}')
        imnum += 1

cv2.destroyAllWindows()
cap.release()
