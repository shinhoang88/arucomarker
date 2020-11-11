"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    3) https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import freenect
# cap = cv2.VideoCapture(-1)

import os
import time
from datetime import datetime


# Photo session settings
total_photos = 50             # Number of images to take
countdown = 5                 # Interval for count-down timer, seconds
font=cv2.FONT_HERSHEY_SIMPLEX # Cowntdown timer font

# Final image capture settings
scale_ratio = 1

# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGRA)
    return array

# # Camera settimgs
# cam_width = 1280              # Cam sensor width settings
# cam_height = 480              # Cam sensor height settings


# # Camera resolution height must be dividable by 16, and width by 32
# cam_width = int((cam_width+31)/32)*32
# cam_height = int((cam_height+15)/16)*16
# print ("Camera resolution: "+str(cam_width)+" x "+str(cam_height))
#
# # Buffer for captured image settings
# img_width = int (cam_width * scale_ratio)
# img_height = int (cam_height * scale_ratio)
# Lets start taking photos!
counter = 0
t2 = datetime.now()
###------------------ ARUCO TRACKER ---------------------------
while (True):
    frame = get_video()
    t1 = datetime.now()
    cntdwn_timer = countdown - int((t1 - t2).total_seconds())

    # If cowntdown is zero - let's record next image
    if cntdwn_timer == -1:
        counter += 1
        filename = './scenes/scene_' + str(counter) + '.jpg'
        cv2.imwrite(filename, frame)
        print(' [' + str(counter) + ' of ' + str(total_photos) + '] ' + filename)
        t2 = datetime.now()
        time.sleep(1)
        cntdwn_timer = 0  # To avoid "-1" timer display
        next

    # Draw cowntdown counter, seconds
    cv2.putText(frame, str(cntdwn_timer), (50, 50), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("pair", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'Q' key to quit, or wait till all photos are taken
    if (key == ord("q")) | (counter == total_photos):
        break

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()
