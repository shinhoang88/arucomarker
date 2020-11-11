<img src="https://docs.opencv.org/3.1.0/markers.jpg" height="100">          <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/OpenCV_Logo_with_text.png" height="100">

# Aruco Tracker
[![HitCount](http://hits.dwyl.io/njanirudh/Aruco_Tracker.svg)](http://hits.dwyl.io/njanirudh/Aruco_Tracker)
Forked from:
https://github.com/njanirudh/Aruco_Tracker/

## Dependencies
* Python 3.x
* Numpy
* OpenCV 3.3+ 
* OpenCV 3.3+ Contrib modules

## Scripts
1. **calib_image_taken.py** : Move the checkedboard inside camera range and take 50 photos (for calibration)

2. **camera_calibration.py** : opencv default calibration images and writes the value to calib_images/test.yaml.

3. **extract_calibration.py**  : This script shows how to open and extract the calibration values from a file.

4. **object_manipulator_pose_estimation.py**  : compute the difference of rotation & translation vectors between the manipulator base and object (between 2 aruco marker IDs).


## References
1. https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
2. https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
3. https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html

--------------------------------------------------------------
- Author        : Phi Tien Hoang
- E-mail        : phitien@skku.edu
- Organization  : Robotory-SKKU-S.Korea


 
 
 
 
