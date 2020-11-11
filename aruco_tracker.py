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
import math


####---------------------- CALIBRATION ---------------------------
# termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# # checkerboard of size (8 x 6) is used
# objp = np.zeros((6 * 8, 3), np.float32)
# objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
#
# # arrays to store object points and image points from all the images.
# objpoints = []  # 3d point in real world space
# imgpoints = []  # 2d points in image plane.
#
# # iterating through all calibration images
# # in the folder
# images = glob.glob('calib_images/checkerboard/*.jpg')
#
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # find the chess board (calibration pattern) corners
#     ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
#
#     # if calibration pattern is found, add object points,
#     # image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#
#         # Refine the corners of the detected corners
#         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners2)
#
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
#
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array

cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

###------------------ ARUCO TRACKER ---------------------------
while (True):
    # ret, frame = cap.read()
    # get a frame from RGB camera
    frame = get_video()

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, mtx, dist)

        # print('position different:', tvec)
        # (rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            # aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)


            aruco.drawAxis(frame, mtx, dist, rvec[0], tvec[0], 0.02)  # np.array([0.0, 0.0, 0.0])
            # print(f, "\t", end = " ")
            # print("%d táv: %.2f" % (i, math.sqrt(rvec[i][0][0]**2 + rvec[i][0][1]**2 + rvec[i][0][2]**2)))
            # cv2.putText(image, image.shape())
            # cv2.putText(image, "%.1f cm" % ((20000 / rr**0.5) * 0.116 - 2.08), (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244, 244, 244))
            cv2.putText(frame, "%.1f cm -- %.0f deg" % ((tvec[0][0][2] * 100), (rvec[0][0][2] / math.pi * 180)),
                        (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244, 244, 244))
            R, _ = cv2.Rodrigues(rvec[0])
            cameraPose = -R.T * tvec[0]
            # print(type(cameraPose))
            
            # print((int)(tvec[0][0][2] * 1000))

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0]) + ', '

        cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()
