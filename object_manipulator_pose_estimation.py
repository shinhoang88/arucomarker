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
import time

# Manipulator base
firstMarkerID = 6
# Random Object
secondMarkerID = 7

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

def inversePerspective(rvec, tvec):
    """ Applies perspective transform for given rvec and tvec. """
    rvec, tvec = rvec.reshape((3, 1)), tvec.reshape((3, 1))
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)

    invTvec = invTvec.reshape((3, 1))
    invTvec = invTvec.reshape((3, 1))
    return invRvec, invTvec

def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """ Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 """
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def euclideanDistanceOfTvecs(tvec1, tvec2):
    print("tvec1",tvec1)
    return math.sqrt(math.pow(tvec1[0]-tvec2[0], 2) + math.pow(tvec1[1]-tvec2[1], 2) + math.pow(tvec1[2]-tvec2[2], 2))

def euclideanDistanceOfTvec(tvec):
    return euclideanDistanceOfTvecs(tvec, [0, 0, 0])

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
    isFirstMarkerDetected = False
    isSecondMarkerDetected = False
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

        for i in range(0, ids.size):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.095, mtx, dist)



            if ids[i] == firstMarkerID:
                firstRvec = rvec[i]
                firstTvec = tvec[i]
                isFirstMarkerDetected = True
                firstMarkerCorners = corners[i]
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.06)
            elif ids[i] == secondMarkerID:
                secondRvec = rvec[i]
                secondTvec = tvec[i]
                isSecondMarkerDetected = True
                secondMarkerCorners = corners[i]
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.06)

            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            # aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
            aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
            # time.sleep(0.5)

        # if isFirstMarkerDetected and isSecondMarkerDetected:
        #     composedRvec, composedTvec = relativePosition(firstRvec, firstTvec, secondRvec, secondTvec)
        #     # print infos
        #     # print("firstTvec: ", firstTvec)
        #     # print("secondTvec: ", secondTvec)
        #
        #     camerafirstRvec, cameraFirstTvec = inversePerspective(firstRvec, firstTvec)
        #     camerasecondRvec, camerasecondTvec = inversePerspective(secondRvec, secondTvec)
        #
        #     differenceRvec, differenceTvec = camerafirstRvec - camerasecondRvec, cameraFirstTvec - camerasecondTvec
        #
        #     # print infos
        #     print("first Rvec: ", camerafirstRvec)
        #     print("first Tvec: ", cameraFirstTvec)
        #
        #     print("Second marker Rvec: ", camerasecondRvec)
        #     print("Second marker Tvec: ", camerasecondTvec)
        #
        #     print("differenceRvec: ", differenceRvec)
        #     print("differenceTvec: ", differenceTvec)

            # # print("secondTvec[0]: ", secondTvec[0])
            # realDistanceInTvec = euclideanDistanceOfTvec(secondTvec[0])
            # print(cv2.norm(secondTvec[0]))
            #
            # difference = euclideanDistanceOfTvecs(composedTvec.T[0], secondTvec[0])
            # print(difference)
            # calculatedDistance = realDistanceInTvec * (distanceBetweenTwoMarkers / difference)

    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame', frame)
    # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
    key = cv2.waitKey(3) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Calibration
        if isFirstMarkerDetected and isSecondMarkerDetected:
            composedRvec, composedTvec = relativePosition(firstRvec, firstTvec, secondRvec, secondTvec)

            # print infos
            print("composedRvec: ", -composedRvec)
            print("composedTvec: ", -composedTvec)

            # camerafirstRvec, cameraFirstTvec = inversePerspective(firstRvec, firstTvec)
            # camerasecondRvec, camerasecondTvec = inversePerspective(secondRvec, secondTvec)
            #
            # differenceRvec, differenceTvec = camerafirstRvec - camerasecondRvec, cameraFirstTvec - camerasecondTvec
            #
            # # print infos
            # print("first Rvec: ", camerafirstRvec)
            # print("first Tvec: ", cameraFirstTvec)
            #
            # print("Second marker Rvec: ", camerasecondRvec)
            # print("Second marker Tvec: ", camerasecondTvec)
            #
            # print("differenceRvec: ", differenceRvec/ math.pi * 180)
            # print("differenceTvec: ", differenceTvec* 100)



# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()
