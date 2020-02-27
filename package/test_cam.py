from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2

# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
webcam = VideoStream(src=0).start()
#picam = VideoStream(usePiCamera=True).start()

while True:
    # read the next frame from the video stream and resize
    # it to have a maximum width of 400 pixels
    frame = webcam.read()
    frame = imutils.resize(frame, width=400)

    # convert the frame to grayscale, blur it slightly, update
    # the motion detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    cv2.imshow('test', frame)
    # check to see if a key was pressed
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
