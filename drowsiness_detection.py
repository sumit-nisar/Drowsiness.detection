# -*- coding: utf-8 -*-

import dlib
import cv2
import numpy as np
import time 
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from utility_functions import calculate_eye_aspect_ratio,play_alarm
from threading import Thread

arguments = {"shape_predictor":"shape_predictor_68_face_landmarks.dat",
             "alarm":"TextToo.mp3"} #this we created a dictonary whenever we want to change we can change it from here and we use as a input


eye_aspect_ratio_threshold = 0.4
min_number_of_consecutive_frams = 50 #50 is for we are checkinng for 2 seconds

detector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(arguments["shape_predictor"])

(l_start,l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #this will gives indexes of the left eyes
(r_start,r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

video_stream = VideoStream(src=0).start() #src=0 for internal webcam
frame_counter = 0
alarm_on=False
time.sleep(1) #to wait till the webcam opens.. as in if dont open then it will start reading the frames so we use time

while True:
    frame = video_stream.read()
    if frame.any(): #it to check whether image came proper or not and if not then take till it gets a proper image
        frame = imutils.resize(frame,width=450)
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #converting into grayscale
        
        rectangles=detector(gray_image,0) # taking a rectangles from gray_image & 0 is for take image as it don't do any filter
        
        for rect in rectangles:
            shape = predictor(gray_image,rect)
            shape = face_utils.shape_to_np(shape) #converting into single numpy array
            
            left_eye = shape[l_start:l_end] #slicing the left eye postion form all facial co-ordinates
            right_eye = shape[r_start:r_end] #slicing the right eye postion form all facial co-ordinates
            
            left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye)
            right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye)
            
            average_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio)/2.0
            
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull =cv2.convexHull(right_eye) #it returns the boundary 
            
            cv2.drawContours(frame,[left_eye_hull],-1,(0,255,0),2)
            cv2.drawContours(frame,[right_eye_hull],-1,(0,255,0),2)
            
            if average_eye_aspect_ratio < eye_aspect_ratio_threshold:
                frame_counter += 1
                
                if frame_counter >= min_number_of_consecutive_frams:
                    if not alarm_on:
                        alarm_on = True
                        
                        thread = Thread(target=play_alarm,args=(arguments["alarm"],))
                        thread.daemon =True #false rahega jabh main process bandh hai and true hai matlab sabh subprocess bhi on rahega
                        thread.start()
                    
                    cv2.putText(frame,"Drowsiness Alert",(10,30),cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,(0,0,255),2)
            else:
                frame_counter=0
                alarm_on=False
                
            cv2.putText(frame,"Eye Aspect Ratio {:.2f}".format(average_eye_aspect_ratio),
                        (300,30),cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,(0,0,255),2)
            
        cv2.imshow("frame",frame)
        
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
video_stream.stop()