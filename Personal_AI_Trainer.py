import cv2
import time
import numpy as np
import Pose_Estimation_Module as pem


cap = cv2.VideoCapture(0) # 1 ?

pTime = 0

detector = pem.PoseDetector()

while 1:
	success, img = cap.read()
	img = detector.find_pose(img, draw=False)
	lm_list = detector.find_position(img, draw=False)

	if lm_list:
		h, w, c = img.shape
		# Right
		angle = detector.find_angle(img, 12, 14, 16)
		bar = np.interp(angle, (210,310), (h-100,100))
		cv2.rectangle(img, (30,100), (100, h-100), (0,255,0), 2)
		cv2.rectangle(img, (30,int(bar)), (100, h-100), (0,255,0), cv2.FILLED)
		# Left
		angle = detector.find_angle(img, 11, 13, 15)
		bar = np.interp(angle, (210,310), (h-100,100))
		cv2.rectangle(img, (w-100,100), (w-30, h-100), (0,255,0), 2)
		cv2.rectangle(img, (w-100,int(bar)), (w-30, h-100), (0,255,0), cv2.FILLED)

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)
