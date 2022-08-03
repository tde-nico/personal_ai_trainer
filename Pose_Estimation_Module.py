import cv2
import mediapipe as mp
import time
import math


class PoseDetector:
	def __init__(self, mode=False, model_complexity=1, upper_body=False, smooth=True, detection_con=0.5, track_con=0.5):
		self.mode = mode
		self.model_complexity = model_complexity
		self.upper_body = upper_body
		self.smooth = smooth
		self.detection_con = detection_con
		self.track_con = track_con
		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(mode, model_complexity, upper_body, smooth, detection_con, track_con)
		self.mpDraw = mp.solutions.drawing_utils


	def find_pose(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = self.pose.process(imgRGB)
		self.lms = results.pose_landmarks
		if self.lms and draw:
				self.mpDraw.draw_landmarks(img, self.lms, self.mpPose.POSE_CONNECTIONS)
		return img


	def find_position(self, img, draw=True):
		self.lm_list = []
		try:
			for id, lm in enumerate(self.lms.landmark):
				#print(id, lm)
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				#print(id, cx, cy)
				self.lm_list.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
		except:
			pass
		return self.lm_list


	def find_angle(self, img, p1, p2, p3, draw=True):
		x1, y1 = self.lm_list[p1][1:]
		x2, y2 = self.lm_list[p2][1:]
		x3, y3 = self.lm_list[p3][1:]
		angle = math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2)
		angle = math.degrees(angle)
		if angle < 0:
			angle += 360
		if draw:
			cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
			cv2.line(img, (x3,y3), (x2,y2), (255,255,255), 3)
			cv2.circle(img, (x1,y1), 10, (0,0,255), cv2.FILLED)
			cv2.circle(img, (x1,y1), 15, (0,0,255), 2)
			cv2.circle(img, (x2,y2), 10, (0,0,255), cv2.FILLED)
			cv2.circle(img, (x2,y2), 15, (0,0,255), 2)
			cv2.circle(img, (x3,y3), 10, (0,0,255), cv2.FILLED)
			cv2.circle(img, (x3,y3), 15, (0,0,255), 2)
			cv2.putText(img, str(int(angle)), (x2-50,y2+50),
				cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
		return angle


def main():
	cap = cv2.VideoCapture(0) # 1 ?
	pTime = 0
	cTime = 0
	detector = PoseDetector()
	while 1:
		success, img = cap.read()
		img = detector.find_pose(img)
		lm_list = detector.find_position(img)
		#print(lm_list)

		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime

		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
		cv2.imshow("Image", img)
		cv2.waitKey(1)

if __name__ == '__main__':
	main()
