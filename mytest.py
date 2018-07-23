from iou_tracker import track_iou
from face import face_analysis
import time
import cv2

def track_visualization(video_path, track):
	cap = cv2.VideoCapture(video_path)
	count = 0
	while 1:
		ret, frame = cap.read()
		if not ret:
			break
		count += 1

		for i in range(0, len(track)):
			if track[i]['start_frame']<=count and (track[i]['start_frame'] + len(track[i]['bboxes']) - 1)>= count:
				print track[i]['start_frame'], len(track[i]['bboxes']), count
				bbx = track[i]['bboxes'][count - track[i]['start_frame']]
				fleft = int(bbx[0])
				ftop = int(bbx[1])
				fright = int(bbx[2])
				fbot = int(bbx[3])

				frame = cv2.rectangle(frame,(fleft, ftop),(fright, fbot), (0,0,255),2)
				frame = cv2.putText(frame,'ID:'+ str(i),(fleft,ftop),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
		cv2.imwrite('results/'+str(count)+'.jpg', frame)
		cv2.imshow('tracking result', frame)
		cv2.waitKey(1)


video_path = './1.mp4'
fa = face_analysis(0)
t0 = time.time()
result = fa.run_video_det(video_path)
t1 = time.time()
print "detection time:" + str(t1 - t0)

t2 = time.time()
tracks = track_iou(result, 0.2, 0.6, 0.3, 90)
t3 = time.time()
print "tracking time" + str(t3-t2)

track_visualization(video_path, tracks)