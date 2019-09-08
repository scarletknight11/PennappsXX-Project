from cv2 import *
import time

# initialize the camera
videoFile = "capture.avi"
headcc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)   # 0 -> index of camera
fps = cam.get(cv2.CAP_PROP_POS_FRAMES)

s, img = cam.read()
if s:    # frame captured without any errors
   # namedWindow("cam-test", cv2.WINDOW_NORMAL)
   # imshow("cam-test", img)
    #waitKey(0)
    #destroyWindow("cam-test")


    def video():
        ret, frame = cam.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        head = headcc.detectMultiScale(frame, 1.2, 2, 0, (20, 20), (40, 40))

        # print type(head)
        # print head
        # print head.shape
       # print("Number of heads detected: " + str(head.shape[0]))

        if len(head) > 0:
            for (x, y, w, h) in head:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

        # cv2.rectangle(frame, ((0,frame.shape[0] -25)),(270, frame.shape[0]), (255,255,255), -1)
        # cv2.putText(frame, "Number of head detected: " + str(head.shape[0]), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)

        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera', frame)


    while cam.isOpened():
        video()
        cf = cam.get(cv2.CAP_PROP_POS_FRAMES) - 1
        cam.set(cv2.CAP_PROP_POS_FRAMES, cf + 60)
        cv2.setTrackbarPos("pos_trackbar", "Frame Grabber",
        int(cam.get(cv2.CAP_PROP_FPS)))
        time.sleep(2)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    print(fps)
    cam.release()
    cv2.destroyAllWindows()