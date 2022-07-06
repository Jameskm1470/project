import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self,mode = False,complexity= 1,landmarks = True,
                 enSeg = False, smoSeg = True, detCon = 0.5, traCon= 0.5 ):
        self.mode=mode
        self.complexity=complexity
        self.landmarks=landmarks
        self.enSeg=enSeg
        self.smoSeg=smoSeg
        self.detCon=detCon
        self.traCon=traCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.complexity,self.landmarks,
                                     self.enSeg,self.smoSeg,self.detCon,self.traCon)

    def findPose(self,img,draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return(img)

    def findPosition(self,img,draw=True):
        lmList = []
        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                #print(id,lm)
                cx,cy =int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('Dataset/ThrowsDataset/GoodTechnique(1320)/FT1-GT.mp4')
    pTime = 0
    detector= poseDetector()

    while True:
        success, img = cap.read()
        img= detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("image", img)

        cv2.waitKey(1)

if __name__ == "__main__"  :
    main()