import handy
import cv2
import classifier

# getting video feed from webcam


# capture the hand histogram by placing your hand in the box shown and
# press 'A' to confirm
# source is set to inbuilt webcam by default. Pass source=1 to use an
# external camera.
channel = 0
hist = handy.capture_histogram(source=channel)
cap = cv2.VideoCapture(channel, cv2.CAP_DSHOW)
clf = classifier.Classifier('RandomForest','a.csv')
clf.loadOurData()
clf.train()
area = []
defe = []
cla = []
Itrno = 0;
while True:
    Itrno = Itrno + 1
    ret, frame = cap.read()
    if not ret:
        break

    # to block a faces in the video stream, set block=True.
    # if you just want to detect the faces, set block=False
    # if you do not want to do anything with faces, remove this line
    handy.detect_face(frame, block=True)

    # detect the hand
    hand = handy.detect_hand(frame, hist)

    # to get the outline of the hand
    # min area of the hand to be detected = 10000 by default
    custom_outline = hand.draw_outline(
        min_area=10000, color=(0, 255, 255), thickness=2)

    # to get a quick outline of the hand
    quick_outline = hand.outline

    # draw fingertips on the outline of the hand, with radius 5 and color red,
    # filled in.
    for fingertip in hand.fingertips:
        cv2.circle(quick_outline, fingertip, 5, (0, 0, 255), -1)

    # to get the centre of mass of the hand
    com = hand.get_center_of_mass()
    if com:
        cv2.circle(quick_outline, com, 10, (255, 0, 0), -1)

    cv2.imshow("Handy", quick_outline)

    # display the unprocessed, segmented hand
    # cv2.imshow("Handy", hand.masked)
    temp = hand.binary;
    # print(type(temp))
    # print(temp.shape)
    fileName = 'debug.jpg'
    cv2.imwrite(fileName, temp)
    # display the binary version of the hand
    cv2.imshow("Handy", hand.binary)
    final = cv2.imread(fileName)
    pred, features, image = clf.predict(final)
    image, ratio, defet= clf.preprocessing(final)
    print(ratio,defet)
    area.append(ratio)
    defe.append(defet[1])
    cla.append(0)
    print(Itrno)
    k = cv2.waitKey(5)
    # Press 'q' to exit
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import numpy as np
#
# data = np.c_[cla, area, defe]
#
# import pandas as pd
#
# df = pd.DataFrame(data)
# df.to_csv('a.csv', index=None)
