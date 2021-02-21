import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import math

def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

def handGesture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    d = np.zeros(img.shape, np.uint8)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            max_area = area
            ci = i
    cnt = contours[ci]
    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    center = (cx, cy)
    hullarea = cv2.contourArea(hull)
    handarea = cv2.contourArea(cnt)
    ratio = handarea / hullarea
    new = calculateFingers(cnt, img)
    img = thresh1
    return (img, ratio, new)

class Classifier:
    def __init__(self, clf_type, csv_path='newFeatures.csv'):
        self.type = clf_type
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.selected_labels = []

        #selecting classifier
        if (self.type == 'RandomForest'):
            self.clf = RandomForestClassifier(n_estimators=100)
        else:
            print('type mismatch')

        #intialising feature and class dataframes
        sarr = []
        num_feat = len(self.data.columns) - 1
        for i in range(num_feat):
            sarr.append(str(i + 1))
        self.X = pd.DataFrame(columns=sarr)
        self.y = pd.DataFrame(columns=['0'])

    def loadData(self, gestures='advw'):
        #extracting the labels
        labels = list(gestures)
        self.selected_labels = [ord(i) - ord('a') for i in labels]

        #load data
        for i in self.selected_labels:
            print('\n-----------')
            print('Including ' + chr(i + ord('a')) + ' in data')
            letter = self.data.loc[self.data['0'] == i]
            letter_y = letter[['0']]
            letter_X = letter.loc[:, letter.columns != '0']
            self.X = self.X.append(letter_X)
            self.y = self.y.append(letter_y)
        print(self.X.shape)
        print(self.y.shape)

    def loadOurData(self):
        # extracting the labels
        self.selected_labels = [0, 1, 2, 3]
        # load data
        for i in self.selected_labels:
            print('\n-----------')
            print('Including ' + chr(i + ord('a')) + ' in data')
            letter = self.data.loc[self.data['0'] == i]
            letter_y = letter[['0']]
            letter_X = letter.loc[:, letter.columns != '0']
            self.X = self.X.append(letter_X)
            self.y = self.y.append(letter_y)
        print(self.X.shape)
        print(self.y.shape)

    def train(self):
        self.clf.fit(self.X.values, self.y.values.astype('int').flatten())
        print('completed training')

    def preprocessing(self,image):
        img, ratio, defects = handGesture(image)
        return img, ratio, defects

    def predict(self,image):
        img, ratio, defects = self.preprocessing(image)
        features = [[ratio], [defects[1]]]
        #print(features)
        features = np.array(features)
        features = features.T
        y_pred = self.clf.predict(features)
        #print(chr(y_pred[0]+ord('a')))
        print(y_pred)
        return y_pred,features,img

# clf = Classifier('RandomForest')
# clf.loadData()
# clf.train()
# img = cv2.imread('debug3.jpeg')
# pred,image = clf.predict(img)
# cv2.imwrite("debug.jpeg", image)
