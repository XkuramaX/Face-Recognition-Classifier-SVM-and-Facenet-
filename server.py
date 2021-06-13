
import cv2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import  Normalizer
from sklearn.svm import SVC
import numpy as np
from datetime import datetime, timedelta
from keras_facenet import FaceNet
import requests as req
import json
from concurrent.futures import ThreadPoolExecutor
import pickle
# from mtcnn.mtcnn import MTCNN




embedder = FaceNet()



import os

BASE_DIR = os.path.dirname(__file__)
cascade_file_location = os.path.join(BASE_DIR,"frontal_face.xml")

def file_check(s):
    l = ['png', 'jpg', 'jpeg']
    s2 = s.split('.')[-1]
    if s2 in l:
        return True
    else:
        return False

def face_locations(test_img):
    
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_RGB2GRAY)
    face_haar_cascade = cv2.CascadeClassifier(cascade_file_location)
    #image processing


    
    faces = face_haar_cascade.detectMultiScale( gray_img, scaleFactor= 1.2,minNeighbors= 3)

    # detector = MTCNN()

    # pixels = detector.detect_faces(test_img)
    # print(pixels)

    return faces



def face_detection(test_image):
    face = []
    img = test_image
    faces = face_locations(img)
    print(faces)
    if len(faces):
        x,y,w,h = faces[0]
        cropped_face = test_image[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face,(160,160))
        # cv2.imshow("face testing",cropped_face)
        # cv2.waitKey(10)
    face_encoding = embedder.embeddings([cropped_face])
    # print(face_encoding)
    return face_encoding, img



def labels_for_training_data(directory):
    faces = []
    faceID = []
    count = 1

    for path,subdirnames,filenames in os.walk(directory):

        for filename in filenames:
            if filename.startswith("."):
                continue
            if not file_check(filename):
                continue
            ID = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print(img_path)
            test_img = cv2.imread(img_path)
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            print("Image successfully loaded")

            faces_rect, gray_img = face_detection(test_img)
            if len(faces_rect):
                print(faces_rect[0].shape)
                faces.append(faces_rect[0])
                faceID.append(int(ID))

    return faces, faceID

def train_model(KnownEncodings, KnownClasses):
    
    KnownEncodings = np.array(KnownEncodings)
    KnownClasses = np.array(KnownClasses)

    #normalizing the input
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(KnownEncodings)

    #converting the output to integers
    trainY = KnownClasses

    index = {}
    ind = 0
    for i in range(len(trainY)):
        if trainY[i] not in index.keys():
            index[trainY[i]] = ind
            ind += 1


    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainY)

    return model, index


def test_model(img, model, index):
    
    in_encoder = Normalizer(norm="l2")

    name = ""
    imgS = img

    results = set()

    rgb_img = imgS[:,:,::-1]
     
    prob = 0

    facesCurFrame = face_locations(rgb_img)
    cropped_faces = []

    for i in facesCurFrame:
        x,y,w,h = i
        cropped_face = img[y : y+h, x : x+w]
        cropped_faces.append(cropped_face)

    
    
    print(len(cropped_faces))
    

    encodesCurFrame = []
    if len(cropped_faces):
        encodesCurFrame = embedder.embeddings(cropped_faces)
    
    print(len(encodesCurFrame))


    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        testX = in_encoder.transform([encodeFace])
        name = model.predict(testX)
        
        prob = model.predict_proba(testX)
        
        probability = prob[0][index[name[0]]] * 100
        print(probability)
        
        if probability > 65:
            
            results.add(name[0])
            
            x, y, w, h = faceLoc
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(name[0]) + "(" + str(round(probability,2)) + ")", (x + 6, y+h - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if True:
                cv2.putText(img, "Marked!", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    prev = datetime.now()
            
    print("Time before response:",prev )
    # resp = mark_attendance(name)
    print("Time taken by api:", datetime.now() - prev)

    mark_attendance_async(results)

    return img, name, prob


def fetch(session, url):
    with session.get(url) as response:
        print(response.json()['success'])



def mark_attendance_async(results):
    results = list(results)
    length = len(results)
    department = {1:"ECE", 2:"CSE", 3:"EE", 4:"IT"}
    urls = []
    # making the urls
    for name in results:
        college = "STCET"
        label = int(name)
        dept = label//1000
        roll = label % 1000
        url = "https://attendface.herokuapp.com/api/attendance/mark/"+college+"/"+department[dept]+"/"+str(roll)
        
        # url = "http://localhost:3000/api/attendance/mark/"+college+"/"+department[dept]+"/"+str(roll)

        urls.append(url)

    with ThreadPoolExecutor() as executor:
        with req.Session() as session:
            executor.map( fetch, [session]*length, urls)
            executor.shutdown(wait=True)


########################################################### Main Code ###############################################################3

print(BASE_DIR)

filename_model = "SVM_model.sav"
filename_index = "index.txt"

try :
    model = pickle.lead(open(filename_model, 'rb'))
    print("Model is loaded successfully")

    with open(filename_index, 'rb') as index_file:
        index = json.load(index_file)

    ind = index
    index = {}
    for i in ind.keys():
        index[int(i)] = ind[i]
    print("Index file is loaded successfully!")

except :

    X, Y = labels_for_training_data(os.path.join(BASE_DIR,"training_imgs"))

    model, index = train_model(X, Y)

    ind = {}
    for i in index.keys():
        ind[int(i)] = index[i]

    pickle.dump(model, open(filename_model, 'wb'))

    print("Model is succesfully saved!")

    with open(filename_index, 'w') as outfile:
        json.dump(ind, outfile)
    print("Index file is saved!")

vid = cv2.VideoCapture(0)

while True:
    success, current_image = vid.read()
    time = datetime.now()
    img, prediction, probability = test_model(current_image, model, index)
    time_now = datetime.now()
    diff = time_now - time

    print(diff)
    if prediction:
        print(prediction)
        # print(prediction)
        # print(probability)
    cv2.imshow('FaceAttend', img)
    inp = cv2.waitKey(1)

    if inp == ord('q'):
        break


vid.release()
cv2.destroyAllWindows







# print(X,Y)