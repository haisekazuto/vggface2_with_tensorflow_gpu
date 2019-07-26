# getting important stuffs
import cv2
import os
from keras_vggface import VGGFace
from mtcnn.mtcnn import MTCNN
import json
import pickle
from numpy import asarray
from keras_vggface.utils import preprocess_input
from PIL import Image
from matplotlib import pyplot
import tensorflow as tf
from scipy.spatial.distance import cosine

# VGGFace Face-feature extraction model resnet50 with average pooling
model = VGGFace(model = "resnet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')

# modified face extraction function for time-saving
def extract_face(pipi, required_size=(224, 224)):
	image = Image.fromarray(pipi)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# modified face feature extracting function also for time-saving
def get_embedding(face):
    faces = [extract_face(face)]
    sample = asarray(faces, 'float32')
    ret = model.predict(sample)
    return ret

# comparing two faces
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		return score
	else:
		return -1

# getting pre-trained faces from data.pickle (where we store the names and their embeddings)
name_faces = {}
with open('data.pickle', 'rb') as f:
    name_faces = pickle.load(f)

# starts the webcam (or your major video-capturing device)
video_capture = cv2.VideoCapture(0)
face_locations = []

# face detecting model (I think this is easier to implement than ssh)
detector = MTCNN()

while 1: 
    ret, fframe = video_capture.read()
    sframe = cv2.resize(fframe, (0, 0), fx=0.5, fy=0.5) # face-detection does not require much so I downsized the frame
    outframe = cv2.resize(fframe, (0, 0), fx=1.5, fy=1.5) # output-frame, you can change fx, fy to fit your consumption needs
    srgb_frame = sframe[:, :, ::-1]
    rgb_frame = fframe[:, :, ::-1]
    if 1: 
        results = detector.detect_faces(srgb_frame) # detect all faces in the frame
        for d in results: # iterate through each face
            x1, y1, width, height = d['box'] 
            x2, y2 = x1+width, y1+height
            x1 = max(0, (x1-2) * 2)
            x2 = (x2+2) * 2
            y1 = max(0, (y1-2) * 2)
            y2 = (y2+2) * 2
            # translating the coordinates
            face = rgb_frame[y1:y2, x1:x2]
            # get the portion of the frame that is the face and extract features with get_embedding()
            embeddin = get_embedding(face)
            name = "Unknown" # will be changed if there is matching faces
            curthres = 0.55 # 0.5 is default
            for i in name_faces:
                k = is_match(embeddin, name_faces[i], curthres)
                if k > 0:
                    name = i
                    curthres = k # optimal & closest-alike face
            cv2.rectangle(outframe, (int(x1*1.5), int(y1*1.5)), (int(x2*1.5), int(y2*1.5)), (0, 0, 0), 2) # box the face
            cv2.putText(outframe, name, (int(x1*1.5) + 6, int(y2*1.5) - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1) # face-owner's name
    cv2.imshow('Video', outframe) # show the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # PRESS q to Quit!
        break

video_capture.release()
cv2.destroyAllWindows()