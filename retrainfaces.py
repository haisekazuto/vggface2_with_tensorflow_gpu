# including important stuffs
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

# VGGFace2's extract_faces() function, read in readme.md 
def extract_faces(filename, required_size=(224, 224)):
	pixels = pyplot.imread(filename)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# VGGFace2's get_embeddings() function, read in readme.md 
def get_embeddings(filenames):
    faces = [extract_faces(f) for f in filenames]
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model = 'resnet50', include_top = False, input_shape = (224, 224, 3), pooling = 'avg')
    yhat = model.predict(samples)
    return yhat

# acquire a list/ an array full of zeros
def get_fullzero(size):
    ret = []
    for i in range(0, size):
        ret.append(0)
    return ret

# declare a dictionary with names: face-embedding
people_faces = {}

# checks if data has already existed
if os.path.exists('data.pickle'):
    with open('data.pickle', 'rb') as f:
        people_faces = pickle.load(f)

al = [] # number of images of a person in the folder name
faces = [] # need-to-update face images directory
embeddings = [] # need-to-update faces embeddings
names = [] # need-to-update names

# iterate through the folder need_training and acquire information
for folder_name in os.listdir('.\\need_training'):
    cnt = 0
    names.append(folder_name)
    folder_directory = '.\\need_training\\' + '\\' + folder_name
    for face in os.listdir(folder_directory):
        k = folder_directory + '\\' + face
        faces.append(k)
        cnt += 1
    
    al.append(cnt)

# getting the embeddings
embeddings = get_embeddings(faces)

cnt = 0

# taking the mean of each embedding for each person to acquire accurate facial information
for i in range(0, len(al)):
    mean_face_embeddings = get_fullzero(2048)
    for j in range(cnt, cnt+al[i]):
        for k in range(0, len(embeddings[j])):
            mean_face_embeddings[k] += embeddings[j][k]
    cnt += al[i]
    for it in range(0, 2048):
        mean_face_embeddings[it] /= al[i]
    people_faces[names[i]] = mean_face_embeddings

# dumping the data back for later use
with open('data.pickle', 'wb') as f:
    pickle.dump(people_faces, f)