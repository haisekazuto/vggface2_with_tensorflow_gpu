# vggface2_with_tensorflow_gpu
This is my implementation of VGGFace2 using tensorflow_gpu 1.13.2 for facial recognition with Webcam.
It is still at its early stages so any comment will be really helpful.
It is highly recommended that people check out rcmalli's github here to get the basic information.


Instead of comparing two pictures of two people, we can get the mean of each person's face features from multiple images, in which way would provide
more accurate data of that person's face. 

To pretrain more faces, just insert a folder with those people's names and in those folder include the images of their faces and run "retrainfaces.py". 
Afterwards, you should remove the image folder to avoid repetition (which would cause a lot of pretraining time.)
"has_trained" folder is where you can view some of the images that I had included for testing. 

To run the application, just execute "startstream.py" and press q if you want to quit. 

During the process of making this "program", I have gathered some that would be helpful for some people:

Where I get the functions: https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/

Where I started: https://pypi.org/project/face_recognition/

More information on rcmalli's github: https://github.com/rcmalli/keras-vggface 
