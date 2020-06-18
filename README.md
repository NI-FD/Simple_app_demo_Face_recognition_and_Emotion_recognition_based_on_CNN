# Simple app demo of face recognition and emotion recognition based on CNN
Use open source code and trained CNN model to build a simple app demo

## Features
* Determine when a person is in front of the robot.
* Recognize whether the person is the owner of the robot.
* Determine the mood of the person.
* Robot will give different notifications for different situations.
* For owner, the detector rectangle box and text will be green; for unknown people, those will be red.
![demo](/IMG/demo.gif)
## Requirement
1. Environment:
	* Python 3.50
	* Keras
	* Tensorflow
2. Module:
	* numpy
	* imutils
	* cv2
	* face_recognition

## File Structure
1. Key_picture: Storing the owner's picture
2. haarcascade: Storing the face detector
3. models: Storing the CNN trained mood classification 
4. recognition.py: Main code

## Run
* Directly run the recognition.py


## PS & References
* If you want to change the owner, you can replace the picture that was stored in Key_picture folder. And change the owner name in line 36.
* The trained mood classification CNN model downloaded from Internet. [ageitgey/face_recognition](https://github.com/omar178/Emotion-recognition#p4)
* The face classification came from Module face_recognition. [omar178/Emotion-recognition](https://github.com/ageitgey/face_recognition)
