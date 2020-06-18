import face_recognition
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

class Camera():

    def startCamera(self):
        # Get a reference to webcam #0 (the default one)
        self.video_capture = cv2.VideoCapture(0)

    def getFrame(self):
        ret, frame = self.video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        return self.video_capture, frame, rgb_small_frame

class Face_detector():
    
    def loadKeyPicture(self):
        # Load a sample picture and learn how to recognize it.
        owner_image = face_recognition.load_image_file("./key_picture/owner.jpg")
        owner_face_encoding = face_recognition.face_encodings(owner_image)[0]

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            owner_face_encoding
        ]
        self.known_face_names = [
            "Owner:Daniel Wu"
        ]
        # parameters for loading data and images
        self.detection_model_path = './haarcascade_files/haarcascade_frontalface_default.xml'
        self.emotion_model_path = './models/_mini_XCEPTION.102-0.66.hdf5'

        # hyper-parameters for bounding boxes shape
        # loading models
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
        "neutral"]

    def compare(self,rgb_small_frame):
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.45)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        return face_locations, face_names

    def emotion(self,frame):
        frame = frame.read()[1]
        #reading the frame
        frame = imutils.resize(frame,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        print(faces)
        labels = []
        for face in faces:

            (fX, fY, fW, fH) = face
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.emotion_classifier.predict(roi)[0]
            label = self.EMOTIONS[preds.argmax()]
            labels.append(label)
        
        print(labels)
        return labels



    def labelVideo(self,frame,face_locations,face_names,emotion_labels):
        unknown_person = 0
        known_person = 0
        # Display the results
        font = cv2.FONT_HERSHEY_DUPLEX
        for (top, right, bottom, left), name, emotion in zip(face_locations, face_names, emotion_labels):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            if name == "Unknown":
                rectangle_color = (0, 0, 255)
                name_color = (0, 0, 255)
            else:
                rectangle_color = (0,255,0)
                name_color = (0,255,0)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 70), (right, bottom), name_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 10 ), font, 0.7, (0, 0, 0), 1) 

            # Indicate the emotion of owner
            cv2.putText(frame, 'Mood:' + emotion, (left + 6, bottom - 50 ), font, 0.7, (0, 0, 0), 1) 
            
            if name == "Unknown":
                unknown_person += 1
            else:
                known_person += 1
        cv2.putText(frame, "Sport Robot:", (5, 45), font, 1.5, (0, 255, 0), 1) 
        if known_person == 0 and unknown_person == 0:
            cv2.putText(frame, " -'No one in front of the camera!'", (5, 80), font, 1.3, (0, 255, 0), 1) 

        elif known_person == 0 and unknown_person != 0:
            cv2.putText(frame, " -'Found " + str(len(face_locations)) + " stranger in front of the camera!'", (5, 80), font, 1.3, (0, 0, 255), 1) 

        elif known_person == 1 and unknown_person == 0:
            cv2.putText(frame, " -'Hello, my owner!'", (5, 80), font, 1.3, (0, 255, 0), 1)

        elif known_person == 1 and unknown_person != 0:
            cv2.putText(frame, " -'Hello, my owner!'", (5, 80), font, 1.3, (0, 255, 0), 1)
            cv2.putText(frame, " -'It seems that you brought friends!'", (5, 115), font, 1.3, (0, 0, 255), 1)

        # elif known_person == 0 and unknown_person != 0:
        #     cv2.putText(frame, "Found " + str(len(face_locations)) + " Unknown Person!", (0, 20), font, 1.0, (0, 0, 255), 1) 

        # elif known_person != 0 and unknown_person == 0:
        #     cv2.putText(frame, "Found " + str(len(face_locations)) + " Person from Household!", (0, 20), font, 1.0, (0, 255, 0), 1)

        # else:
        #     cv2.putText(frame, "Found " + str(unknown_person) + " Unknown Person and " + str(known_person) + " Person from Household!", (0, 20), font, 1.0, (255, 0, 0), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)        

if __name__ == "__main__":
    camera = Camera()
    detector = Face_detector()

    camera.startCamera()
    detector.loadKeyPicture()

    while True:
        capture, frame, rgb_frame = camera.getFrame()
        face_locations, face_names = detector.compare(rgb_frame)
        emotion_label = detector.emotion(capture)
        detector.labelVideo(frame, face_locations, face_names, emotion_label)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    camera.video_capture.release()
    cv2.destroyAllWindows()
