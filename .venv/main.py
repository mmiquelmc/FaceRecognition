import face_recognition
import os
import cv2
import math
import sys
import numpy as np

def face_confidence(face_distance, face_match_threshold=0.6):
    # Calculate the confidence level based on the face distance and match threshold
    range_value = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_value * 2.0)

    if face_distance > face_match_threshold:
        # If the face distance is greater than the match threshold, return the linear value as a percentage
        return str(round(linear_val * 100, 2)) + '%'
    else:
        # If the face distance is within the match threshold, calculate a confidence value using a formula
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        # Initialize the FaceRecognition class by encoding the known faces
        self.encode_faces()

    def encode_faces(self):
        # Encode the known faces by loading the images and extracting the face encodings
        for image_file in os.listdir('./faces'):
            face_image = face_recognition.load_image_file(f'./faces/{image_file}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(os.path.splitext(image_file)[0])

        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)  

        if not video_capture.isOpened():
            sys.exit("¡Cámara no encontrada!")

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                # Process the current frame by resizing and converting it to RGB
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Locate and encode the faces in the frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # Compare the face encodings with the known face encodings
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    # Calculate the face distance and find the best match
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        # If there is a match, assign the name and confidence level
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale the face locations back to the original frame size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw rectangles around the faces and display the name and confidence level
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Create an instance of the FaceRecognition class and run the face recognition
    fr = FaceRecognition()
    fr.run_recognition()