import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces
ahmed_image = face_recognition.load_image_file("faces/ahmed.png")
ahmed_encoding = face_recognition.face_encodings(ahmed_image)[0]
neha_image = face_recognition.load_image_file("faces/neha.png")
neha_encoding = face_recognition.face_encodings(neha_image)[0]

# Known face encodings and names
known_face_encoding = [ahmed_encoding, neha_encoding]
known_face_names = ["ahmed", "neha"]

# Open CSV to log attendance
now = datetime.now()
current_date = now.strftime("%y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# Main loop for face recognition
while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Record the attendance
            print(f"Attendance logged for {name}")
            now = datetime.now()
            lnwriter.writerow([name, now.strftime("%H:%M:%S")])

            # Draw rectangle and name on the frame
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Show the frame with the recognized faces
    cv2.imshow("Attendance", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close CSV file
video_capture.release()
f.close()
cv2.destroyAllWindows()
