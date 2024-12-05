# import numpy as np
# import cv2
# import pickle

# # Load Haar cascades
# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

# # Load trained recognizer and labels
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("./recognizers/face-trainner.yml")

# labels = {"person_name": 1}
# with open("pickles/face-labels.pickle", 'rb') as f:
#     og_labels = pickle.load(f)
#     labels = {v: k for k, v in og_labels.items()}

# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Unable to access the camera.")
#         break

#     # Resize frame for faster processing
#     small_frame = cv2.resize(frame, (800, 700))
#     gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,  # Slightly reduce to detect smaller faces
#         minNeighbors=5,
#         minSize=(50, 50)  # Minimum face size
#     )

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = small_frame[y:y+h, x:x+w]

#         # Recognize face
#         id_, conf = recognizer.predict(roi_gray)
#         if conf >= 40 and conf <= 80:  # Adjusted confidence range
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             name = labels.get(id_, "Unknown")
#             color = (255, 255, 255)  # White text color
#             stroke = 2
#             cv2.putText(small_frame, f"{name} ({int(conf)})", (x, y - 10), font, 0.8, color, stroke, cv2.LINE_AA)

#         # Save detected face
#         cv2.imwrite("detected_face_live.png", roi_color)

#         # Draw rectangle around face
#         rect_color = (0, 255, 0)  # Green rectangle
#         stroke = 2
#         cv2.rectangle(small_frame, (x, y), (x+w, y+h), rect_color, stroke)

#     # Display the result
#     cv2.imshow('Video Capture', small_frame)

#     # Optionally display grayscale for debugging
#     # cv2.imshow('Grayscale', gray)

#     # Exit on pressing 'q'
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()



# Implimentaion for image uplouding and output

# import numpy as np
# import cv2
# import pickle

# # Load the Haar cascades
# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

# # Load the trained recognizer and label mapping
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("./recognizers/face-trainner.yml")

# labels = {"person_name": 1}
# with open("pickles/face-labels.pickle", 'rb') as f:
#     og_labels = pickle.load(f)
#     labels = {v: k for k, v in og_labels.items()}

# # Load the image to process
# image_path = "full_frame_live.png"  # Replace with your image path
# frame = cv2.imread(image_path)

# if frame is not None:
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         # Region of interest (ROI) for the face
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]

#         # Recognize the face
#         id_, conf = recognizer.predict(roi_gray)
#         if conf >= 4 and conf <= 85:
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             name = labels[id_]
#             color = (255, 255, 255)  # White color
#             stroke = 2
#             cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

#         # Save the detected face
#         img_item = "detected_face.png"
#         cv2.imwrite(img_item, roi_color)

#         # Draw a rectangle around the face
#         color = (255, 0, 0)  # Blue color
#         stroke = 2
#         end_cord_x = x + w
#         end_cord_y = y + h
#         cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

#     # Display the resulting image
#     cv2.imshow('Processed Image', frame)
#     cv2.waitKey(0)  # Wait until a key is pressed
#     cv2.destroyAllWindows()
# else:
#     print("Error: Could not load image. Please check the image path.")


# # # impliment for longer distance 

# import numpy as np
# import cv2
# import pickle

# # Load Haar cascades
# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

# # Load trained recognizer and labels
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("./recognizers/face-trainner.yml")

# labels = {"person_name": 1}
# with open("pickles/face-labels.pickle", 'rb') as f:
#     og_labels = pickle.load(f)
#     labels = {v: k for k, v in og_labels.items()}

# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Unable to access the camera.")
#         break

#     # Resize frame for faster processing
#     small_frame = cv2.resize(frame, (800, 700))
#     gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,  # Slightly reduce to detect smaller faces
#         minNeighbors=5,
#         minSize=(50, 50)  # Minimum face size
#     )

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]

#         # Recognize face
#         id_, conf = recognizer.predict(roi_gray)
#         if conf >= 40 and conf <= 80:  # Adjusted confidence range
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             name = labels.get(id_, "Unknown")
#             color = (255, 255, 255)  # White text color
#             stroke = 2
#             cv2.putText(small_frame, f"{name} ({int(conf)})", (x, y - 10), font, 0.8, color, stroke, cv2.LINE_AA)

#         # Draw rectangle around face
#         rect_color = (0, 255, 0)  # Green rectangle
#         stroke = 2
#         cv2.rectangle(small_frame, (x, y), (x+w, y+h), rect_color, stroke)

#         # Save the whole frame instead of just the face
#         cv2.imwrite("detected_frame_live.png", small_frame)

#     # Display the result
#     cv2.imshow('Video Capture', small_frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()


# impliment for only save image 

import numpy as np
import cv2
import pickle

# Load the Haar cascades
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

# Load the trained recognizer and label mapping
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Resize frame for consistency (optional)
        resized_frame = cv2.resize(frame, (800, 700))

        # Display the frame
        cv2.imshow('Video Capture', resized_frame)

        # Save the entire frame (without any rectangles or predictions)
        cv2.imwrite("full_frame_live.png", resized_frame)

        # Load the image to process
        image_path = "face1long.jpg"  # Replace with your image path
        frame = cv2.imread(image_path)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Region of interest (ROI) for the face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Recognize the face
                id_, conf = recognizer.predict(roi_gray)
                if conf >= 4 and conf <= 85:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 255, 255)  # White color
                    stroke = 2
                    cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

                # Save the detected face
                img_item = "detected_face.png"
                cv2.imwrite(img_item, roi_color)

                # Draw a rectangle around the face
                color = (255, 0, 0)  # Blue color
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            # Display the resulting image
            cv2.imshow('Processed Image', frame)

        else:
            print("Error: Could not load image. Please check the image path.")

        # Exit on pressing 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()













# live 


# import numpy as np
# import cv2
# import pickle

# # Load the Haar cascades
# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# # Load the trained recognizer and label mapping
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("./recognizers/face-trainner.yml")

# # Load labels
# labels = {"person_name": 1}
# with open("pickles/face-labels.pickle", 'rb') as f:
#     og_labels = pickle.load(f)
#     labels = {v: k for k, v in og_labels.items()}

# # Initialize video capture
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Unable to access the camera.")
# else:
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read from the camera.")
#             break

#         # Save the captured frame, overwriting the previous one
#         image_path = "captured_image.png"
#         cv2.imwrite(image_path, frame)

#         # Process the saved image for face detection and recognition
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             # Region of interest (ROI) for the face
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frame[y:y+h, x:x+w]

#             # Recognize the face
#             id_, conf = recognizer.predict(roi_gray)
#             if conf >= 4 and conf <= 85:
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 name = labels.get(id_, "Unknown")
#                 color = (255, 255, 255)  # White color
#                 stroke = 2
#                 cv2.putText(frame, name, (x, y - 10), font, 1, color, stroke, cv2.LINE_AA)

#             # Save the detected face, overwriting the previous one
#             detected_face_path = "detected_face.png"
#             cv2.imwrite(detected_face_path, roi_color)

#         # Display the frame with detections
#         cv2.imshow('Video Capture', frame)

#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()





