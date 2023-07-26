import face_recognition
import os
import cv2
import threading
from queue import Queue

KNOWN_FACES_DIR = 'known_faces'
RTSP_URL = os.environ['RTSP_URL']

TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'
SKIP_FRAMES = 5

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
video = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# Returns (R, G, B) from name
def name_to_color(name):
    color = [(ord(c.lower()) - 97) * 8 for c in name[:3]]
    return color

print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

# Queue to store recognized faces and their locations
face_queue = Queue()

# Thread function for face recognition
def recognize_faces(image, locations):
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            face_queue.put((match, top_left, bottom_right))


# Function to draw text on the image from the main thread
def draw_text_on_image(image, match, top_left, bottom_right):
    color = name_to_color(match)
    cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

    top_left = (top_left[0], top_left[1] - 22)
    bottom_right = (bottom_right[0], top_left[1] + 22)

    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

    cv2.putText(image, match, (top_left[0] + 10, top_left[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

# Thread function to read frames from video stream
def read_frames_thread():
    frame_count = 0
    while True:
        ret, image = video.read()

        # Skip frames if needed
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        # Resize image for faster processing
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(image, model=MODEL)

        # Use a separate thread for face recognition to run concurrently
        recognition_thread = threading.Thread(target=recognize_faces, args=(image, locations))
        recognition_thread.start()
        recognition_thread.join()  # Wait for face recognition to finish before drawing

        # Check if any recognized faces in the queue and draw them on the image
        while not face_queue.empty():
            match, top_left, bottom_right = face_queue.get()
            draw_text_on_image(image, match, top_left, bottom_right)

        # Show image with face boxes
        cv2.imshow('Face Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Start the thread to read frames from the video stream
read_frames_thread()

