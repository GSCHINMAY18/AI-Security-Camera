#Face and body detection function:

def face_and_body_detection():
    
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import time
    import sys
    import datetime
    import csv
    
    cap = cv2.VideoCapture(0)
    count = 0
    face_detected_frames = 0
    total_frames = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    

    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    recording = False

    frame_size = (int(cap.get(3)),int(cap.get(4)))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    output_file = open("face_detection_timings.csv", "w")
    output_writer = csv.writer(output_file)
    prev = False

    
    while True:
        ret , frame = cap.read()

        frame = cv2.flip(frame , 1)

        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray , 1.1 ,5)

        #eyes = eye_cascade.detectMultiScale(gray , 1.1 ,5)

        #body = body_cascade.detectMultiScale(gray , 1.1 ,5)

        if(len(faces) > 0):
            recording = True
            face_detected_frames += 1
        else:
            recording = False

        total_frames += 1

        for (x,y,w,h) in faces:
            cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0) , 2)
            #roc = (gray[x:x+w,y:y+h])

        current_time = datetime.datetime.now()

        cv2.imshow('frame',frame)

        if recording and not prev:
            output_writer.writerow([current_time])
            t = current_time.strftime("%H-%M-%S")
            out = cv2.VideoWriter(f'Recording({current_time.strftime("%d-%m-%y")})({t}).mp4' , fourcc , 10 , frame_size)
            out.write(frame)
            prev = True
        if recording :
            out.write(frame)
            prev = True
        elif prev:
            out.release()
            count +=1
            prev = False

        if(cv2.waitKey(1) == ord('q')):
            cap.release()
            cv2.destroyAllWindows()
            out.release()
            output_file.close() # Close the file when done
            break

    accuracy = face_detected_frames/total_frames
    print(f"Face detection rate: {accuracy}")
    print('Your Recordings have been saved to your current directory')
    return

#Intrusion Detection function:

def Intrusion_detection():
    
    import cv2
    from cvzone.PoseModule import PoseDetector
    import time
    import sys
    import datetime
    
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 480)

    prev = False
    recording = False
    count = 0
    frame_size = (int(cap.get(3)),int(cap.get(4)))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    output_file = open("face_detection_timings.csv", "w")

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        imlist, bbox = detector.findPosition(img)

        if len(imlist) > 0:
            recording = True

        else:
            recording = False

        current_time = datetime.datetime.now()

        if recording and not prev:
            t = current_time.strftime("%H-%M-%S")
            out = cv2.VideoWriter(f'Recording({current_time.strftime("%d-%m-%y")})({t}).mp4' , fourcc , 10 , frame_size)
            print(current_time)
            out.write(img)
            prev = True

        if recording :
            out.write(img)
            prev = True

        elif prev:
            out.release()
            count +=1
            prev = False

        if(cv2.waitKey(1) == ord('q')):
            cap.release()
            cv2.destroyAllWindows()
            out.release()
            output_file.close() # Close the file when done
            break

#Crowd monitoring function:

def Crowd_monitoring():
    
    import cv2
    import matplotlib.pyplot as plt
    from datetime import datetime
    # Initialize video capture from default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Initialize lists to store timestamps and face counts
    timestamps = []
    face_counts = []

    while True:
        # Capture frames continuously
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use OpenCV's built-in face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))

        # Count the number of faces
        num_faces = len(faces)
        face_counts.append(num_faces)

        # Record timestamp
        timestamps.append(datetime.now())

        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {num_faces}', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Terminate the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    

    # Plot the number of faces detected over time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, face_counts, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Number of Faces Detected')
    plt.title('Number of Faces Detected vs Time')
    plt.grid(True)
    plt.show()
    return

#License Plate detection function:

def LicensePlate_detection():
    import cv2
    import keyboard
    import numpy as np
    import imutils
    import sys
    import pytesseract
    import pandas as pd
    import time

    def capture_image():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        ret, frame = cap.read()
        if ret:
            cv2.imwrite("captured_image.jpeg", frame)
            print("Image captured and saved as captured_image.jpeg")
        else:
            print("Error: Failed to capture image.")

        cap.release()

    def display_camera():
        #display_camera()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def listen_for_keypress():
        print("Press 'c' to capture an image from the webcam.")
        while True:
            try:
                if keyboard.is_pressed('c'):
                    capture_image()
                    break
            except:
                print("Error occurred while detecting key press.")

    display_camera()

    listen_for_keypress()

    image = cv2.imread('captured_image.jpeg')

    image = imutils.resize(image, width=500)

    cv2.imshow("Original Image", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("1 - Grayscale Conversion", gray)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #cv2.imshow("2 - Bilateral Filter", gray)

    edged = cv2.Canny(gray, 170, 200)
    #cv2.imshow("4 - Canny Edges", edged)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
    NumberPlateCnt = None 

    count = 0
    for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  
                NumberPlateCnt = approx 
                break

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
    new_image = cv2.bitwise_and(image,image,mask=mask)
    cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
    cv2.imshow("Final_image",new_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Configuration for tesseract
    config = ('-l eng --oem 1 --psm 3')

    # Run tesseract OCR on image
    text = pytesseract.image_to_string(new_image)
    print(text)
    #Data is stored in CSV file
    raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
            'v_number': [text]}
    df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
    print(df)
    df.to_csv('data.csv',mode='a')

    # Print recognized text
    print(text)

    cv2.waitKey(0)

#Main Program:

def main():
    print('Welcome to Ai Camera Setup')
    flag = True;
    while(flag):
        mode = int(input('Enter mode of operation of the camera 1.Home Surveillance 2.Intrusion Detection 3.Crowd Monitoring 4.License Plate Detection : '))
        match(mode):
            case 1:
                face_and_body_detection()
            case 2:
                Intrusion_detection()
            case 3:
                Crowd_monitoring()
            case 4:
                LicensePlate_detection()
            case 5:
                flag = False
    return
main()

