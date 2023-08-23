import cv2

#load the trained data
trained_cars_data = cv2.CascadeClassifier('car.xml')
trained_pedestrain_data = cv2.CascadeClassifier('pedestrian.xml')

#load the video files or the camera
video = cv2.VideoCapture('videoplayback (2).mp4')

while True:
    successful_frame_read, frame = video.read()

    if successful_frame_read:
    #convert to gray
        grayscaled_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    coordinates = trained_cars_data.detectMultiScale(grayscaled_video, scaleFactor=1.2, minNeighbors=5)
    coordinates2 = trained_pedestrain_data.detectMultiScale(grayscaled_video, scaleFactor=1.1, minNeighbors=7)
    

    #draw rectangle on each item identified as car or pedestrain
    for (x, y, w, h) in coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0 ), 2)
        if len(coordinates) >0 :
            cv2.putText(frame, 'car', (x, y+h+30), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,0,0))
    
    for (x_, y_, w_, h_)in coordinates2:
        cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), (0,0,255 ), 2)
        if len(coordinates2) > 0:
            cv2.putText(frame, 'human', (x_, y_+h_+30), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,0,0))



    #show frame
    cv2.imshow('cars_pedestrain_detector',frame)

    
    key= cv2.waitKey(1)
    if key==81 or key==113: #when ever q or Q is pressed
        break
video.release()
print('code completed!')