import cv2
import face_recognition

#load known face encoding
known_face_encodings = []
known_face_names = []

#load face and name
known_person1_image = face_recognition.load_image_file("person1.jpg")

known_person1_encoding =face_recognition.face_encodings(known_person1_image)[0]

known_face_encodings.append(known_person1_encoding)

known_face_names.append("Naman Verma")

#initialize webcam
video_capture=cv2.VideoCapture(0)

while True:
    #capture frame by frame
    ret,frame = video_capture.read()

    #find all face location in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame,face_locations)

    #loop through each face found
    for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
        #check matches
        matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
        name="unknown"
        if True in matches:
            first_match_index=matches.index(True)
            name=known_face_names[first_match_index]

        #Draw a box around the face

        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,225),2)
        cv2.putText(frame,name,(left,top - 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,225,),2)

    #display the resulting frame
    cv2.imshow("Video",frame)        

    #break the loop
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

    #release webcam
video_capture.release()    
cv2.destroyAllWindows()