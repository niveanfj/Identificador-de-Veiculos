import cv2
from time import sleep

video = cv2.VideoCapture("videoeditado.mp4")      

model = 'car.xml'		
carro_cas = cv2.CascadeClassifier(model)    

while True:
    ret, frame = video.read()
    if ret:
        delay = float(1 / 35)  	
        sleep(delay)

        cv2.rectangle(frame, (630, 370), (820, 520), (0, 0, 0), 2)		
        roi = frame[370:520, 630:820]		

        img_roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
        carros = carro_cas.detectMultiScale(img_roi, 1.1, 1)  	

        for (x, y, w, h) in carros:
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 0), 2)		
            cv2.putText(frame, 'Carro', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)		
        cv2.imshow('frame', frame)	

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
video.release()		
cv2.destroyAllWindows()
