import cv2
from time import sleep

# var
video = cv2.VideoCapture("videoeditado.mp4")

# Video de Saida
fourcc = cv2.VideoWriter_fourcc(*'XVID')
saida = cv2.VideoWriter('resultado.avi', fourcc, 20.0, (int(video.get(3)), (int(video.get(4)))))

model = 'car.xml'
carro_cas = cv2.CascadeClassifier(model)

while True:

    ret, frame = video.read()
    delay = float(1 / 30)
    sleep(delay)

    cv2.rectangle(frame, (630, 370), (820, 520), (128, 128, 128), 2)
    roi = frame[370:520, 630:820]

    img_roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
    carros = carro_cas.detectMultiScale(img_roi, 1.1, 1)

    for (x, y, w, h) in carros:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame, 'Carro', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)

    saida.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
saida.release()
cv2.destroyAllWindows()
