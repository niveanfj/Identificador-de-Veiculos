import cv2
from time import sleep

video = cv2.VideoCapture("videoeditado.mp4")      # Abrir video

# Video de Saida
fourcc = cv2.VideoWriter_fourcc(*'XVID')
saida = cv2.VideoWriter('resultado.avi', fourcc, 20.0, (int(video.get(3)), (int(video.get(4)))))

model = 'car.xml'		#  Definindo carros como modelo
carro_cas = cv2.CascadeClassifier(model) 

while True:

    ret, frame = video.read() 	#  Leitura dos frames
    delay = float(1 / 30)  	#Tempo pra leitura
    sleep(delay)

    cv2.rectangle(frame, (630, 370), (820, 520), (128, 128, 128), 2)		# Retangulo da area de interesse
    roi = frame[370:520, 630:820]		# Dimensoes da area de interesse

    img_roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
    carros = carro_cas.detectMultiScale(img_roi, 1.1, 1)  	# Detecção

    for (x, y, w, h) in carros:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 0), 2)		# Desenhar retangulo na detecção
        cv2.putText(frame, 'Carro', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)		#Nome do objeto reconhecido

    saida.write(frame)		# Grava o frame no video de saida
    cv2.imshow('frame', frame)	# Mostra o Processo

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#  libera os videos
video.release()		
saida.release()
cv2.destroyAllWindows()
