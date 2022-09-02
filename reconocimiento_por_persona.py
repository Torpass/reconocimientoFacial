from re import S
import cv2 as cv 
import os 

#Archivo más importante del proyecto, aquí es donde vamos a usar todo lo antes generado, para poder clasificar, enmarcar y reconocer cada cara de cada individuo registrado 

data_ruta=(r'C:/Users/Pastor Jose/OneDrive/Documentos/Practicas en Python/Proyecto opencv/reconocimiento_facial/Data')
lista_data=os.listdir(data_ruta)

entrenaminedo_modelo1= cv.face.EigenFaceRecognizer_create()

entrenaminedo_modelo1.read('C:/Users/Pastor Jose/OneDrive/Documentos/Practicas en Python/Proyecto opencv/reconocimiento_facial/Entrenamiento1.xml')

ruidos = cv.CascadeClassifier("C:/Users/Pastor Jose/OneDrive/Documentos/Practicas en Python/Proyecto opencv/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml")

camara= cv.VideoCapture(0)

while True:
    _,captura= camara.read()
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)   
    id_captura= grises.copy()
    cara= ruidos.detectMultiScale(grises, 1.2, 3)
    for(x,y,e1,e2) in cara: 
        rostro_capturado= id_captura[y:y+e2, x:x+e1]
        rostro_capturado=cv.resize(rostro_capturado, (160,160), interpolation=cv.INTER_CUBIC)
        resultado= entrenaminedo_modelo1.predict(rostro_capturado) #Con la funsion Predict, lo que haces es hacer un estimado de cada cara encontrada la carpeta de Data, esto va a depender mucho de la luz, la posicion y la resolucion de la cámara, por eso es importante alimentar el xml con la mayor cantidad de caras y de variables posibles, para el predict sea más exacto por cada individuo. 
        cv.putText(captura, '{}'.format(resultado), (x,y-5), 1,1.3, (0,255,0),1, cv.LINE_AA) #Imprimer justo encima de la cara el resultado del predict (ronda generalmente entre los valores mil a dies mil)

#En esta sentencia comparamos los valores que arroja el predict y los comparamos con una valor determiando, en base a eso, el programa buscará en cada carpeta del xml, comparando con los otros predict y dando cómo output el nombre de la persona (O lo que se escriba en la carpeta que coincidió)
        if resultado[1] >= 2000:
            cv.putText(captura, '{}'.format(lista_data[resultado[0]]), (x,y-40), 2,1.1, (60, 60, 255),1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, x+e2), (255,0,0), 2)
        else:
            cv.putText(captura, 'no coincide\nPersona Desconocida', (x,y-20), 2,1.1, (0,255,0),1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, x+e2), (255,0,0), 2)
            
    cv.imshow('Resultado', captura)
    if cv.waitKey(1) == ord('s'):
        break

camara.release()
cv.destroyAllWindows()