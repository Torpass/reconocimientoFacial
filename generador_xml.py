import cv2 as cv
import os
import numpy as np 
#asd
#En esta sección del programa vamos a recolectar todas las imagenes generadas por el generador y las guardaremos como un archivo xml, para posteriormente ser usadas

data_ruta=(r'C:/Users/Pastor Jose/OneDrive/Documentos/Practicas en Python/Proyecto opencv/reconocimiento_facial/Data')
lista_data=os.listdir(data_ruta)

ids=[]
rostros_data=[]
id= 0 

#Ciclo for para recorrer cada carpeta y cada imagen que contenga la carpeta
for fila in lista_data:
    ruta_completa= data_ruta + '/' + fila   
    for archivo in os.listdir(ruta_completa):
        ids.append(id)
        rostros_data.append(cv.imread(ruta_completa+'/'+archivo, 0))
    id= id+1 

entrenaminedo_modelo1= cv.face.EigenFaceRecognizer_create() #EigenFaceRecognizer_create es una funsión para hacer un tratado de imagen más rapido

print('Inicio train')
entrenaminedo_modelo1.train(rostros_data, np.array(ids)) #'Train', funsión que se encarga de procesar todas la imágenes

entrenaminedo_modelo1.write('Entrenamiento1.xml') #Nombre del archivo

print('Listo')


