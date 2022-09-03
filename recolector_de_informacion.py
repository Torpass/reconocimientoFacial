import cv2 as cv   
import os 
import imutils

# Este programa reconoce los patrones en la cámara, toma fotos y las guarda en la ruta seleccionada, se puede denominar como un generador de fotos para alimentar a la inteligencia posteriormente 

modelo='face_1' #Nombre de la carpeta ha guardar
ruta1=r'C:\Users\Pastor Jose\OneDrive\Documentos\Practicas en Python\Proyecto opencv\reconocimiento_facial\Data' #Ruta donde se va a guardar la carpeta
ruta_completa= ruta1+ '/' + modelo 
if not os.path.exists(ruta_completa):
    os.makedirs(ruta_completa) #creacion de la carpeta, pregunta si existe el nombre para ingresar las eimagenes, sino, crea la carpeta


ruidos = cv.CascadeClassifier(r"C:/Users/Pastor Jose/OneDrive/Documentos/Practicas en Python/Proyecto opencv/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml") #Ruidos es un conjunto de imagenes entrenadas por openCv, para poder determinar y eliminar loos ruidos de la imagen (LLamese ruido a objetos que no sean caras (Vasos, cuadernos, cuadros, etc...))

camara=cv.VideoCapture(0) #Captura de video por la cámara local

id=0 
#Ciclo para confirmar que está capturando video 
while True:
    respuesta,captura= camara.read() #función para leer cada frame del video
    if respuesta==False:
        break

    captura= imutils.resize(captura,width=640) #disminuyendo la resolucion de cada imagen, para que pesen menos


 #Estas 3 funciones lo que hacen es hacerle un tratado a la imagen y detectar los ruidos antes mensionados
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    id_captura=captura.copy()
    caras=ruidos.detectMultiScale(grises,1.1,3) #detectMultiscale es la funsión principal del programa, porque es la elimina todos los ruidos y reconoce los patrones de la cara con operaciones matemáticas
#----------------------------------------------------

#Este ciclo for es para enmarcar las caras reconocidas, definirle un nuevo tamaño, y guardarlas
    for(x,y,e1,e2) in caras:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0), 2)
        rostro_capturado= id_captura[y:y+e2, x:x+e1]
        rostro_capturado=cv.resize(rostro_capturado, (160,160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(ruta_completa+'/R{}.jpg'.format(id), rostro_capturado)
        id= id+1

    cv.imshow('Resultado', captura)

    if id == 999 : #Catidad de fotos que desea tomar 
        break

    if cv.waitKey(1) ==ord('s'):
        break
camara.release()
cv.destroyAllWindows() 
