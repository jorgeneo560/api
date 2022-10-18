import cv2
import cvzone
import numpy as np
#geotagging 
import piexif

#CONTADORES
i=0
cantON=0
cantOFF=0

thres = 0.55
nmsThres = 0.2
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#RANGOS LUCES
lower = np.array([0, 0, 200])
upper = np.array([179, 255, 255])


while True:
    luces=0
    success, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGR a HSV 
    mask = cv2.inRange(hsv, lower, upper) # aplicamos los filtros
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contornos luces

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
    except:
        pass
    #success, video = webcam_video.read() 
    
    # donde estan?
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3) #dibuja el rectangulo en rojo para las luces
                cv2.putText(img, 'LUCES ON',(20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2) #muestra que las luces estan prendidas si las encuentra
                luces=1
    else:
        cv2.putText(img, 'LUCES OFF',(20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), thickness=2) #luces apagadas si no encuentra 
    cv2.imshow("Image", img)
    x=cv2.waitKey(1)
    #sacar la foto cuando pulso c
    if x==99:
        #ver como recibo las coordenadas y en que formato
        if(luces==1):
            nombre='lucesON'+'captura'+str(i)+'.jpg'
            cantON=cantON+1
        else:
            nombre='lucesOFF'+'captura'+str(i)+'.jpg'
            cantOFF=cantOFF+1
        cv2.imwrite(nombre, img)
        i=i+1    
        #http://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf
        #https://piexif.readthedocs.io/en/latest/functions.html
        #las coordenadas se definen en RACIONALES (numerador,denominador) 
        #ver si la conversion la hago yo o jorge lo implementa en la API 
        exif_dict = piexif.load(nombre, key_is_name=False)
        gps_ifd = {piexif.GPSIFD.GPSLatitudeRef:"N", 
                    piexif.GPSIFD.GPSLatitude:[(120, 1), (37,1), (429996, 10000)],
                    piexif.GPSIFD.GPSLongitudeRef:"W",
                    piexif.GPSIFD.GPSLongitude:[(120, 1), (37,1), (42, 1)],
                    piexif.GPSIFD.GPSAltitudeRef:(0),
                    piexif.GPSIFD.GPSAltitude:(1700,1)
                }

        exif_dict = {"0th":{}, "Exif":{}, "GPS":gps_ifd, "1st":{}, "thumbnail":None}
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes,nombre)
        exif_dict = piexif.load(nombre)
        print(exif_dict)
    #cerrar el programa al pulsar s
    if x==115:
        print('cant luces prendidas=',cantON,'cant luces apagadas',cantOFF)
        break
cv2.destroyAllWindows
