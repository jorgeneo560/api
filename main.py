from concurrent.futures import thread
from email import message
from fileinput import filename
from operator import index, truediv
from select import select
import sys
from tkinter.tix import Tree
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
import tkinter
import tkinter.filedialog
import socket
import threading as th
from threading import Thread
from time import sleep
from tkinter import messagebox

from cv2 import VideoCapture


from GUI import Ui_MainWindow, Ui_MainWindow
from tkinter import filedialog
from tkinter import *


import cv2
import cvzone
import numpy as np
#geotagging 
import piexif



TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024
MESSAGE = "Hello, World!"
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)



def Stream():
    while True:
        while main_win.Transmit:
            f = open(main_win.filename, "r")
            #list=f.readlines()
            x = f.readlines()
            f.close()
            while len(x)>=main_win.Lastline:
                #data=f.readline(main_win.Lastline+1)
                data=x[main_win.Lastline-1]
                main_win.Lastline=main_win.Lastline+1
                #MESSAGE2 = "Hello, World!xx"
                #main_win.ui.lineEdit_3.setText(data)
                main_win.conn.send(data.encode()) #out of scope
                #main_win.conn.send(MESSAGE2.encode()) #out of scope
            #f.close()
            main_win.ui.lineEdit_3.setText(data)
            sleep(main_win.ui.doubleSpinBox.value())
    
    

T = th.Timer(0.10, Stream,args=(''))



def Videorecognition(index):
    
    i=0
    cantON=0
    cantOFF=0
    thres = 0.55
    nmsThres = 0.2
    cap = cv2.VideoCapture(index)
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
    while main_win.Videocapturestarted:
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
        #exif_dict = piexif.load(nombre, key_is_name=False)
        #gps_ifd = {piexif.GPSIFD.GPSLatitudeRef:"N",
        #            piexif.GPSIFD.GPSLatitude:[(120, 1), (37,1), (429996, 10000)],
        #            piexif.GPSIFD.GPSLongitudeRef:"W",
        #            piexif.GPSIFD.GPSLongitude:[(120, 1), (37,1), (42, 1)],
        #            piexif.GPSIFD.GPSAltitudeRef:(0),
        #            piexif.GPSIFD.GPSAltitude:(1700,1)
        #        }
        #exif_dict = {"0th":{}, "Exif":{}, "GPS":gps_ifd, "1st":{}, "thumbnail":None}
        #exif_bytes = piexif.dump(exif_dict)
        #piexif.insert(exif_bytes,nombre)
        #exif_dict = piexif.load(nombre)
        #print(exif_dict)
        #cerrar el programa al pulsar s
        #if not main_win.Videocapture:
            #print('cant luces prendidas=',cantON,'cant luces apagadas',cantOFF)
            #cap.release()
    cap.release()
    cv2.destroyAllWindows()
    return

            
def task():
    # block for a moment
    sleep(1)
    # display a message
    print('This is from another thread')
#thread = Thread(target=task, args=(arg1, arg2))
#thread = Thread(target=task)
#T2 = Thread(target=Videorecognition,args=[index])






class MainWindow:
    geotag=False
    Transmit = False
    Lastline=0
    filename=""
    Started=False
    #Videocapturethreadstarted=False
    Videocapturestarted=False
    Connected=False
    index=0
    videodevicesindex = []
    conn=socket.AF_INET
    addr=socket.SOCK_STREAM
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)
        self.ui.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.ui.pushButton_2.clicked.connect(self.transmit_clicked)
        self.ui.pushButton_3.clicked.connect(self.Videocapture)
        self.ui.pushButton_4.clicked.connect(self.OpenConn)
        self.ui.lineEdit.setText("C:/Users/drons/Documents/GitHub/build-qgroundcontrol-Desktop_Qt_5_15_2_MSVC2019_64bit-Debug/staging/Data.txt")
        self.filename="C:/Users/drons/Documents/GitHub/build-qgroundcontrol-Desktop_Qt_5_15_2_MSVC2019_64bit-Debug/staging/Data.txt"
        self.ui.doubleSpinBox.setValue(1.00)
        index = 0
        
        while True:
            cap = cv2.VideoCapture(index)
            try:
                #list=(cap.getBackendName())
                self.ui.listWidget.addItem(cap.getBackendName())
                self.videodevicesindex.append(index)
                cap.release()
                index += 1
            except:
                break
        
        #print (arr)
    def show(self):
        self.main_win.show()

    def on_pushButton_clicked(self):
        filename = filedialog.askopenfilename(
            initialdir="/", title="Select file", filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
        self.ui.lineEdit.setText(filename)
        self.filename=filename

    def transmit_clicked(self):
        if not self.Connected:
            messagebox.showerror('Error', 'Error: No hay una conexion establecida')
            return
        if self.filename == "":
            messagebox.showerror('Error', 'Error: No ha seleccionado el archivo')
            return
        if (not self.Transmit):
            self.Transmit = True
            self.ui.pushButton_2.setText("Detener Streaming de datos")
            # iniciar transmicion de datos
            print(T.interval)
            if (not self.Started):
                T.start()
                self.Started=True
        else:
            self.Transmit = False
            self.ui.pushButton_2.setText("Iniciar Streaming de datos")

    def OpenConn(self):
        self.conn, self.addr = s.accept()
        #conn, addr = s.accept()
        #data = "Hello, World!"
        #conn.send(data.encode()) 
        self.Connected=True
    
    def Videocapture(self):
        if self.Videocapturestarted:
            print ("xxx")
            self.ui.pushButton_3.setText("Iniciar captura de video")
            self.Videocapturestarted=False
            return
        else:
            print ("yyyy")
            if self.ui.listWidget.selectedIndexes()==[]:
                messagebox.showerror('Error', 'Error: No hay fuente de video seleccionada')
                return
            self.ui.pushButton_3.setText("Detener captura de video")
            self.Videocapturestarted=True
            selectedindex=self.ui.listWidget.selectedIndexes()
            print (selectedindex[0].row())
            thread = th.Thread(target=Videorecognition, args=[selectedindex[0].row()])
            thread.start()
            self.Videocapture=True
            return



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    import socket
    TCP_IP = '127.0.0.1'
    TCP_PORT = 5005
    BUFFER_SIZE = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    #s.setblocking(0)
    #conn, addr = s.accept()
    sys.exit(app.exec_())
