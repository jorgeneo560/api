# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(700, 300)

        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(10, 30, 231, 31))
        self.pushButton.setObjectName("pushButton")
        
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(250, 30, 191, 20))
        self.lineEdit.setObjectName("lineEdit")
        
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 60, 231, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox.setGeometry(QtCore.QRect(380, 65, 62, 22))
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(250, 50, 120, 51))
        self.label.setObjectName("label")
        
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 180, 231, 31))
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 90, 231, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        
        #self.label_2 = QtWidgets.QLabel(Dialog)
        #self.label_2.setGeometry(QtCore.QRect(180, 80, 71, 31))
        #self.label_2.setObjectName("label_2")
        
        #self.lineEdit_2 = QtWidgets.QLineEdit(Dialog)
        #self.lineEdit_2.setGeometry(QtCore.QRect(250, 90, 71, 30))
        #self.lineEdit_2.setObjectName("lineEdit_2")
        
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(10, 120, 120, 31))
        self.label_3.setObjectName("label_3")
        
        self.lineEdit_3 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_3.setGeometry(QtCore.QRect(10, 150, 631, 30))
        self.lineEdit_3.setObjectName("lineEdit_3")

        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(250, 180, 390, 30))
        self.listWidget.setObjectName("listWidget")

        self.retranslateUi(Dialog)
        #self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        #self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("API", "API"))
        self.pushButton.setText(_translate("Dialog", "Cargar archivo de telemetria"))
        self.pushButton_2.setText(_translate("Dialog", "Iniciar Streaming de datos"))
        self.label.setText(_translate("Dialog", "Tasa de emision"))
        self.pushButton_3.setText(_translate("Dialog", "Iniciar captura de video"))
        self.pushButton_4.setText(_translate("Dialog", "Establecer conexion"))
        #self.label_2.setText(_translate("Dialog", "Puerto de salida"))
        self.label_3.setText(_translate("Dialog", "Ultimo comando emitido"))
