# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------

from scipy.signal.windows import general_cosine
from scipy.fftpack import next_fast_len
from numpy.fft import rfft, irfft
from numpy import argmax, mean, log, concatenate, zeros
from _common import rms_flat, parabolic
from ABC_weighting import A_weight

# ----------
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from pyqtgraph.Qt import QtGui, QtCore
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
import pyvisa
import numpy as np
from numpy import fft
import random
from PyQt5.QtWidgets import QMenuBar, QAction
from scipy.fft import fft, fftshift, fftfreq
     
class MatplotlibWidget(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)
        self.setGeometry(100,100,1121,855)
        loadUi("c:/Users/48604/Desktop/inzynierka/MDO 3024/final version/Oscilloscope/widget.ui",self)
        self.timeout_general = QtCore.QTimer()
        self.timeout_one = QtCore.QTimer()
        self.timeout_two = QtCore.QTimer()
        self.timeout_three = QtCore.QTimer()
        self.timeout_four = QtCore.QTimer()
        self.timeout_general.setInterval(100)
        self.timeout_one.setInterval(100)
        self.timeout_two.setInterval(100)
        self.timeout_three.setInterval(100)
        self.timeout_four.setInterval(100)
        self.timeout_general.timeout.connect(self.update_graph)
        self.timeout_one.timeout.connect(self.just_chOne)
        self.timeout_two.timeout.connect(self.just_chTwo)
        self.timeout_three.timeout.connect(self.just_chThree)
        self.timeout_four.timeout.connect(self.just_chFour)

        self.start()
        self.setWindowTitle("Oscilloscope")
        self.pushButton_CH1.clicked.connect(self.single_channel_information_one)
        self.pushButton_CH2.clicked.connect(self.single_channel_information_two)
        self.pushButton_CH3.clicked.connect(self.single_channel_information_three)
        self.pushButton_CH4.clicked.connect(self.single_channel_information_four)
       

        #startowanie wykresu
        self.pushButton_generate_random_signal.clicked.connect(self.update_graph)
        self.pushButton_reset.clicked.connect(self.cursor_reset)
        #stopowanie wykresu
        self.pushButton_generate_random_signal_stop.clicked.connect(self.stoped_graph)
        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        #Ustawianie tytułu
        self.title_lineEdit.returnPressed.connect(self.append_title)
        #Pokrętło sensitivity
        self.sensitivity_dial.setMinimum(0)
        self.sensitivity_dial.setMaximum(13)
        self.sensitivity_dial.valueChanged.connect(self.sensitivityDialMoved)
        
        #Pokrętło sampling
        self.sampling_dial.setMinimum(0)
        self.sampling_dial.setMaximum(7)
        self.sampling_dial.valueChanged.connect(self.samplingDialMoved)

        self.actionFFT_2.triggered.connect(self.FFTPlot)
        self.actionTHD.triggered.connect(self.THDPlot)
        self.radioButton_CursorOff.clicked.connect(self.cursormenuoff)
        self.radioButton_CursorOn.clicked.connect(self.cursormenu)
    def cursormenuoff(self):
        self.horizontalScrollBar_A.setMinimum(0)
        self.horizontalScrollBar_A.setMaximum(0)
        self.horizontalScrollBar_B.setMinimum(0)
        self.horizontalScrollBar_B.setMaximum(0)
    def cursormenu(self):
        self.horizontalScrollBar_A.valueChanged.connect(self.cursor)
        self.horizontalScrollBar_B.valueChanged.connect(self.cursor)
        self.horizontalScrollBar_A.setMinimum(-5000)
        self.horizontalScrollBar_A.setMaximum(5000)
        self.horizontalScrollBar_B.setMinimum(-5000)
        self.horizontalScrollBar_B.setMaximum(5000)
        self.horizontalScrollBar_A.setValue(0)
        self.horizontalScrollBar_B.setValue(0)
    def cursor(self):
        self.textBrowser_CursorAX.clear()
        self.textBrowser_CursorBX.clear()
        self.textBrowser_Cursordiff.clear()
        self.textBrowser_ACH1.clear()
        self.textBrowser_ACH2.clear()
        self.textBrowser_ACH3.clear()
        self.textBrowser_ACH4.clear()
        self.textBrowser_BCH1.clear()
        self.textBrowser_BCH2.clear()
        self.textBrowser_BCH3.clear()
        self.textBrowser_BCH4.clear()
        self.MplWidget.canvas.axes.clear()
        self.counter = 10000/self.sdivaxe
        t_A = [2*self.horizontalScrollBar_A.value()/self.counter,2*self.horizontalScrollBar_A.value()/self.counter]
        t_B = [2*self.horizontalScrollBar_B.value()/self.counter,2*self.horizontalScrollBar_B.value()/self.counter]
        t = np.linspace(-self.sdivaxe,self.sdivaxe,10000)
        self.textBrowser_CursorAX.append(str(t_A[0])+self.sunit)
        self.textBrowser_CursorBX.append(str(t_B[0])+self.sunit)
        self.textBrowser_Cursordiff.append(str(t_A[0]-t_B[0])+self.sunit)
        self.textBrowser_ACH1.append(str(self.plotA[self.horizontalScrollBar_A.value()+4999])[0:5])
        self.textBrowser_ACH2.append(str(self.plotB[self.horizontalScrollBar_A.value()+4999])[0:5])
        self.textBrowser_ACH3.append(str(self.plotC[self.horizontalScrollBar_A.value()+4999])[0:5])
        self.textBrowser_ACH4.append(str(self.plotD[self.horizontalScrollBar_A.value()+4999])[0:5])
        self.textBrowser_BCH1.append(str(self.plotA[self.horizontalScrollBar_B.value()+4999])[0:5])
        self.textBrowser_BCH2.append(str(self.plotB[self.horizontalScrollBar_B.value()+4999])[0:5])
        self.textBrowser_BCH3.append(str(self.plotC[self.horizontalScrollBar_B.value()+4999])[0:5])
        self.textBrowser_BCH4.append(str(self.plotD[self.horizontalScrollBar_B.value()+4999])[0:5])
        self.MplWidget.canvas.axes.plot(t,self.plotA, 'b') #
        self.MplWidget.canvas.axes.plot(t,self.plotB, 'g')
        self.MplWidget.canvas.axes.plot(t,self.plotC, 'r')
        self.MplWidget.canvas.axes.plot(t,self.plotD, 'violet') #

        self.cursory = [-100,100]
        self.MplWidget.canvas.axes.plot(t_A,self.cursory, 'k', linestyle='dashed')   #### i tutaj zmienić 
        self.MplWidget.canvas.axes.plot(t_B,self.cursory, 'y',linestyle='dashed') #### tutaj zmienić
        self.MplWidget.canvas.axes.legend(('CH1', 'CH2', 'CH3', 'CH4'),loc='upper right')
        self.MplWidget.canvas.axes.grid(True)
        self.MplWidget.canvas.axes.autoscale(enable= False)
        self.MplWidget.canvas.axes.set_ylim(-30,30)
        self.MplWidget.canvas.axes.set_xlim(-self.sdivaxe,self.sdivaxe)
        self.MplWidget.canvas.draw()
    def cursor_reset(self):
        self.horizontalScrollBar_A.setValue(0)
        self.horizontalScrollBar_B.setValue(0)

    def FFTPlot(self):
        N = 10000 
        self.sample = self.my_instrument.query('WFMOutpre:XINcr?')
        self.samplevalue = int(self.sample[0:self.sample.index(".")])
        self.samplemultiple = self.sample[self.sample.index("E"):-1]
        if self.samplemultiple[1] == "+":
            self.samplingfrequency = int(self.samplevalue)/(10**int(self.samplemultiple[self.samplemultiple.index("+")+1:]))
        if self.samplemultiple[1] == "-":
            self.samplingfrequency = int(self.samplevalue)/(10**int(self.samplemultiple[self.samplemultiple.index("-")+1:]))

        
        T = self.samplingfrequency ##wyznaczyć to sample
        x = np.linspace(0.0,N*T,N,endpoint=False)
        y = self.resultA
        yf = fft(y)/np.sqrt(2)
        xf = fftfreq(N,T)[:N//2]
        plt.plot(xf,2.0/N*np.abs(yf[0:N//2]))
        plt.grid()
        plt.show()
    # -------------------------------
    def THD(self,signal, fs):

        flattops = {
            'dantona3': [0.2811, 0.5209, 0.1980],
            'dantona5': [0.21557895, 0.41663158, 0.277263158, 0.083578947,
                        0.006947368],
            'SFT3F': [0.26526, 0.5, 0.23474],
            'SFT4F': [0.21706, 0.42103, 0.28294, 0.07897],
            'SFT5F': [0.1881, 0.36923, 0.28702, 0.13077, 0.02488],
            'SFT3M': [0.28235, 0.52105, 0.19659],
            'SFT4M': [0.241906, 0.460841, 0.255381, 0.041872],
            'SFT5M': [0.209671, 0.407331, 0.281225, 0.092669, 0.0091036],
            'FTSRS': [1.0, 1.93, 1.29, 0.388, 0.028],
            'FTNI': [0.2810639, 0.5208972, 0.1980399],
            'FTHP': [1.0, 1.912510941, 1.079173272, 0.1832630879],
            'HFT70': [1, 1.90796, 1.07349, 0.18199],
            'HFT95': [1, 1.9383379, 1.3045202, 0.4028270, 0.0350665],
            'HFT90D': [1, 1.942604, 1.340318, 0.440811, 0.043097],
            'HFT116D': [1, 1.9575375, 1.4780705, 0.6367431, 0.1228389, 0.0066288],
            'HFT144D': [1, 1.96760033, 1.57983607, 0.81123644, 0.22583558, 0.02773848,
                        0.00090360],
            'HFT169D': [1, 1.97441842, 1.65409888, 0.95788186, 0.33673420, 0.06364621,
                        0.00521942, 0.00010599],
            'HFT196D': [1, 1.979280420, 1.710288951, 1.081629853, 0.448734314,
                        0.112376628, 0.015122992, 0.000871252, 0.000011896],
            'HFT223D': [1, 1.98298997309, 1.75556083063, 1.19037717712, 0.56155440797,
                        0.17296769663, 0.03233247087, 0.00324954578, 0.00013801040,
                        0.00000132725],
            'HFT248D': [1, 1.985844164102, 1.791176438506, 1.282075284005,
                        0.667777530266, 0.240160796576, 0.056656381764, 0.008134974479,
                        0.000624544650, 0.000019808998, 0.000000132974],
            }
        signal -= mean(signal)

        window = general_cosine(len(signal), flattops['HFT248D'])
        windowed = signal * window
        del signal

        f = rfft(windowed)
        i = argmax(abs(f))
        true_i = parabolic(log(abs(f)), i)[0]
        print('Frequency: %f Hz' % (fs * (true_i / len(windowed))))

        print('fundamental amplitude: %.3f' % abs(f[i]))
        for x in range(2, 15):
            print('%.3f' % abs(f[i * x]), end=' ')

        THD = sum([abs(f[i*x]) for x in range(2, 15)]) / abs(f[i])
        print('\nTHD: %f%%' % (THD * 100))
        return


    # -------------------------------
    def THDPlot(self):
        print("test THD")
        self.THD(self.plotA, 5)





        
    def start(self):
        try:
            self.rm = pyvisa.ResourceManager()
            self.rm.list_resources()
            self.log_textBrowser.append("Zainicjalizowano połączenie: "+str(self.rm.list_resources()))
            self.my_instrument = self.rm.open_resource(self.rm.list_resources()[0])
        except IndexError:
            self.log_textBrowser.append("Utracono połączenie z urządzeniem")
        

    def initialize(self,channel):
        
        self.my_instrument.write(':DATA:SOUrce '+channel )
        self.my_instrument.write(':DATa:STARt 1') 
        self.my_instrument.write(':DATa:STOP 10000') ## sprawdzić czy jak wstawię np 5tys próbek to czy zadziała :
        self.my_instrument.write(':DATa:ENCdg ASCII')
        self.my_instrument.write(':DATa:WIDth 1')
        self.my_instrument.write('ACQuire:STOPAfter')
        self.my_instrument.write(':HEADer 1')
        self.my_instrument.write(':VERBose')
        self.my_instrument.query(':WFMOutpre?')
        self.my_instrument.write(':HEADer 0')
        self.my_instrument.write('CURV?')
        
    def single_channel_information_one(self):
        self.log_textBrowser.append('Uruchomiono pojedyńczy kanał: CH1 ')
    
    def chOne(self):
            # fs = 500
            # f = random.randint(1, 10)
            # length_of_signal = 4000
            # t = np.linspace(-2,2,length_of_signal)
            # self.cosinus_signal = np.cos(2*np.pi*f*t)
            # sinus_signal = np.sin(2*np.pi*f*t)
            # cosinus2_signal = np.cos(np.pi*f*t)
            # sinus2_signal =  np.sin(np.pi*f*t)
            # self.MplWidget.canvas.axes.plot(t,self.cosinus_signal, 'c')
            # self.MplWidget.canvas.draw()

                self.my_instrument.write(':SELect:CH1 1')
                self.initialize(channel='CH1')
                self.div = self.my_instrument.query(':WFMOutpre?') #wszystkie dane
                self.vdiv = self.div[self.div.index("coupling")+10:self.div.index("V/div")+5] #oś Y
                self.sdiv = self.div[self.div.index("V/div")+7:self.div.index("s/div")+5] #oś x
                self.resultA = self.my_instrument.query_ascii_values('CURV?')
                self.sdivaxe = float(self.sdiv[0:5]) *5
                self.sunit = self.sdiv[5:7]
                length_of_signal = 10000
                t = np.linspace(-self.sdivaxe,self.sdivaxe,length_of_signal)
                
                divisorY = [["1.000V/div",25.6],["500.0mV/div",48.0],["2.000V/div",12.0],["5.000V/div",4.8],
                ["10.00V/div",2.4],["200.0mV/div",112.5],["100.0mV/div",225.0],["50.00mV/div",450],
                ["20.00mV/div",1125],["10.00mV/div",2250],["5.000mV/div",4500],["2.000mV/div",11250],
                ["1.000mV/div",22500]]
                for x in range(len(divisorY)):
                    if self.vdiv == divisorY[x][0]:
                        axismax = divisorY[x][1]*4
                        axismax = 10
                        axis = plt.axis([0,10000,-axismax,axismax])
                        self.scale =  divisorY[x][1]
                    else: 
                        continue
                for x in range(length_of_signal):
                    self.resultA[x] /= self.scale

                self.MplWidget.canvas.axes.plot(t,self.resultA , 'b')
                self.MplWidget.canvas.draw()

    def just_chOne(self):
        self.pushButton_CH1.setStyleSheet('font-weight:bold; font-size: 10pt') 
        self.timeout_one.start()
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.grid(True)
        self.chOne()

    def single_channel_information_two(self):
        self.log_textBrowser.append('Uruchomiono pojedyńczy kanał: CH2 ')

    def chTwo(self):
            # fs = 500
            # f = random.randint(1, 10)
            # length_of_signal = 4000
            # t = np.linspace(-2,2,length_of_signal)
            # self.sinus_signal = np.sin(2*np.pi*f*t)
            # self.MplWidget.canvas.axes.plot(t,self.sinus_signal,'g')
            # self.MplWidget.canvas.draw()

            # try:
                self.my_instrument.write(':SELect:CH2 1')
                self.initialize(channel='CH2')
                self.div = self.my_instrument.query(':WFMOutpre?') #wszystkie dane
                self.vdiv = self.div[self.div.index("coupling")+10:self.div.index("V/div")+5] #oś Y
                self.sdiv = self.div[self.div.index("V/div")+7:self.div.index("s/div")+5] #oś x
                self.resultB = self.my_instrument.query_ascii_values('CURV?')
                self.couplingvalue = self.my_instrument.query('WFMOutpre:WFId?').split(', ')
                self.couplingvalue = self.couplingvalue[1][0:3]
                length_of_signal = 10000
                # t = np.linspace(-2,2,length_of_signal)
                t = np.linspace(-self.sdivaxe,self.sdivaxe,length_of_signal)
                divisorY = [["1.000V/div",25.6],["500.0mV/div",48],["2.000V/div",12],["5.000V/div",4.8],["10.00V/div",2.4],["200.0mV/div",112.5],["100.0mV/div",225.0],["50.00mV/div",450],["20.00mV/div",1125],["10.00mV/div",2250],["5.000mV/div",4500],["2.000mV/div",11250],["1.000mV/div",22500]]
                for x in range(len(divisorY)):
                    if self.vdiv == divisorY[x][0]:
                        axismax = divisorY[x][1]*4
                        axis = plt.axis([0,10000,-axismax,axismax])
                        self.scale =  divisorY[x][1]
                    else: 
                        continue
                for x in range(length_of_signal):
                    self.resultB[x] /= self.scale

                self.MplWidget.canvas.axes.plot(t,self.resultB, 'g')
                self.MplWidget.canvas.draw()

            # except Exception:
            #     self.log_textBrowser.append("Brak kanału 2")
            #     self.resultB = 0
            #     t = 10000
            #     self.MplWidget.canvas.axes.plot(t, self.resultB)
    def just_chTwo(self):
        self.pushButton_CH2.setStyleSheet('font-weight:bold; font-size: 10pt') 
        self.timeout_two.start()
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.grid(True)
        self.chTwo()


    def single_channel_information_three(self):
        self.log_textBrowser.append('Uruchomiono pojedyńczy kanał: CH3 ')
    def chThree(self):
            
                self.my_instrument.write(':SELect:CH3 1')
                self.initialize(channel='CH3')
                self.div = self.my_instrument.query(':WFMOutpre?') #wszystkie dane
                self.vdiv = self.div[self.div.index("coupling")+10:self.div.index("V/div")+5] #oś Y
                self.sdiv = self.div[self.div.index("V/div")+7:self.div.index("s/div")+5] #oś x
                self.resultC = self.my_instrument.query_ascii_values('CURV?')
                length_of_signal = 10000
                t = np.linspace(-self.sdivaxe,self.sdivaxe,length_of_signal)
                divisorY = [["1.000V/div",25.6],["500.0mV/div",48.0],["2.000V/div",12.0],["5.000V/div",4.8],["10.00V/div",2.4],["200.0mV/div",112.5],["100.0mV/div",225.0],["50.00mV/div",450],["20.00mV/div",1125],["10.00mV/div",2250],["5.000mV/div",4500],["2.000mV/div",11250],["1.000mV/div",22500]]
                for x in range(len(divisorY)):
                    if self.vdiv == divisorY[x][0]:
                        axismax = divisorY[x][1]*4
                        axis = plt.axis([0,10000,-axismax,axismax])
                        self.scale =  divisorY[x][1]
                    else: 
                        continue
                for x in range(length_of_signal):
                    self.resultC[x] /= self.scale

                self.MplWidget.canvas.axes.plot(t,self.resultC, 'r')
                self.MplWidget.canvas.draw()

            # except Exception:
            #      self.log_textBrowser.append("Brak kanału 3")
            #      self.resultC = 0
            #      t = 10000
            #      self.MplWidget.canvas.axes.plot(t, self.resultC)        
            # fs = 500
            # f = random.randint(1, 10)
            # length_of_signal = 4000
            # t = np.linspace(-2,2,length_of_signal)
            # self.cosinus2_signal = np.cos(np.pi*f*t)
            # self.MplWidget.canvas.axes.plot(t,self.cosinus2_signal,'r')
            # self.MplWidget.canvas.draw()

    def just_chThree(self):
            self.pushButton_CH3.setStyleSheet('font-weight:bold; font-size: 10pt') 
            self.timeout_three.start()
            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.grid(True)
            self.chThree()

    
    def single_channel_information_four(self):
        self.log_textBrowser.append('Uruchomiono pojedyńczy kanał: CH4 ')
    def chFour(self):
            
                self.initialize(channel='CH4')
                self.my_instrument.write(':SELect:CH4 1')
                self.div = self.my_instrument.query(':WFMOutpre?') #wszystkie dane
                self.vdiv = self.div[self.div.index("coupling")+10:self.div.index("V/div")+5] #oś Y
                self.sdiv = self.div[self.div.index("V/div")+7:self.div.index("s/div")+5] #oś x
                self.resultD = self.my_instrument.query_ascii_values('CURV?')
                length_of_signal = 10000
                t = np.linspace(-self.sdivaxe,self.sdivaxe,length_of_signal)
                divisorY = [["1.000V/div",25.6],["500.0mV/div",48],["2.000V/div",12],["5.000V/div",4.8],["10.00V/div",2.4],["200.0mV/div",112.5],["100.0mV/div",225.0],["50.00mV/div",450],["20.00mV/div",1125],["10.00mV/div",2250],["5.000mV/div",4500],["2.000mV/div",11250],["1.000mV/div",22500]]
                for x in range(len(divisorY)):
                    if self.vdiv == divisorY[x][0]:
                        axismax = divisorY[x][1]*4
                        axis = plt.axis([0,10000,-axismax,axismax])
                        self.scale =  divisorY[x][1]
                    else: 
                        continue
                for x in range(length_of_signal):
                    self.resultD[x] /= self.scale

                self.MplWidget.canvas.axes.plot(t,self.resultD, 'violet')
                self.MplWidget.canvas.draw()

            
            # fs = 500
            # f = random.randint(1, 10)
            # length_of_signal = 4000
            # t = np.linspace(-2,2,length_of_signal)
            # self.sinus2_signal =  np.sin(np.pi*f*t)
            # self.MplWidget.canvas.axes.plot(t,self.sinus2_signal, 'violet')
            # self.MplWidget.canvas.draw()

    def just_chFour(self):
            self.pushButton_CH4.setStyleSheet('font-weight:bold; font-size: 10pt') 
            self.timeout_four.start()
            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.grid(True)
            self.chFour()


    
    def append_title(self):
        self.MplWidget.canvas.axes.set_title(self.title_lineEdit.text())
        self.MplWidget.canvas.draw()
        self.title_lineEdit.clear()
    def sensitivityDialMoved(self):
        self.sensitivity_textBrowser.clear()
        divisorY = [["1V/div",3.0,[-4,4]],["0.5V/div",4,[-2,2]],["2V/div",2,[-8,8]],["5V/div",1,[-20,20]],["10V/div",0,[-40,40]],["0.2V/div",5,[-0.8,0.8]],["0.1V/div",6,[-0.4,0.4]],["0.05V/div",7,[-0.2,0.2]],["0.02V/div",8,[-0.08,0.08]],["100mV/div",9,[-0.04,0.04]],["0.005V/div",10,[-0.02,0.02]],["0.002V/div",11,[-0.008,0.008]],["0.001V/div",12,[-0.004,0.004]],["OFF",13]]
        for x in range(0,14):
            if self.sensitivity_dial.value() == divisorY[x][1]:
                self.sensitivity_textBrowser.append(divisorY[x][0])
                if self.sensitivity_dial.value() != 13:
                    self.MplWidget.canvas.axes.set_ylim(divisorY[x][2])
                    self.cursory = self.MplWidget.canvas.axes.get_ylim()[1]
                    
                    
                else:
                    continue

                self.MplWidget.canvas.draw()


    def samplingDialMoved(self):
        self.sampling_textBrowser.clear()
        self.divisorX = [["400us/div",0,[-2,2],],["200us/div",1,[-1,1]],["100us/div",2,[-0.5,0.5]],["40us/div",3,[-0.200,0.200]],["20us/div",4,[-0.1,0.1]],["10us/div",5,[-0.05,0.05]],["4us/div",6,[-0.02,0.02]],["OFF",7,]]
        for x in range(0,8):
            if self.sampling_dial.value() == self.divisorX[x][1]:
                self.sampling_textBrowser.append(self.divisorX[x][0])

                if self.sampling_dial.value() != 7:
                    self.MplWidget.canvas.axes.set_xlim(self.divisorX[x][2])
                else:
                    continue
            self.MplWidget.canvas.draw()
            



    def update_graph(self):
            self.textBrowser_Coupling.clear()
            self.pushButton_generate_random_signal.setStyleSheet('font-weight:bold; font-size: 10pt')
            self.timeout_general.start()
            legend_list = ["CH1","CH2","CH3","CH4"]
            self.sensitivity_dial.setValue(13)
            self.sampling_dial.setValue(7)
            self.MplWidget.canvas.axes.clear()
            self.pushButton_CH1.clicked.connect(self.just_chOne)
            self.pushButton_CH2.clicked.connect(self.just_chTwo)
            self.pushButton_CH3.clicked.connect(self.just_chThree)
            self.pushButton_CH4.clicked.connect(self.just_chFour)
            self.chOne()
            self.chTwo()
            self.chThree()
            self.chFour()
            self.textBrowser_Coupling.append(self.couplingvalue)
            self.MplWidget.canvas.axes.legend(('CH1', 'CH2', 'CH3', 'CH4'),loc='upper right')
            self.MplWidget.canvas.axes.set_xlim(right=self.sdivaxe, left = -self.sdivaxe)
            self.MplWidget.canvas.axes.set_xlabel("Time ["+self.sunit+"]")
            self.MplWidget.canvas.axes.set_ylabel("Voltage [V]")
            self.MplWidget.canvas.axes.grid(True)
            self.MplWidget.canvas.draw()

            
    def stoped_graph(self):
        self.pushButton_CH1.setStyleSheet('font-weight:normal; font-size: 10pt')
        self.pushButton_CH2.setStyleSheet('font-weight:normal; font-size: 10pt')
        self.pushButton_CH3.setStyleSheet('font-weight:normal; font-size: 10pt')
        self.pushButton_CH4.setStyleSheet('font-weight:normal; font-size: 10pt')
        self.pushButton_generate_random_signal.setStyleSheet('font-weight:normal; font-size: 10pt')
        self.pushButton_generate_random_signal_stop.setStyleSheet('font-weight:bold; font-size: 10pt')
        self.timeout_one.stop()
        self.timeout_two.stop()
        self.timeout_three.stop()
        self.timeout_four.stop()
        self.timeout_general.stop()
        # self.plotA = self.cosinus_signal
        # self.plotB = self.sinus_signal
        # self.plotC = self.cosinus2_signal
        # self.plotD = self.sinus2_signal
        self.plotA = self.resultA
        self.plotB = self.resultB
        self.plotC = self.resultC
        self.plotD = self.resultD
        self.log_textBrowser.append("Zatrzymano")
        # Cursors
        
        


        

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()