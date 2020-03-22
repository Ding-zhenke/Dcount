# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:54:26 2020

@author: 31214
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
np.random.seed(0)
#信号
def sig(fs,fre,fun):
    t = np.arange(0,1,1/fs)
    def sin_wave(): #正弦波发生器
        sin = np.cos(2*np.pi*fre*t)
        return sin
    def square_wave():
        #方波发生器
        square = np.where(sin_wave()<0,1,0)
        return square
    def triangle_wave():
       #三角波发生器
        #离散信号的积分是卷积
        triangle = np.zeros(fs)
        square = square_wave()
        for i in range(0,len(square)-1):
            for j in range(0,i+1):
                triangle[i]+=square[j]
        return triangle
    def noise_wave():
        #噪声数据
        noise = np.random.normal(0,0.1,fs)+1.5
        return noise
    def funsig(ac):
        return ac()+noise_wave()
    if fun == 'sin':
        return funsig(sin_wave)
    elif fun =='square':
        return funsig(square_wave)
    elif fun =='triangle':
        return funsig(triangle_wave)
    elif fun =='noise':
        return funsig(noise_wave)
    else :
        print('input error!')
class myfft:
    def __init__(self,Fs,winname,fre,ADC_bit,ADC_Vpp,BW,unit):
        #信号频率
        self.f = fre 
        if(self.f == 0):
            print('please confirm the signal is DC')
        #采样率
        self.fs = Fs
        if(self.fs == 0):
            print('fs inputs error')
        #奈特斯奎定理检查
        if(self.fs < 2*self.f):
            print('it doesnt meet sample requirement' )
        #窗函数选择
        self.winname1 = winname
        if self.winname1 == None:
            self.powerrecover = 1
            self.altituderecover = 1
            self.ENBW = 1
        elif self.winname1 == 'hann':
            self.winname = signal.windows.hann
            self.powerrecover = 1.663
            self.altituderecover = 2
            self.ENBW = 1.5
        elif self.winname1 == 'blackman': 
            self.winname = signal.windows.blackman
            self.powerrecover = 1.812
            self.altituderecover = 2.381
            self.ENBW = 2
        elif self.winname1 =='hamming' :
            self.winname = signal.windows.hamming
            self.powerrecover = 1.586
            self.altituderecover = 1.852
            self.ENBW = 1.36
        elif self.winname1 == 'flattop': 
            self.winname = signal.windows.flattop
            self.powerrecover = 1.069
            self.altituderecover = 1.110
            self.ENBW = 3.77
        else :
            print('winname is flaut!!!')
        #窗函数数据
        if(self.winname == None):
            self.windata = np.ones(self.fs)
        else:
            self.windata = self.winname(self.fs,sym=0)
        #处理窗函数增益
        self.windata *= self.altituderecover
        #FFT后的单位
        self.unit=unit
        #ADC位数
        self.ADCbit = ADC_bit
        if(self.ADCbit == 0):
            print('ADCbit inputs error')
        self.ADCvpp = ADC_Vpp
        if(self.ADCvpp == 0):  
            print('ADCvpp inputs error')
        self.BW = 1.57*BW    #带宽
        if(self.BW == 0):
            print('BW inputs error')
        self.fullscale = 10*np.log10(self.ADCvpp ** 2 /8/50*1000)
    #加窗
    def win(self,data):
        return self.windata*data
    #画图
    def sigpaint(self,data):
        plt.subplot(121)
        plt.title('signal process')
        plt.xlabel("time（s）") 
        plt.ylabel("signal")
        t = np.linspace(0,1,self.fs)
        plt.plot(t,data,label='signal',color='blue')
        plt.legend()
        plt.show()
    #窗函数图像
    def winpaint(self,data):
        plt.subplot(121)
        plt.title('signal process')
        plt.xlabel("time（s）") 
        plt.ylabel("windows {0}".format(self.winname1))
        t = np.linspace(0,1,self.fs)
        plt.plot(t,self.windata,'b--',label='win',color='green')
        plt.legend()
        plt.show()
    #加窗后的图像
    def swpaint(self,data):
        plt.subplot(121)
        plt.title('signal process')
        plt.xlabel("time（s）") 
        plt.ylabel("windowsdata {0}".format(self.winname1))
        t = np.linspace(0,1,self.fs)
        plt.plot(t,self.win(data),label='after win',color='yellow')
        plt.legend()
        plt.show()
    #FFT后的图像
    def my_fft(self,data):
        #频率分辨率
        self.data = self.win(data)
        self.size = len(self.data)
        self.df = self.fs/self.size
        if self.df == 0:
            print("please confirm fs")
        #fft之后归一化
        self.da = abs(np.fft.fft(self.data)) / self.size*2
        self.da[0] /= 2 
        self.power = self.da ** 2
        if self.unit =='dB':
            self.da = 20*np.log10(self.da)
        elif self.unit =='dBFS':
            self.da = 20*np.log10(np.sqrt(2)*self.da)
        elif self.unit =='dBm':
            self.da = 20*np.log10(np.sqrt(2)*self.da)+self.fullscale
        elif self.unit =='dBmhz':
            self.da = 20*np.log10(np.sqrt(2)*self.da)+self.fullscale
            self.da -= 10*np.log10(self.fs/self.size*self.ENBW)         
            self.da -= 10*np.log10(2*self.BW/self.fs)
        elif self.unit == 'altitude':
            self.da = self.da            
        else:
            print("uint is error!")
        self.output = self.da
        self.fplain = np.arange(0,self.fs,self.df)
        
        plt.subplot(122)
        plt.title('FFT signal process')
        
        plt.xlabel("frequence（Hz)") 
        plt.ylabel("unit={0}".format(self.unit))
        plt.plot(self.fplain[0:int(self.size/2-1)],
                 self.output[0:int(self.size/2-1)],
                 label='fft')
        plt.legend()
        plt.show()
        return self.output
    
    
#参数设定    
Fs = 300
winname = 'hann'
fre = 30
ADC_bit = 12
ADC_Vpp = 3.3
BW = 1000
unit ='dB'#'dB'
wave ='sin'
#图像大小
plt.figure(figsize=(16,8))
#程序开始
sig1 = myfft(Fs,winname,fre,ADC_bit,ADC_Vpp,BW,unit)   
data = sig(Fs,fre,wave)
sig1.sigpaint(data)
sig1.winpaint(data)
sig1.swpaint(data)
out=sig1.my_fft(data)
