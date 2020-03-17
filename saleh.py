# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:14:50 2020
copied from saleh in the mp_model 
@author: 31214
"""
import numpy as np
import matplotlib.pyplot as plt
def saleh_model( x,k=1 ):# successfully achieved
# y: output signal
# x: input signal 
# k: the nonlinearity 
    sh=x.shape
    L = np.max(sh)
    Out1=np.zeros(L)[:, np.newaxis] 
    Out2=np.zeros(L)[:, np.newaxis] 
    y=np.zeros(L)[:, np.newaxis] 
    Out1=np.abs(x)/(k+np.abs(x) ** 2)
    Out2=(np.pi/3)*np.abs(x)**2/(1+1*np.abs(x)**2)
    s = Out2 + np.angle(x)
    y=Out1*np.exp(1j*s)
    
    h=[1,0.085-0.0206j,-0.0229-0.027j]
    h=h/np.max(np.abs(h))
    for n in range(2,L-1):
        y[n-1]=h[0]*y[n-1]+h[1]*y[n-2]+h[2]*y[n-3]
    y[1]=h[0]*y[1]+h[1]*y[0]+h[1]*y[L-1]
    y[0]=h[0]*y[0]+h[1]*y[L-1]+h[2]*y[L-2]
    y=y/np.max(np.abs(y))
    print(y)
    plt.subplot(1,2,1)
    plt.title("model")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.scatter(abs(x),abs(y))
    plt.show()
    return y
def saleh_out( x ):
#   y= saleh_out( x )模拟功放，根据输入信号得到输出信号
#   x表示归一化输入信号，y表示得到的输出信号
#   归一化即 |H(jw)|/|max(H(jw))|
    sh=x.shape
    L = max(sh)
    Out1=np.zeros(L)
    Out2=np.zeros(L)
    y=np.zeros(L) 
    Out1=80*np.abs(x)/(2+1.8*np.abs(x)**2)
    Out2=(np.pi/3)*np.abs(x)**2/(1+np.abs(x)**2)
    s = Out2 + np.angle(x)
    y =Out1 * np.exp(1j*s)
    plt.subplot(1,2,2)
    plt.title("out")
    plt.xlabel("Input after normalization")
    plt.ylabel("Output")
    plt.scatter(abs(x),abs(y))    
    plt.show()
    return y 
'''
x = np.linspace(-5,5,100)
saleh_model(x)   
saleh_out(np.abs(x)/np.max(np.abs(x)) )
'''