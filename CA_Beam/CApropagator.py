#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:37:27 2022

@author: mateo
8
"""
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import imageio
import os
from scipy.signal import find_peaks
from IPython.display import Audio
import shutil
import time
import progressbar
    
class NCApropagator():
    
    def __init__(self,dt,L,dist,ExtraLong=1.5,phi=1,c = 343,rho = 1.225,halfPlane=False):
        self.phi=phi
        self.c=c
        self.rho=rho
        self.dx=np.sqrt(phi)*c*dt 
        self.dt=dt
        self.dist=dist
        self.L=L
        self.cntr=L/2
        self.l=int(np.sqrt((L/2+ExtraLong*dist)**2+(ExtraLong*dist)**2)/self.dx)
        self.radDict=None
        self.halfPlane=halfPlane
        
        
    def FFTpeaks(self,V,Num,min_val=0,dist=50):
        #V=np.array(V)
        dt=self.dt
        fs=int(1/dt)
        n = len(V)
        fft_norm=2.0/n
        FFT=np.fft.fft(V)
        N = len(FFT)
        phase=np.angle(FFT)[:(N // 2)]
        FFT_pow = np.abs(fft_norm*FFT)
        FFT_pow=FFT_pow[:(N // 2)]
        #f_pos=np.fft.fftfreq(V.shape[-1],dt)[:(N // 2)]
        f_pos=np.fft.fftfreq(len(V),dt)[:(N // 2)]
        peaks, prop = find_peaks(FFT_pow, height=min_val,distance=dist)
        f=f_pos[peaks]
        ph=phase[peaks]
        u=prop['peak_heights']
        l=len(u)
        U=sorted(u)[l-Num:l]
        F=[f[i] for i,_ in sorted(enumerate(u),key=lambda x: x[1])]
        F=F[l-Num:l]
        Ph=[ph[i] for i,_ in sorted(enumerate(u),key=lambda x: x[1])]
        Ph=Ph[l-Num:l]
        if len(U)<Num:
            U=np.zeros(Num)
            Ph=np.zeros(Num)
            F=np.zeros(Num)+440
        return U,F,Ph
    
    def get_u_f_ph(self,VelMat,N):
        U = []
        F = []
        Ph = []
        for v in VelMat:
            u,f,ph=self.FFTpeaks(v,N)
            U.append(u)
            F.append(f)
            Ph.append(ph)
        self.N=N
        self.U = np.array(U).T
        self.F = np.array(F).T
        self.Ph = np.array(Ph).T
    
    def P_r(self,ph=None,u = 2,r0=0.05,f=200,l=500,plotting=False):
        phi = self.phi
        c = self.c
        rho = self.rho
        dx = self.dx

        omega=2*np.pi*f #Angular frequency
        lamb=c/f      #longitud de onda
        k=2*np.pi/lamb #número de onda
        
        if ph is not None:
            theta=ph
        else:
            theta = np.arctan(1/(k*r0))  #Phase theta
        
        dt=np.sqrt(phi)*dx/c #paso temporal
        T=l*dx/(c)
        lenT=T/dt
        
        Cte=lambda r: rho*c*k*r0*r0*u/(r*np.sqrt(1+(k*r0)**2))
    
        p=lambda r,t: Cte(r)*np.cos(omega*t-k*r+theta)
        
        #Gird
        Y=np.zeros([int(lenT),l+1])
        P=np.zeros([int(lenT),l+1])
        for t in range(len(Y)):
            Time=t*dt-dt
            Y[t,0]=p(r0,Time)*r0
            for r in range(len(Y[0])-1):
                if Time == -dt:
                    Y[t,r]=0;P[t,r]=0
                if Time == 0:
                    if r == 0:
                        Y[t,r]=p(r0,Time)*r0
                        P[t,r]=p(r0,Time)
                    else:
                        Y[t,r]=0;P[t,r]=0
                if Time>0:
                    if r == 0:
                        Y[t,r]=p(r0,Time)*r0
                        P[t,r]=p(r0,Time)
                    else:
                        Y[t,r]=phi*(Y[t-1,r+1]+Y[t-1,r-1]+(2/phi-2)*Y[t-1,r]-(1/phi)*Y[t-2,r])
                        P[t,r]=Y[t,r]/(r*dx+r0)
        
        if plotting==True:
            plt.imshow(P)
            plt.colorbar(label='presión en Pa') 
        
        return P
        
    def CAdictFun(self,R0,U,F,Ph, EqPhase=False):
        print('\nCreando diccionarios de AC')
        start=time.time()
        l=self.l
        CAdict={}
        for i in range(len(U)):
            r0=R0[i]
            u=U[i]
            f=F[i]
            if EqPhase:
                ph=None
            else:
                ph=Ph[i]
            Pr=self.P_r(ph=ph,u=u,r0=r0,f=f,l=l)
            CAdict[i]=Pr
        now=time.time()
        print(f'Total time={now-start:0.5f}')
        return CAdict
    
    def CApropagation(self,CAdict,Xbar,ct=5,delta=0.05):
        dx = self.dx
        d=self.dist
        c = self.c
        dt=self.dt
        Ttot=len(CAdict[0])
        CalcTime=int(Ttot/ct)
        fpx=np.arange(self.cntr-(d),self.cntr+d,delta)
        #Half_Plane:
        if self.halfPlane:
            fpy=np.arange(0,d,delta)
        else:
            fpy=np.arange(-d,d,delta)
        origin=np.array([0,1])
        radDict={}
        cosDict={}
        TotlaFP=fpy.shape[0]*fpx.shape[0]
        if self.radDict is None:
            print('Haciendo radDict:')
            start=time.time()
            for i in range(len(Xbar)):
                radArray=np.zeros((fpy.shape[0],fpx.shape[0]))
                cosArray=np.zeros((fpy.shape[0],fpx.shape[0]))
                for k in range(len(fpx)):
                    for l in range(len(fpy)):
                        xbar=Xbar[i]
                        bp=np.array([xbar,0])
                        x=fpx[k]
                        y=fpy[l]
                        fp=np.array([x,y])
                        h=fp-bp
                        r=np.linalg.norm(bp-fp)
                        r_int=round(r/dx)
                        cosb=np.matmul(origin,h)/(np.linalg.norm(origin)*np.linalg.norm(h))
                        radArray[l,k]=r_int
                        cosArray[l,k]=cosb
                radDict[i]=radArray
                cosDict[i]=cosArray
                print(f'{i+1} de {len(Xbar)}', end=', ')
            self.radDict=radDict
            self.cosDict=cosDict
            now=time.time()
            print(f'Total time={now-start:0.5f}')
        print(f'Total propagtion time in seconds={self.l*dx/(c):0.5f}')
        PropagationT={}
 
        for t in range(1,Ttot):
            if t%CalcTime==0 or t==Ttot-1:
                start=time.time()
                Time=t*dt
                Propt=np.zeros((fpy.shape[0],fpx.shape[0]))
                for k in range(len(fpx)):
                    for l in range(len(fpy)):
                        pi=0
                        for i in range(len(Xbar)):
                            r_int=self.radDict[i][l,k]
                            cosb=self.cosDict[i][l,k]
                            pi+=CAdict[i][t,int(r_int)]*cosb
                        Propt[l,k]=pi     
                PropagationT[Time]=Propt
                now=time.time()
                print(f'{Time:0.5f}_{now-start:0.5f}',end=', ')
        return PropagationT
    
    def FPmeasure(self,FP,CAdict,Xbar):
        start=time.time()
        fp=np.array(FP)
        start=time.time()
        dx = self.dx
        d=self.dist
        c = self.c
        dt=self.dt
        Ttot=len(CAdict[0])
        origin=np.array([0,1])
        FPm={}
        #T=[]
        for t in range(Ttot):
            Time=t*dt
            #T.append[Time]
            pi=0
            for i in range(len(Xbar)):
                xbar=Xbar[i]
                bp=np.array([xbar,0])
                r=np.linalg.norm(bp-fp)
                r_int=round(r/dx)
                h=fp-bp
                cosb=np.matmul(origin,h)/(np.linalg.norm(origin)*np.linalg.norm(h))
                pi+=CAdict[i][t,int(r_int)]*cosb
            FPm[Time]=pi     
        now=time.time()
        print(f'Mode_Measure_finished_{now-start:0.5f}',end=', ')
        return FPm
    
    def NCApropagation(self,R0,Xbar,ct=5,delta=0.05,EqPhase=False):
        print(f'Iniciando propagación de {self.N} modos')
        startTotal=time.time()
        U=self.U
        F=self.F
        Ph=self.Ph
        d=self.dist
        NPropagation={}
        fpx=np.arange(self.cntr-(d),self.cntr+d,delta)
        if self.halfPlane:
            fpy=np.arange(0,d,delta)
        else:
            fpy=np.arange(-d,d,delta)
        for i in range(len(U)):
            u=U[i]
            f=F[i]
            ph=Ph[i]
            CADict=self.CAdictFun(R0,u,f,ph,EqPhase)
            Prop=self.CApropagation(CADict,Xbar,ct,delta)
            NPropagation[i]=Prop
        print('\nCalculando contribución de todos los modos:', end=' ')
        start=time.time()
        Total_Propagation={}
        ceros=np.zeros((fpy.shape[0],fpx.shape[0]))
        for t in list(NPropagation[0].keys()):
            
            Propt=np.zeros((fpy.shape[0],fpx.shape[0]))
            
            for i in list(NPropagation.keys()):
                
                Propt=Propt+NPropagation[i][t]
            
            Total_Propagation[t] = Propt
        now=time.time()
        print(f'time={now-start:0.5f}')  
        
        nowTotal=time.time()
        
        print(f'Total time={ nowTotal - startTotal:0.5f}')
        Title=f'N={self.N}_D={self.dist}_L={self.L}_nModes={self.N}_nXbar={len(Xbar)}_dt={self.dt}_dr={delta}_ct={ct}_EqPhase={EqPhase}_HalfPlane={self.halfPlane}'
        return Total_Propagation, Title
    
    def NCA_FPmeasure(self,FP,R0,Xbar,EqPhase=False):
        start=time.time()
        U=self.U
        F=self.F
        Ph=self.Ph
        d=self.dist
        NFPM={}
        for i in range(len(U)):
            u=U[i]
            f=F[i]
            ph=Ph[i]
            CADict=self.CAdictFun(R0,u,f,ph,EqPhase)
            FPM=self.FPmeasure(FP,CADict,Xbar)
            NFPM[i]=FPM
        Total_FPM={}
        for t in list(NFPM[0].keys()):
            FPMt=0
            for i in list(NFPM.keys()):
                FPMt=FPMt+NFPM[i][t]
            Total_FPM[t] = FPMt
        now=time.time()
        print(f'\nTotal time={now - start:0.5f}')
        return Total_FPM        

HoraFecha=f'{time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}-{time.localtime().tm_mday}-{time.localtime().tm_mon}-{time.localtime().tm_year}'
        
def makeGif(PT,Title):
    filenames=[]
    images=[]
    bar = progressbar.ProgressBar(maxval=len(list(PT.keys())),widgets=[progressbar.Bar('=', '[', ']'),' ',progressbar.Percentage()])
    bar.start()
    Dir=f'CAGifs/'
    if not os.path.exists(Dir):
        os.mkdir(Dir)

    for j,i in enumerate(list(PT.keys())):
        bar.update(j+1)
        plt.figure(figsize = (20,20))
        plt.imshow(PT[i],origin='lower')
        plt.title(f't={i:0.5f}')
        plt.axis('off')
        filename='CAGifs/modes.png'
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)
        images.append(imageio.imread(filename))
    imageio.mimsave('CAGifs/'+Title+'.gif', images)
    bar.finish()

        
def FFT(V,dt,min_val=0.05,dist=50):
        #V=np.array(v)
        fs=int(1/dt)
        n = len(V)
        fft_norm=2.0/n
        FFT=np.fft.fft(V)
        N = len(FFT)
        phase=np.angle(FFT)[:(N // 2)]
        FFT_pow = np.abs(fft_norm*FFT)
        FFT_pow=FFT_pow[:(N // 2)]
        #f_pos = np.arange(0, fs / 2, step=fs / n)
        f_pos=np.fft.fftfreq(len(V),dt)[:(N // 2)]
        #f_pos=np.fft.fftfreq(V.shape[-1],dt)[:(N // 2)]
        maxVal,indx=max([(j, i) for i, j in enumerate(FFT_pow)])
        #plt.plot(f_pos[peaks], FFT_pow[peaks], "x")
        plt.plot(f_pos[indx],maxVal,'r.')
        plt.plot(f_pos, FFT_pow)
        plt.xlabel('frecuencia en Hz')
        plt.ylabel('Amplitud~|FFT|')
        plt.show()
        return f_pos[indx],maxVal,phase[indx]

def FFTpeaks(V,dt,Num,min_val=0.0005,dist=5):
        fs=int(1/dt)
        n = len(V)
        fft_norm=2.0/n
        FFT=np.fft.fft(V)
        N = len(FFT)
        FFT_pow = np.abs(fft_norm*FFT)
        FFT_pow=FFT_pow[:(N // 2)]
        phase=np.angle(FFT)[:(N // 2)]
        #f_pos = np.arange(0, fs / 2, step=fs / n)
        f_pos=np.fft.fftfreq(n,dt)[:(N // 2)]
        peaks, prop = find_peaks(FFT_pow, height=min_val,distance=dist)
        plt.plot(f_pos[peaks], FFT_pow[peaks], "r.")
        plt.plot(f_pos, FFT_pow)
        f=f_pos[peaks]
        ph=phase[peaks]
        u=prop['peak_heights']
        l=len(u)
        U=sorted(u)[l-Num:l]
        F=[f[i] for i,_ in sorted(enumerate(u),key=lambda x: x[1])]
        F=F[l-Num:l]
        Ph=[ph[i] for i,_ in sorted(enumerate(u),key=lambda x: x[1])]
        Ph=Ph[l-Num:l]
        plt.plot(F,U,'d')
        plt.xlabel('frecuencia en Hz')
        plt.ylabel('Amplitud~|FFT|')
        plt.show()
        return F,U,Ph
    
def FFTpeaksPhase(V,dt,Num,Title='',min_val=0.0005,dist=5):
        fs=int(1/dt)
        n = len(V)
        fft_norm=2.0/n
        FFT=np.fft.fft(V)
        N = len(FFT)
        FFT_pow = np.abs(fft_norm*FFT)
        FFT_pow=FFT_pow[:(N // 2)]
        phase=np.angle(FFT)[:(N // 2)]
        #f_pos = np.arange(0, fs / 2, step=fs / n)
        f_pos=np.fft.fftfreq(n,dt)[:(N // 2)]
        peaks, prop = find_peaks(FFT_pow, height=min_val,distance=dist)
        plt.plot(f_pos[peaks], FFT_pow[peaks], "r.")
        plt.plot(f_pos, FFT_pow)
        f=f_pos[peaks]
        ph=phase[peaks]
        u=prop['peak_heights']
        l=len(u)
        U=sorted(u)[l-Num:l]
        F=[f[i] for i,_ in sorted(enumerate(u),key=lambda x: x[1])]
        F=F[l-Num:l]
        Ph=[ph[i] for i,_ in sorted(enumerate(u),key=lambda x: x[1])]
        Ph=Ph[l-Num:l]
        for i in range(len(U)):
            plt.plot(F[i],U[i],'d',label=f'fase={Ph[i]:0.4f}')
        plt.legend(loc='best')
        plt.title(Title)
        plt.xlabel('frecuencia en Hz')
        plt.ylabel('Amplitud~|FFT|')
        #plt.show()
        return F,U,Ph
    
def readCSVDir1D(Dir):
    PosArrayFile=Dir+'PosArray.csv'
    posData=pd.read_csv(PosArrayFile)
    Coords=posData.columns
    X=posData['Xpos'].values
    if len(Coords)==2:
        Y=posData['Ypos'].values
    if len(Coords)==2:
        Z=posData['Zpos'].values
        
def savePT(PT,Dir):
    os.mkdir(Dir)
    i=0
    for t in list(PT.keys()):
        with open(Dir+f'{i}.npy', 'wb') as f:
            np.save(f, PT[t])
        i+=1
    with open(Dir+'Time.csv','w') as g:
            g.write('t\n')
            for t in list(PT.keys()):
                g.write(f'{t}\n')
def loadPT(Dir):
    Time=pd.read_csv(Dir+'Time.csv')
    T=Time['t'].values
    T
    PTload={}
    for i in range(len(T)):
        t=T[i]
        with open(Dir+f'{i}.npy', 'rb') as f:
            a = np.load(f)
            PTload[t]=a
    return PTload
    
        
        