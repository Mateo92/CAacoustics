#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:52:03 2022

@author: mateo
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import imageio
import os
import shutil
from scipy.integrate import quad
import time



class circularBar():
    def __init__(self,L,r,Young,rho,nx=50,V=0.1):
        self.L=L
        self.r=r
        self.Young=Young
        self.rho=rho
        self.nx=nx
        self.V=V
        self.Xax,self.h=self.X_ax()
        
    def g_n(self,n):
        L=self.L
        g_nL= [ 1.87510407,  4.69409113,  7.85475744, 10.99554073, 14.13716839,17.27875953, 20.42035225, 23.5619449 , 26.70353756, 29.84513021,32.98672286, 36.12831552, 39.26990817, 42.41150082, 45.55309348,48.69468613, 51.83627878, 54.97787144, 58.11946409, 61.26105675,64.4026494 , 67.54424205, 70.68583471, 73.82742736, 76.96902001,80.11061267, 83.25220532, 86.39379797, 89.53539063, 92.67698328]
        gn=[gnL/L for gnL in g_nL]
        return gn[n]
    def g_nL(self,n):
        gnL= [ 1.87510407,  4.69409113,  7.85475744, 10.99554073, 14.13716839,17.27875953, 20.42035225, 23.5619449 , 26.70353756, 29.84513021,32.98672286, 36.12831552, 39.26990817, 42.41150082, 45.55309348,48.69468613, 51.83627878, 54.97787144, 58.11946409, 61.26105675,64.4026494 , 67.54424205, 70.68583471, 73.82742736, 76.96902001,80.11061267, 83.25220532, 86.39379797, 89.53539063, 92.67698328]
        return gnL[n]
    
    def Y_n(self,x,n):
        L=self.L
        A=np.cosh(self.g_n(n)*x)-np.cos(self.g_n(n)*x)
        B=(np.cos(self.g_n(n)*L)+np.cosh(self.g_n(n)*L))/(np.sin(self.g_n(n)*L)+np.sinh(self.g_n(n)*L))
        C=np.sin(self.g_n(n)*x)-np.sinh(self.g_n(n)*x)
        return A+B*C
    
    def X_ax(self):
        h=self.L/self.nx
        X= np.arange(0.0,self.L+h,h)
        return X,h
    
    def g_f(self):
        V=self.V
        X,h=self.Xax,self.h
        L=self.L
        g=np.zeros(len(X))
        for i in range(len(g)):
            if (i*h>=L/3):
                if (i*h<L/2):
                    g[i]=(X[i]-L/3)*V
            if (i*h>=L/2):
                if (i*h<2*L/3):
                    g[i]=(-X[i]+2*L/3)*V
        return g
    
    def omega_n(self,n):
        """
        esto se llama docstring
        kjhshjh sjh s sks skjs skj jks

        Parameters
        ----------
        n : int
            hsjhg sjhg jhs jsDESCRIPTION.

        Returns
        -------
        omegan : TYPE
            DESCRIPTION.

        """
        r=self.r;Young=self.Young;rho=self.rho;
        I=np.pi*r**4/4
        S=np.pi*r**2
        k=np.sqrt(Young*I/(rho*S))
        omegan=self.g_n(n)*self.g_n(n)*k
        return omegan
    
    def N_n(self,n):
        #TODO: scipy.integrate
        X,h=self.Xax,self.h
        N=0
        for x in X:
            N=N+self.Y_n(x,n)*self.Y_n(x,n)*h
        # def integrand(x,n):
        #     self.Y_n(x,n)*self.Y_n(x,n)
        # N=quad(integrand, 0,self.L, args=n)
        return N

    def B_n(self,n):
        X,h=self.Xax,self.h
        g=self.g_f()
        a=1/(self.N_n(n)*self.omega_n(n))
        inte=0
        for i in range(len(X)):
            inte=inte+g[i]*self.Y_n(X[i],n)*h
        B=a*inte
        return B
    def y_n(self,x,t,n):
        yn=(self.B_n(n)*np.sin(self.omega_n(n)*t))*self.Y_n(x,n)
        return yn
    
    def y(self,x,t,n_modes,m=0):
        y=0
        for n in range(m,n_modes):
            y=y+self.y_n(x,t,n)
        return y
    
    def y_arr(self,t,n_modes,m=0):
        X=self.Xax
        return np.array([self.y(x,t,n_modes,m) for x in X])
    
    def v_Mat(self,TF,N,n_modes,fs = 44100,writeCSV=False):
        dt=1/fs
        Time=np.arange(0, TF+dt, step=dt)
        T1=Time[0:len(Time)-1]
        T2=Time[1:]
        #T=(T1+T2)*0.5
        T=T2
        h=self.L/N
        X= np.arange(h,self.L+h,h)
        VelMAt=np.zeros((len(X),len(T1)))
        #vMAt[0]=self.g_f()
        Vel=np.zeros(len(X))
        if writeCSV:
            Dir=f'Bar_L={self.L}_r={self.r}_nx={self.nx}_n={n_modes}_T={TF}/'
            if os.path.exists(Dir):
                shutil.rmtree(Dir)
                os.makedirs(Dir)
            else:
                os.mkdir(Dir)
        for i in range(len(X)):
            x=X[i]
            y=self.y(x,Time,n_modes)
            posNow=y[0:len(Time)-1]
            posNext=y[1:]
            Vel=(posNext-posNow)*fs
            VelMAt[i]=Vel
            if writeCSV:
                with open(Dir+f"{i}x.csv", "w") as f:
                    f.write("t,v\n")
                    for j in range(len(Vel)):   
                        f.write(f"{T[j]},{Vel[j]}\n")
        if writeCSV:
            with open(Dir+"PosArray.csv", "w") as f:
                f.write("Xpos\n")
                for x in X :
                    f.write(f'{x}\n')
        return VelMAt,T,X,h,dt
    
    def f_n(self,n):
        fn=self.omega_n(n-1)/(2*np.pi)
        return fn

    def midi_n(self,n):
        midin=round(69+12*np.log2(self.f_n(n)/440),4) 
        return midin
    
    def modesGifs(self,min_mode=2,max_mode=10,delta_t=0.01,Time_steps=20,sep_im=False,rm_im=False):
        X=self.Xax
        L=self.L
        r=self.r
        for mode in range(min_mode,max_mode):
            filenames = []
            images= []
            def nodes(x):
                return self.y(x,np.pi/(2*self.omega_n(mode-1)),mode,mode-1)
            Nodes=optimize.newton(nodes, [m*L/20 for m in range (2,20)])
            mL=[n<L for n in Nodes]
            Nodes=Nodes[mL]
            non_zero=[n>1e-5 for n in Nodes]
            Nodes=Nodes[non_zero]
            zero_val=[abs(nodes(x))<abs(nodes(L))/2 for x in Nodes]
            Nodes=Nodes[zero_val]
            Nodes=[round(n,4) for n in Nodes]
            Nodes=np.unique(Nodes)
            Nodes=np.sort(Nodes)
            for t in range(Time_steps):
                Time=t*delta_t
    
                y_x_t=[self.y(x,Time,mode,mode-1) for x in X]
    
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
    
                ax.plot(X,y_x_t)
                for i in range(len(Nodes)):
                    ax.scatter(Nodes[i],0,label='Node'+str(int(i+1))+':'+str(round(Nodes[i],4))+'m')
                ax.legend(loc='upper right', frameon=False)
    
                ax.set_title('Mode:'+str(mode)+', L='+str(L)+'m'+', r='+str(r)+'m'+', t='+str(round(Time,4))+'s')
    
                # Major ticks every L/5, minor ticks every L/50
                major_ticks = np.arange(0, L+L/5, L/5)
                minor_ticks = np.arange(0, L+L/50, L/50)
                ax.set_xlabel('||='+str(L/50)+'m'+', freq='+str(round(self.f_n(mode),3))+'Hz'+', MIDI#='+str(self.midi_n(mode)))
    
                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)
                ax.set_ylim(-abs(self.y(L,np.pi/(2*self.omega_n(mode-1)),mode,mode-1))*1.2,abs(self.y(L,np.pi/(2*self.omega_n(mode-1)),mode,mode-1))*1.2)
    
                # And a corresponding grid
                #ax.grid(which='both')
    
                # Or if you want different settings for the grids:
                ax.grid(which='minor', alpha=0.4)
                ax.grid(which='major', alpha=0.8)
    
                    # create file name and append it to a list
                if sep_im:
                    filename = f'BarGifs/{t}.png'
                else:
                    filename='BarGifs/modes.png'
                filenames.append(filename)
                # save frame
                plt.savefig(filename)
                plt.close()
                images.append(imageio.imread(filename))
            # Remove files
            if sep_im:
                if rm_im:
                    for filename in set(filenames):
                        os.remove(filename)
            # build gif
            imageio.mimsave(f'BarGifs/{mode}mode_L={L}_r={r}.gif', images)
    
        
        return 'Modes gifs Ready (Check the folder)'
    
    def barGif(self,modes=10,delta_t=0.001,Time_steps=50,sep_im=False,rm_im=False):
        X=self.Xax
        L=self.L
        r=self.r
        print(f'Total steps={Time_steps}')
        filenames = []
        images= []
        for t in range(Time_steps):
            start=time.time()
            Time=t*delta_t
            y_x_t=[self.y(x,Time,modes,0) for x in X]
    
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
    
            ax.plot(X,y_x_t)
            
            ax.set_title(f'# of Modes:{modes},L={L}m,r={r}m,t={round(Time,4)}s')
    
            # Major ticks every L/5, minor ticks every L/50
            major_ticks = np.arange(0, L+L/5, L/5)
            minor_ticks = np.arange(0, L+L/50, L/50)
            ax.set_xlabel('||='+str(L/50))
    
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_ylim(-abs(self.y(L,np.pi/(2*self.omega_n(0)),1,0))*1.2,abs(self.y(L,np.pi/(2*self.omega_n(0)),1,0))*1.2)
        
            # And a corresponding grid
            #ax.grid(which='both')
    
            # Or if you want different settings for the grids:
            ax.grid(which='minor', alpha=0.4)
            ax.grid(which='major', alpha=0.8)
    
                # create file name and append it to a list
            if sep_im:
                filename = f'BarGifs/{t}.png'
            else:
                filename='BarGifs/bar.png'
            filenames.append(filename)
            # save frame
            plt.savefig(filename)
            plt.close()
            images.append(imageio.imread(filename))
            now=time.time()
            print(f'{t}:{now-start}',end=', ')
       
        # # Remove files
        if sep_im:
            if rm_im:
                for filename in set(filenames):
                    os.remove(filename)
        # build gif
        imageio.mimsave(f'BarGifs/{modes}modes_bar_L={L}r={r}_dt={delta_t}_TF={Time_steps*delta_t}.gif', images)
    
        
        return str(modes)+'Modes Bar gif ready (Check the folder)'
    

def g_n(n,L):
    g_nL= [ 1.87510407,  4.69409113,  7.85475744, 10.99554073, 14.13716839,17.27875953, 20.42035225, 23.5619449 , 26.70353756, 29.84513021,32.98672286, 36.12831552, 39.26990817, 42.41150082, 45.55309348,48.69468613, 51.83627878, 54.97787144, 58.11946409, 61.26105675,64.4026494 , 67.54424205, 70.68583471, 73.82742736, 76.96902001,80.11061267, 83.25220532, 86.39379797, 89.53539063, 92.67698328]
    gn=[gnL/L for gnL in g_nL]
    return gn[n]
def g_nL(n):
    gnL= [ 1.87510407,  4.69409113,  7.85475744, 10.99554073, 14.13716839,17.27875953, 20.42035225, 23.5619449 , 26.70353756, 29.84513021,32.98672286, 36.12831552, 39.26990817, 42.41150082, 45.55309348,48.69468613, 51.83627878, 54.97787144, 58.11946409, 61.26105675,64.4026494 , 67.54424205, 70.68583471, 73.82742736, 76.96902001,80.11061267, 83.25220532, 86.39379797, 89.53539063, 92.67698328]
    return gnL[n]

def omega_n(n,L,r,Young,rho):
    I=np.pi*r**4/4
    S=np.pi*r**2
    k=np.sqrt(Young*I/(rho*S))
    omegan=g_n(n,L)*g_n(n,L)*k
    return omegan

def f_n(n,L,r,Young,rho):
    fn=omega_n(n-1,L,r,Young,rho)/(2*np.pi)
    return fn

def midi_n(n,L,r,Young,rho):
    midin=round(69+12*np.log2(f_n(n,L,r,Young,rho)/440),4) 
    return midin

def L_f_r(n,f,r,Young,rho):
    g=g_nL(n-1)*2/np.pi
    return np.sqrt(g*g*np.sqrt(Young/rho)*np.pi*r/(16*f))

def L_m_r(n,midi,r,Young,rho):
    g=g_nL(n-1)*2/np.pi
    f=2**((midi-69)/12)*440
    return np.sqrt(g*g*np.sqrt(Young/rho)*np.pi*r/(16*f))

Y_zinc=10e10
rho_zinc=7138
Y_aluminio=7e10
rho_aluminio=2698.4