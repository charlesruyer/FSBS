# -----------------------------------------------------------------------
# Brillouin - C. Ruyer - 12/2019
# -----------------------------------------------------------------------
#
# >>>>>> Analyse de l'XP omega EP de traverse de cavite
#
# >>>>>> Requirements
#   python2.7 with the following packages: numpy, matplotlib, pylab, scipy

# >>>>>> Advice: install ipython (historics of commands, better interface)
# >>>>>> First step: invoke python and load this file
#      $ ipython -i Luke_xp.py
#
# >>>>>> Second step: in the ipython shell, use the functions

from scipy.linalg import expm, inv

import scipy.special
from scipy import signal
from scipy.special import exp1
from scipy.special import  erf
from scipy.special import  fresnel
import numpy as np
import numpy.matlib as ma
from scipy.interpolate import interp1d
import os.path, glob, re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from scipy.integrate import odeint
from scipy.integrate import complex_ode
from scipy.integrate import ode
from scipy.special import sici
from scipy.special import  lambertw
from scipy.special import  erf
from scipy.special import  erfi
from scipy.special import  sici
from scipy import integrate
from scipy import optimize 
import pylab
import sys
#pylab.ion()

#Fonction plasma et ses derive a variable REELLE !
def Z_scalar(x):
    if np.abs(x)<=25:
        res = np.pi**0.5*np.exp(-x**2)*( 1j - erfi(x) ) 
    else:
        res=-1/x
    return res
def Z(x):
    if np.size(x)==1:
        return Z_scalar(x)
    else:
        res = 0*x+0j*x
        for i in range(len(x)):
            ##print 'i= ',i, x[i], Z_scalar(x[i])
            res[i] = Z_scalar(x[i]) 
        return res 

def Zp(x):
    return -2*(1 + x*Z(x)) 

def Zp_scalar(x):
    return -2*(1 + x*Z_scalar(x))

def Zpp(x):
    return  -2*(Z(x) + x*Zp(x))

def Zppp(x):
    return  -2*( 2*Zp(x) + x*Zpp(x) )

def plot_alpha_kin(figure=1):
    c=3e8
    Te=1000.
    #Ti=1000.
    k0=2*np.pi / 0.35e-6 
    #cs = 0.5*np.sqrt((Te+3*Ti)/511000./1836.)*c
    #vd=0.8*cs
    w0=k0*c*np.sqrt(1.1)
    ks = np.linspace(-0.02*k0,0.02*k0,1000)
    vp = (-0.5*np.abs(ks) *c**2/w0 -0*ks/np.abs(ks))
    #ak1 = alpha_kin(xie,xii,1)
    ak1 = alpha_kin(Te=Te, Ti=[Te/1.], Z=[1.], A=[1.], nisne=[1.], vphi=vp,figure=None) 
    ak3 = alpha_kin(Te=Te, Ti=[Te/3.], Z=[1.], A=[1.], nisne=[1.], vphi=vp,figure=None) 
    ak5 = alpha_kin(Te=Te, Ti=[Te/5.], Z=[1.], A=[1.], nisne=[1.], vphi=vp,figure=None) 

    dak1=np.diff(ak1)/(ks[1]-ks[0])
    dak3=np.diff(ak3)/(ks[1]-ks[0])
    dak5=np.diff(ak5)/(ks[1]-ks[0])
    ksd = ks[1:len(ks)]
    print np.shape(dak1), np.shape(ksd)

    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure,figsize=[7,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.19, right=0.92, top=0.9, bottom=0.16)
        l1,=plt.plot(ks/k0, np.real(ak1), 'k',linewidth=2 )
        l3,=plt.plot(ks/k0, np.real(ak3) , 'r',linewidth=2)
        l5,=plt.plot(ks/k0, np.real(ak5), 'b',linewidth=2 )
        prop = fm.FontProperties(size=18)
        plt.legend([l1,l3,l5],['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=1, prop=prop,bbox_to_anchor=(1.1, 0.5))
        ax.set_xlabel("$k_s/k_0$")
        ax.set_ylabel("$\\Re(\\alpha_\\mathrm{kin})$")
        ax.set_xlim(-0.015,0.015)
        ax.set_xticks([-0.02,-0.01,0,0.01,0.02]) 
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1,figsize=[7,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.19, right=0.92, top=0.9, bottom=0.16)
        l1,=plt.plot(ksd/k0, np.real(dak1), 'k',linewidth=2 )
        l3,=plt.plot(ksd/k0, np.real(dak3) , 'r',linewidth=2)
        l5,=plt.plot(ksd/k0, np.real(dak5), 'b',linewidth=2 )
        prop = fm.FontProperties(size=18)
        plt.legend([l1,l3,l5],['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=1, prop=prop,bbox_to_anchor=(1.1, 0.5))
        ax.set_xlabel("$k_s/k_0$")
        ax.set_ylabel("$\\Re(\\alpha_\\mathrm{kin})$")
        ax.set_xlim(-0.015,0.015)
        ax.set_xticks([-0.02,-0.01,0,0.01,0.02]) 
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        plt.show()

# Fonction de transfert cinetique de la force ponderomotrice
# Zp(xie)/2 * ( 1 - sum_i Zp(xi) ) /eps
def alpha_kin(Te, Ti, Z, A, nisne, vphi, k2lde2=0, ne=1, figure=None, is_chiperp = False):
    c=3e8
    if ne ==0 : # On neglige les e-
        Zpxie = -2+0j
    else: 
        xie = vphi /np.sqrt(2*Te/511000. )/c
        Zpxie = Zp(xie)
    ##print 'Zp(xie) = ', Zpxie
    Xe= Zpxie * 1.
    sumXi = 0. + 0j
    for i in range(len(Ti)):
        xi = vphi /np.sqrt(2*Ti[i]/511000./1836/A[i] )/c
        sumXi += Zp(xi) * nisne[i] *Te* Z[i]**2/Ti[i]
        
        ##print 'Zp(xii) = ', Zp(xi)
    # Si k2lde2 ==0  on suppose k2lde2 <<  1 (Longeur de Debaye electronique
    ak = -0.5*Zpxie *(k2lde2- sumXi) / (k2lde2 - Xe-sumXi)
    ##print np.shape(np.real(ak)), np.shape(np.imag(ak))
    if figure is None:
        return  ak
    else:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        a, = ax.plot(vphi/c,np.real(ak),'k', linewidth=2)              
        a, = ax.plot(vphi/c,np.imag(ak),'--k', linewidth=2)              
        ax.set_ylabel("$F_\mathrm{kin}$")
        ax.set_xlabel("$v_\phi/c$ ")
        #if ylog:
        #    ax.set_yscale("log")
        #if xlog:
        #    ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()

def plot_gkx(x,k,sigma,Gamma02d,Te=1.e3, Ti=[1.e3/3.], Z=[1.], A=[1.], nisne=[1.], k2lde2=0, ne=1,f=6.5,k0=2*np.pi/0.35e-6,Nx=20000,klim=None,clog=False,figure=1):
    c=3e8
    km=k0/(2*f)
    ksmsk0=1.5e-2
    kint=np.linspace(-ksmsk0,+ksmsk0,50000)*k0
    kintp=np.linspace(0,+ksmsk0,50000)*k0
    vphi= -0.5*(np.abs(kint)/ k0)*c
    vphip= -0.5*(np.abs(kintp)/ k0)*c
    ak =np.real(alpha_kin(Te, Ti, Z, A, nisne, (vphi), k2lde2, ne))
    akp=np.real(alpha_kin(Te, Ti, Z, A, nisne, (vphip), k2lde2, ne))

    xint=np.linspace(np.min(x),np.max(x),Nx)
    g02d,g03d=0*xint+0j*xint,0*xint+0j*xint
    G02d,G03d=0*xint+0j*xint,0*xint+0j*xint
    for i in range(len(xint)):
        print 'i = ',i,' sur ',len(xint)
        g02d[i] =   np.trapz(       1j*np.exp(1j*kint**2 /(2*k0)*xint[i]) *ak  , kint )  /(2*np.pi)
        g03d[i] =   np.trapz( kintp*1j*np.exp(1j*kintp**2/(2*k0)*xint[i]) *akp, kintp )  /(2*np.pi)**1
        G02d[i] =    Gamma02d*np.trapz(g02d[range(i+1)], xint[range(i+1)])
        G03d[i] =    Gamma02d*sigma*np.trapz(g03d[range(i+1)], xint[range(i+1)])

    Nxnew=2000
    mesh=range(0,len(xint),Nx/Nxnew)
    x=xint[mesh]
    g02d = g02d[mesh]
    G02d = G02d[mesh]
    g03d = g03d[mesh]
    G03d = G03d[mesh]
    g2d=np.zeros((len(k),len(x)))+0j*np.zeros((len(k),len(x)))
    g3d=np.zeros((len(k),len(x)))+0j*np.zeros((len(k),len(x)))
    G2d=0*k
    G3d=0*k
    ix0 =np.argmin(np.abs(x))
    for ik in range(len(k)):
        print 'ik = ',ik,' sur ',len(k)
        for ix in range(len(x)):
            #g2d[ik,ix] = np.trapz(ak *(1j*      np.exp(1j*(kint**2 -k[ik]**2)/(2*k0)*x[ix]+(G02d[ix]))),kint )/(2*np.pi)
            #g3d[ik,ix] = np.trapz(akp*(1j*kintp*np.exp(1j*(kintp**2-k[ik]**2)/(2*k0)*x[ix]+(G03d[ix]))),kintp)/(2*np.pi)**1
            g2d[ik,ix] = g02d[ix]* np.exp(-1j*k[ik]**2/(2*k0)*x[ix]+(G02d[ix]))
            g3d[ik,ix] = g03d[ix]* np.exp(-1j*k[ik]**2/(2*k0)*x[ix]+(G03d[ix]))
        G2d[ik] =  np.trapz(g2d[ik,range(ix0+1)], x[range(ix0+1)])
        G3d[ik] =  np.trapz(g3d[ik,range(ix0+1)], x[range(ix0+1)])

    tg2d=np.arctan(np.real(g02d),np.imag(g02d))
    n2d = int(np.floor((np.imag(G02d[0])+tg2d[0])/np.pi/2)+1 )
    if np.mod(n2d,2)==0:
        n2d+=1
    n2dmax = int(np.floor(np.abs(np.imag(G02d[0])+tg2d[0] +2*km**2*np.abs(x[0])/k0)))
    print 'D=1 : Nmin, Nmax = ', n2d,n2dmax
    #while n2d*np.pi - Gamma02d*np.imag(G02d[0])-tg2d[0]<0:
    #    n2d+=1.
    tg3d=np.arctan(np.real(g03d),np.imag(g03d))
    n3d = int(np.floor((np.imag(G03d[0])+tg3d[0])/np.pi/2)+1 )
    if np.mod(n3d,2)==0:
        n3d+=1
    n3dmax = int(np.floor(np.abs(np.imag(G03d[0])+tg3d[0] +2*km**2*np.abs(x[0])/k0)))
    print 'D=2 : Nmin, Nmax = ', n3d,n3dmax
    kc2d=np.sqrt( (n2d*np.pi -Gamma02d*np.imag(G02d)-tg2d )*2*k0/np.abs(x) )
    kc3d=np.sqrt(  (n3d*np.pi +Gamma02d*sigma*np.imag(G03d)+tg3d )*2*k0/np.abs(x) )

    Dk22d = 4*k0/np.abs(x)*(np.pi/2. +km**2*np.abs(x)/k0 -0.5*Gamma02d*np.imag(G02d) -0.5*tg2d)
    Dk22d = np.sqrt( Dk22d*(Dk22d>0) )
    Dk23d = 4*k0/np.abs(x)*(np.pi/2. +km**2*np.abs(x)/k0 +0.5*Gamma02d*sigma*np.imag(G03d) +0.5*tg3d)
    Dk23d = np.sqrt( Dk23d*(Dk23d>0) )

    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)    
        if clog:
            data = np.log(np.real(g2d)*(np.real(g2d)>0))
            cf0  = plt.pcolor(km**2*x/k0, k/km , data,vmax=np.max(data),vmin=np.max(data)-2)#,cmap=plt.cm.RdBu) 
        else:
            cf0  = plt.pcolor(km**2*x/k0, k/km , np.real(g2d),cmap=plt.cm.RdBu)

        plt.colorbar(cf0)
        for n in range(n2d,n2dmax+1,2):
            kc2d=np.sqrt( (n*np.pi -np.imag(G02d)-tg2d )*2*k0/(-x) )
            plt.plot(km**2*x/k0, kc2d/km,'k',linewidth=2)
        ax.set_ylabel("$k/k_m$")
        ax.set_xlabel("$k_m^2 x /k_0$ ")
        ax.set_xlim(np.min(km**2*x/k0),np.max(km**2*x/k0))
        ax.set_ylim(np.min( k/km ),np.max( k/km ))
        if klim is not None:
            ax.set_ylim(klim[0],klim[1])
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)       
        if clog:
            data = np.log(np.real(g3d)*(np.real(g3d)>0))
            cf0  = plt.pcolor(km**2*x/k0, k/km , data,vmax=np.max(data),vmin=np.max(data)-2)
        else:
            cf0=plt.pcolor(km**2*x/k0, k/km , np.real(g3d),cmap=plt.cm.RdBu) 
        plt.colorbar(cf0)
        for n in range(n3d,n3dmax+1,2):
            kc3d=np.sqrt( ((n)*np.pi -np.imag(G03d)-tg3d )*2*k0/(-x) )
            plt.plot(km**2*x/k0, kc3d/km,'k',linewidth=2)
        ax.set_ylabel("$k/k_m$")
        ax.set_xlabel("$k_m^2 x /k_0$ ")
        ax.set_xlim(np.min(km**2*x/k0),np.max(km**2*x/k0))
        ax.set_ylim(np.min( k/km ),np.max( k/km ))
        if klim is not None:
            ax.set_ylim(klim[0],klim[1])
        fig.canvas.draw()
        plt.show()
        
        fig = plt.figure(figure+2, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        ax2 = ax.twinx()
        plt.subplots_adjust(left=0.15, right=0.86, top=0.9, bottom=0.2)        
        r1,=ax.plot(km**2*x/(k0),np.real(G02d),'k',linewidth=2)
        i1,=ax.plot(km**2*x/(k0), np.imag(G02d),'--k',linewidth=2)  
        r2,=ax2.plot(km**2*x/(k0),np.real(G03d),'r',linewidth=2)
        i2,=ax2.plot(km**2*x/(k0), np.imag(G03d),'--r',linewidth=2)
        prop = fm.FontProperties(size=18)
        plt.legend([r1,i1],['$G_r$','$G_i$'],loc='best', prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$G(x)$, $D=1$")   
        ax2.set_ylabel("$G(x)$, $D=2$",color='red')
        ax.set_xlabel("$k_m^2 x /k_0$ ")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.set_xlim(np.min(km**2*x/k0),np.max(km**2*x/k0))
        #ax.set_ylim(np.min( k/km ),np.max( k/km ))
        #if klim is not None:
        #    ax.set_ylim(klim[0],klim[1])
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+3, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        ax2 = ax.twinx()
        plt.subplots_adjust(left=0.15, right=0.82, top=0.9, bottom=0.2)       
        d1,=ax2.plot(k/km , G3d*Gamma02d*sigma , '--r' , linewidth=1)           
        d1,=ax.plot(k/km , G2d*Gamma02d , 'k' , linewidth=2)             
        ax.set_xlabel("$k/k_m$")
        ax.set_ylabel("$D=1$,  $I_d/I_{d,0}-1 $")
        ax2.set_ylabel("$D=2$,  $I_d/I_{d,0}-1 $",color='red')
        ax.set_xlim(np.min( k/km ),np.max( k/km ))
        ax.set_ylim(0,1.1*np.max( G2d*Gamma02d  ))
        ax2.set_ylim(0,1.1*np.max( G3d*Gamma02d*sigma  ))
        fig.canvas.draw()
        plt.show()

        fig = plt.figure(figure+4, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)        
        plt.plot(km**2*x/(k0), kc2d/km,'k',linewidth=2)
        plt.plot(km**2*x/(k0), kc3d/km,'--k',linewidth=2)
        ax.set_ylabel("$k/k_m$")
        ax.set_xlabel("$k_m^2 x /k_0$ ")
        ax.set_xlim(np.min(km**2*x/k0),np.max(km**2*x/k0))
        #ax.set_ylim(np.min( k/km ),np.max( k/km ))
        if klim is not None:
            ax.set_ylim(klim[0],klim[1])
        fig.canvas.draw()
        plt.show()
        
        fig = plt.figure(figure+5, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)        
        plt.plot(km**2*x/(k0), Dk22d/km,'k',linewidth=2)
        plt.plot(km**2*x/(k0), Dk23d/km,'--k',linewidth=2)
        ax.set_ylabel("$\\Delta k/k_m$")
        ax.set_xlabel("$k_m^2 x /k_0$ ")
        ax.set_xlim(np.min(km**2*x/k0),np.max(km**2*x/k0))
        #ax.set_ylim(np.min( k/km ),np.max( k/km ))
        #if klim is not None:
        #    ax.set_ylim(klim[0],klim[1])
        fig.canvas.draw()
        plt.show()

def int_alpha_kin_final(figure=1):
    x=np.linspace(-6e-3,6e-3,1000)
    sigma=100e-6
    ksmsk0=1.5e-2
    Te=1.e3
    Ti=[300.]
    Z=[1.]
    A=[1.]
    nisne=[1.]
    k2lde2=0
    ne=1
    f=6.5
    k0=2*np.pi/0.35e-6
    xlog=False
    ylog=False
       
    c=3e8
    km=k0/(2*f)
    kint=np.linspace(-ksmsk0,+ksmsk0,50000)*k0
    #kint=np.linspace(-2*km,+2*km,50000)
    F = 0*x + 0j*x
    vphi= -0.5*(np.abs(kint)/ k0)*c
    ak=np.real(alpha_kin(Te, Ti, Z, A, nisne, (vphi), k2lde2, ne))
    for i in range(len(x)):
        F[i] = np.trapz( np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint ) 
    #print cF 
  
    ak0=np.trapz(ak , kint )
    ak2=np.trapz(ak *kint**2 , kint )
    ak4=np.trapz(ak  *kint**4, kint )
    Fa = ak0+1j*x*ak2/ (2*k0) -x**2*ak4 /(8*k0**2)

    if figure is not None:
    
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)     
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] = np.trapz( 1j*np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma

            ak, = ax.plot(km**2*x/k0,np.real(F),col[c], linewidth=2)           
            aki, = ax.plot(km**2*x/k0,np.imag(F),'--'+col[c], linewidth=2)   
            leg=leg+[ak]
            c+=1    
        r, =plt.plot([],[],'k',   linewidth=2)
        i,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        first_legend = ax.legend([r ,i],['$\\Re(g)$','$\\Im(g)$'],loc=8,prop=prop)#,numpoints=1)
        plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=4, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g(0,x)/\\Gamma_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        ax.set_xlim(-600,600)
        ax.set_ylim(-9e3,9e3)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)    
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        vphi = vphi[kint>=0]
        kint = kint[kint>=0]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] =   np.trapz( 2*np.pi*kint*1j*np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint )  /(2*np.pi)**2#*sigma**2
            ak,   = ax.plot(km**2*x/k0,np.real(F),col[c], linewidth=2)       
            aki, = ax.plot(km**2*x/k0,np.imag(F),'--'+col[c], linewidth=2)
            leg=leg+[ak]
            c+=1
        r, =plt.plot([],[],'k',   linewidth=2)
        i,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        first_legend = ax.legend([r ,i],['$\\Re(g)$','$\\Im(g)$'],loc=3,prop=prop)#,numpoints=1)
        plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=4, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g(0,x)/\\Gamma_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        ax.set_xlim(-600,600)
        ax.set_ylim(-5e8,3e8)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()







        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+2, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.86, top=0.9, bottom=0.2)     
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] = np.trapz( 1j*np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma

            ak, = ax.plot(km**2*x/k0,np.abs(F),col[c], linewidth=2)           
            ax2 = ax.twinx()
            aki, = ax2.plot(km**2*x/k0,np.arctan2(np.real(F),np.imag(F)),'--'+col[c], linewidth=2)   
            leg=leg+[ak]
            c+=1    
        r, =plt.plot([],[],'k',   linewidth=2)
        i,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        #first_legend = ax.legend([r ,i],['$\\vert g\\vert $','$\\theta_g$'],loc=8,prop=prop)#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=4, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$\\vert g(0,x) \\vert/\\Gamma_0$")
        ax2.set_ylabel("$\\theta_g$")
        ax.set_xlabel("$k_m^2x/k_0$")
        ax.set_xlim(-600,600)
        ax.set_ylim(0,5e3)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+3, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.86, top=0.9, bottom=0.2)    
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        vphi = vphi[kint>=0]
        kint = kint[kint>=0]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] =   np.trapz( 2*np.pi*kint*1j*np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint )  /(2*np.pi)**2#*sigma**2
            ak,   = ax.plot(km**2*x/k0,np.abs(F),col[c], linewidth=2)       
            
            ax2 = ax.twinx()
            aki, = ax2.plot(km**2*x/k0,np.arctan2(np.real(F),np.imag(F)),'--'+col[c], linewidth=2)   
            leg=leg+[ak]
            c+=1    
        r, =plt.plot([],[],'k',   linewidth=2)
        i,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        #first_legend = ax.legend([r ,i],['$\\vert g\\vert $','$\\theta_g$'],loc=8,prop=prop)#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=2, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$\\vert g(0,x) \\vert/\\Gamma_0$")
        ax2.set_ylabel("$\\theta_g$")
        ax.set_xlim(-600,600)
        ax.set_ylim(1.5e8,4.5e8)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

def int_alpha_kin(x,sigma,ksmsk0=1.5e-2,Te=1.e3, Ti=[300.], Z=[1.], A=[1.], nisne=[1.], k2lde2=0, ne=1,f=6.5,k0=2*np.pi/0.35e-6,xlog=False, ylog=False, figure=1):    
    c=3e8
    km=k0/(2*f)
    kint=np.linspace(-ksmsk0,+ksmsk0,50000)*k0
    #kint=np.linspace(-2*km,+2*km,50000)
    F = 0*x + 0j*x    
    vphi= -0.5*(np.abs(kint)/ k0)*c
    ak=np.real(alpha_kin(Te, Ti, Z, A, nisne, (vphi), k2lde2, ne))
    for i in range(len(x)):
        F[i] = np.trapz( np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint ) 
    #print cF 
  
    ak0=np.trapz(ak , kint )
    ak2=np.trapz(ak *kint**2 , kint )
    ak4=np.trapz(ak  *kint**4, kint )
    Fa = ak0+1j*x*ak2/ (2*k0) -x**2*ak4 /(8*k0**2)

    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        ak, = ax.plot(x*1e3,np.real(F),'k', linewidth=2)              
        ak, = ax.plot(x*1e3,np.imag(F),'--k', linewidth=2)              
        ak, = ax.plot(x*1e3,np.real(Fa),'r', linewidth=2)              
        ak, = ax.plot(x*1e3,np.imag(Fa),'--r', linewidth=2)              
        ax.set_ylabel("$g(k)$")
        ax.set_xlabel("$x$ mm ")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()
    
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)     
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] = np.trapz( 1j*np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma
            Finf = ak[kint[np.argmin(np.abs(kint ))]] * np.trapz( np.exp(1j*kint**2/(2*k0)*x[i]) , kint )  /(2*np.pi)/(4*km)
            print 'For D=1, Ti =',Ti, 'Finf = ',Finf
            ak2=np.trapz(ak *kint**2 , kint )  +1
            Fa = 1j*x*ak2/ (2*k0)/(2*np.pi)#/(4*km)
            ak, = ax.plot(km**2*x/k0,np.real(F),col[c], linewidth=2)           
            akth, = ax.plot(km**2*x/k0,-np.imag(Fa),'--'+col[c], linewidth=2)   
            leg=leg+[ak]
            c+=1    
        e, =plt.plot([],[],'k',   linewidth=2)
        th,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        first_legend = ax.legend([e ,th],['Exact','Linearization'],loc=4,prop=prop)#,numpoints=1)
        plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=2, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g(0,x)/\\Gamma_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-0.6/sigma,0.6/sigma)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+2, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)     
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] = np.trapz( 1j*np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma
            Finf = ak[kint[np.argmin(np.abs(kint ))]] * np.trapz( np.exp(1j*kint**2/(2*k0)*x[i]) , kint )  /(2*np.pi)/(4*km)
            print 'For D=1, Ti =',Ti, 'Finf = ',Finf
            ak2=np.trapz(ak *kint**2 , kint )  +1
            Fa = 1j*x*ak2/ (2*k0)/(2*np.pi)#/(4*km)
            ak, = ax.plot(km**2*x/k0,np.imag(F),col[c], linewidth=2)           
            leg=leg+[ak]
            c+=1    
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=2, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g(0,x)/\\Gamma_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-0.6/sigma,0.6/sigma
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()


        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+3, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)    
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        vphi = vphi[kint>=0]
        kint = kint[kint>=0]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] =  np.trapz( 2*np.pi*kint*1j*np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint )  /(2*np.pi)**2#*sigma**2
                
            Finf = ak[kint[np.argmin(np.abs(kint ))]] *np.trapz( 2*np.pi*kint*np.exp(1j*kint**2/(2*k0)*x[i]) , kint )  /(2*np.pi)**2/(4*km)**2
            print 'For D=2, Ti =',Ti, 'Finf = ',Finf
            ak2=np.trapz(2*np.pi*kint*ak *kint**2  , kint )/(2*np.pi)**2#/(4*km)**2
            Fa = 1j*x*ak2/ (2*k0)
            ak,   = ax.plot(km**2*x/k0,np.real(F),col[c], linewidth=2)       
            akth, = ax.plot(km**2*x/k0,-np.imag(Fa),'--'+col[c], linewidth=2)
            leg=leg+[ak]
            c+=1
        e, =plt.plot([],[],'k',   linewidth=2)
        th,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        first_legend = ax.legend([e ,th],['Exact','Linearization'],loc=4,prop=prop)#,numpoints=1)
        plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=2, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g(0,x)/\\Gamma_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-6/sigma**2,6/sigma**2)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+4, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)    
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        vphi = vphi[kint>=0]
        kint = kint[kint>=0]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] =  np.trapz( 2*np.pi*kint*1j*np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint )  /(2*np.pi)**2#*sigma**2
            ak,   = ax.plot(km**2*x/k0,np.imag(F),col[c], linewidth=2)       
            leg=leg+[ak]
            c+=1
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=2, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g(0,x)/\\Gamma_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-6/sigma**2,6/sigma**2)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

def ouverture_angulaire(x,f=6.5,k0=2*np.pi/0.35e-6,Z=1.,A=1.,Te=1.e3,Ti=1.e3/300.,figure=1):
    km=k0/(2*f)
    s, c= fresnel( np.sqrt(x/(np.pi*k0)) *2*km )
    Dk2dth =( ((  k0/x*c -2*km/np.pi*np.sqrt(np.pi*k0/x)* np.cos(2*km**2*x/k0)  ) / s ) )
    Dk3dth =( ( k0/x*( np.sin(2*km**2*x/k0) - 2*km**2*x/k0* np.cos(2*km**2*x/k0) ) / ( np.sin(km**2*x/k0)**2) ))
    
    c=3e8
    kint=np.linspace(-1.5e-2,1.5e-2,50000)*k0
    kintp=np.linspace(0,1.5e-2,50000)*k0
    vphi= -0.5*(np.abs(kint)/ k0)*c
    vphip= -0.5*(np.abs(kintp)/ k0)*c
    ak=np.real(alpha_kin(1e3, [1.e3/3.], [1.], [1.], [1.], (vphi)))
    Dk2dth2=0*x
    Dk3dth2=0*x
    for i in range(len(x)):
        Dk2dth2[i] = ( -2*k0/np.abs(x[i])*np.trapz(ak*np.sin(kint**2*np.abs(x[i])/(2*k0) ),kint) / np.trapz(ak*np.cos(kint**2*np.abs(x[i])/(2*k0) ),kint) )
        Dk3dth2[i] = ( -2*k0/np.abs(x[i])*np.trapz(kintp*ak*np.sin(kintp**2*np.abs(x[i])/(2*k0) ),kintp) / np.trapz(kintp*ak*np.cos(kint**2*np.abs(x[i])/(2*k0) ),kintp) )

    print np.real(Dk2dth2)/k0 /(2*km/k0)
    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+2, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)    
        th2,   = ax.plot(km**2*x/k0,np.real(Dk2dth)/k0**2 /(2*km/k0)**2,'k', linewidth=2)   
        th2a,   = ax.plot(km**2*x/k0,np.real(Dk2dth2)/k0**2 /(2*km/k0)**2,'--k', linewidth=2)       
        th3, = ax.plot(km**2*x/k0,np.real(Dk3dth)/k0**2 /(2*km/k0)**2,'r', linewidth=2)
        th3a,   = ax.plot(km**2*x/k0,np.real(Dk3dth2)/k0**2 /(2*km/k0)**2,'--r', linewidth=2)      

        prop = fm.FontProperties(size=18)
        #first_legend = ax.legend([th2,th3],['Aproximation, $D=1$','Aproximation, $D=2$'],loc=3,prop=prop)#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        plt.legend([th2,th3],['Aproximation, $D=1$','Aproximation, $D=2$'],loc='best', prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$\\Delta  \\theta /(2k_m/k_0)$")
        ax.set_xlabel("$k_m^2x/k_0$")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-6/sigma**2,6/sigma**2)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

#Plot la  dependance en z et k de l'intensite diffusee FSBS, Calcul juste ?
def FSBS_spatial(k,z,k0,zf=0,figure=1,f=6.5,ti=300.,ztesti=3., A=1.):
    kk, zz = np.meshgrid(k, z )
    km=k0/2./f
    #F = (np.cos(5*kk**2*zz/(2*k0) ) -1 )/( 5*kk**2*zz/(2*k0)  )  + (np.cos(3*kk**2*zz/(2*k0) ) -1 )/( 3*kk**2*zz/(2*k0)  ) 
    F = (np.cos(kk**2*(zz - zf)/(4*k0))  - np.cos(kk**2*(z[0]-zf)/(4*k0) ) )/( kk**2/k0**2  ) 
    Fy = np.fft.fftshift( np.fft.fft( F * ( np.abs(kk)<2*km ), axis = 1), axes = 1 ) 
    dy = 2*np.pi/(np.max(k)-np.min(k))
    y = dy*np.linspace(-int(len(k)/2),int(len(k)/2), len(k)+1)
    y = y[np.abs(y)>0]
    
    vphi = -.5*np.abs(kk)/k0 *3e8
    x= vphi/np.sqrt(2*ti/511000./1836./A)/3e8
    ak   = betakr(x,ztesti)
    dk=k[1]-k[0]
    F2=0*zz
    for iz in range(len(z)):
        F2[:,iz]  = np.convolve(ak[:,iz], np.sin(kk[:,iz]**2*(zz[:,iz]-zf)/(4*k0)),mode='same')*dk/(2*km)
    
    fig = plt.figure(figure)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.pcolor(z*1e3,k/k0*f ,F.T)
    plt.colorbar()
    ax.set_xlabel("$z$ mm")
    ax.set_ylabel("$k/k_m$")
    #ax.set_xlim(np.min(x),np.max(x))
    #ax.set_ylim(np.min(y), np.max(y))
    fig.canvas.draw()
    plt.show()

     
    fig = plt.figure(figure+1)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.pcolor(z*1e3, y*1e3 ,-np.real(Fy.T))
    plt.colorbar()
    ax.set_xlabel("$z$ mm")
    ax.set_ylabel("$y$ mm")
    #ax.set_xlim(np.min(x),np.max(x))
    #ax.set_ylim(np.min(y), np.max(y))
    fig.canvas.draw()
    plt.show()
  
    fig = plt.figure(figure+2)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.pcolor(z*1e3, k/k0*f,(F2.T))#,vmin=-100,vmax=100)
    plt.colorbar()
    ax.set_xlabel("$z$ mm")
    ax.set_ylabel("$k/k_m$")
    #ax.set_xlim(np.min(x),np.max(x))
    #ax.set_ylim(np.min(y), np.max(y))
    fig.canvas.draw()
    plt.show()



#OBSOLETE
#Convolution de Fkin avec les H(km-|k|) du spectre de la pompe RPP => FSBS
def convolv2_alpha_kin(Te, Ti, Z, A, nisne, ks, t,k2lde2=0, ne=1,f=6.5,k0=2*np.pi/0.35e-6,xlog=False, ylog=False, figure=1):
    c=3e8
    dk = ks[1]-ks[0]
    km=k0/(2*f)
    kk, tt =np.meshgrid(ks, t) 
    vphi= - np.abs(0.5*kk / k0 *c )
    wp= - np.abs(0.5*kk**2 / k0 *c)   
    F=np.real(alpha_kin(Te, Ti, Z, A, nisne, (vphi), k2lde2, ne)*np.exp(1j * wp  * tt ) )
    Fk2=  kk**2 /km**2 * np.real(alpha_kin(Te, Ti, Z, A, nisne, (vphi), k2lde2, ne)*np.exp(1j * wp  * tt ) )
    #print np.max(F)
    H = np.abs(ks)<2*km
    #H/=np.sum(H)
    cF  = 0*tt
    cFk2= 0*tt
    for i in range(len(t)):
        cF[i,:]   = np.convolve(H,F[i,:],mode='same')*dk / (2*km)
        cFk2[i,:]   = np.convolve(H,Fk2[i,:],mode='same')*dk / (2*km)
    #ccF  = np.convolve(H,cF,mode='same')*dk /(2*km)
    if figure is None: 
        return cF
    if figure is not None:
        #plt.rcParams.update({'font.size': 20})
        #fig = plt.figure(figure, figsize=[8,5])
        #fig.clf()
        #ax = fig.add_subplot(1,1,1)
        #plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        #ak, = ax.plot(ks/k0,F,'k', linewidth=2)              
        #ax.set_ylabel("$g(k)$")
        #ax.set_xlabel("$k/k_0 = \\tan(\\theta)$ ")
        #if ylog:
        #    ax.set_yscale("log")
        #if xlog:
        #    ax.set_xscale("log")
        ##ax.set_xlim(0,np.max(kydose/k0))
        #fig.canvas.draw()
        #plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        plt.pcolor(t*1e12,ks/k0,cF.T)              
        plt.colorbar()
        ax.set_xlabel("$t$ ps")
        ax.set_ylabel("$k/k_0 = \\tan(\\theta)$ ")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()
    
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        plt.pcolor(t*1e12,ks/k0,cFk2.T)              
        plt.colorbar()
        ax.set_xlabel("$t$ ps")
        ax.set_ylabel("$k/k_0 = \\tan(\\theta)$ ")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()

##################################################################################"
#Calcul des matrices de susceptibilite Maxwellien avec derive suivant x et k qqc
# xi    = w/k/sqrt(2T/m) reel
# theta : angle de k avec x 
# vd    : derive normalise a la temperature
chixx =  lambda xi, theta, vd : -1 + np.cos(theta)**2*(1-xi**2*Zp_scalar(xi)) - (vd**2+np.sin(theta)**2)*Zp_scalar(xi)*0.5 -2*vd/2**0.5*np.cos(theta)*xi*Zp_scalar(xi)
chiyy =  lambda xi, theta, vd : -1 + np.sin(theta)**2*(1-xi**2*Zp_scalar(xi)) - np.cos(theta)**2*Zp_scalar(xi)*0.5
chixy =  lambda xi, theta, vd : -np.sin(theta)*np.cos(theta) *xi*(Z_scalar(xi)+xi*Zp_scalar(xi)) - np.sin(theta)*vd/2**0.5*xi*Zp_scalar(xi)
chiyx =  chixy  
Chi   =  lambda xi, theta, vd : np.matrix([[ chixx(xi,theta,vd) , chixy(xi,theta,vd) ],[chixy(xi,theta,vd), chiyy(xi,theta,vd) ]])
####################################################################################"

#Plot la partie cinetique de Drake generalise
#on se restrain a k dans le plan  x y
def Fkin_Drake_generalise(Ti, Te, Z, A,  ksx,ksy, nesnc,  vde=0,vdi=0, k0=2*np.pi/0.35e-6, figure=1,cscale='lin'):
    c=3e8
    w0 = k0*c
    Id =  np.matrix([[1.,0.],[0.,1.]]) 
    wpse = nesnc**0.5*w0
    wpsi = np.sqrt(Z/1836./A)*wpse
    ve, vi = vde/c /np.sqrt(Te/511000.), vdi/c/np.sqrt(Ti/511000./A/1836.)
    #FSBS
    Ky, Kx = np.meshgrid(ksy,ksx) 
    theta  = np.arctan(Ky/Kx) +np.pi*(Kx<0)
    K = np.sqrt(Kx**2+Ky**2)
    w = -0.5*K**2*c**2/w0
    #Calcul de la matrice D 
    ##print   'shape : ',  np.shape(w/K), np.shape(vdi), np.shape(np.cos(theta))
    #print 've, vi = ',ve, vi
    xii = 2**-0.5*(w/K  -vdi*np.cos(theta))/np.sqrt(Ti/511000./1836./A)/c
    xie = 2**-0.5*(w/K  -vde*np.cos(theta))/np.sqrt(Te/511000.)/c
    Fkin =0*Kx + 0j*Kx
    
    for ix in range(len(ksx)):
        #print "Calcul  : ",ix, " / ", len(ksx)
        for iy in range(len(ksy)):
            Chie, Chii = Chi(xie[ix,iy],theta[ix,iy],ve), Chi(xii[ix,iy],theta[ix,iy],vi)
            ##print 'xi = ', xii[ix,iy]
            ##print 'Xe = ',  Chie
            ##print 'Xi = ',  Chii
            th = theta[ix,iy]
            vphi = w[ix,iy]/K[ix,iy]
            D  = Id*(1-c**2/vphi**2) + wpse**2/w[ix,iy]**2*Chie +wpsi**2/w[ix,iy]**2*Chii - c**2/vphi**2*np.matrix([[np.cos(th)**2,np.cos(th)*np.sin(th)],[np.cos(th)*np.sin(th),np.sin(th)**2]])
            Di = Id*(1-c**2/vphi**2) + 0                        +wpsi**2/w[ix,iy]**2*Chii - c**2/vphi**2*np.matrix([[np.cos(th)**2,np.cos(th)*np.sin(th)],[np.cos(th)*np.sin(th),np.sin(th)**2]])
            eparallel = np.matrix([[np.cos(th)], [np.sin(th)]])
            ##print D
            Fkin[ix,iy] = ( eparallel.T*( Di )*D.getI()*eparallel )
            
            vphi = -w[ix,iy]/K[ix,iy]
            D  = Id*(1-c**2/vphi**2) + wpse**2/w[ix,iy]**2*Chie +wpsi**2/w[ix,iy]**2*Chii - c**2/vphi**2*np.matrix([[np.cos(th)**2,np.cos(th)*np.sin(th)],[np.cos(th)*np.sin(th),np.sin(th)**2]])
            Di = Id*(1-c**2/vphi**2) + 0                        +wpsi**2/w[ix,iy]**2*Chii - c**2/vphi**2*np.matrix([[np.cos(th)**2,np.cos(th)*np.sin(th)],[np.cos(th)*np.sin(th),np.sin(th)**2]])
            eparallel = np.matrix([[np.cos(th)], [np.sin(th)]])
            ##print D
            Fkin[ix,iy] += ( eparallel.T*( Di )*D.getI()*eparallel )

            
            ##print Fkin[ix,iy]
    Fkin = np.real(Fkin)
    if figure is not None:
    	plt.rcParams.update({'font.size': 25})
    	fig = plt.figure(figure,figsize=[7,6])
    	fig.clf()
    	ax0 = fig.add_subplot(1,1,1)
    	plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.2)
    	if cscale=='log':
    	    data=np.log10(Fkin.T)
    	    dM = np.max(np.abs(data))
    	    dm = dM-3
    	else:
    	    data=(Fkin.T)
    	    dM = np.max(np.abs(data))
    	    dm = 0
    	cf0 = plt.pcolor(ksx/k0 ,ksy/k0 , data, vmin =-dM, vmax =dM ,cmap =  plt.cm.RdBu)
    	plt.colorbar(cf0)#, ticks=[ 0, cmax/2., cmax])
    	#plt.axes().set_aspect('equal', 'datalim')
    	ax0.set_xlabel("$k_x/k_0$")
    	ax0.set_ylabel("$k_y/k_0$")
    	fig.canvas.draw()

def Z(x):
	res= 1j*np.sqrt(np.pi)*np.exp(-x**2)-1./x
	crit=np.abs(x)<20
	res[crit]=np.exp(-x[crit]**2)*np.pi**0.5*(1j-erfi(x[crit]))
	return res

def Zp(x):
	return -2*(1+x*Z(x))



def Fkin(xie,xii,ztesti):
	sol = -0.5*Zp(xie)*Zp(xii) / (Zp(xii)+Zp(xie)/ztesti)
	return sol

# Convolution  de Im(Fkin exp(iwt)) par la forme du laser
def convolve_alphakin(Te,Ti,kskm,t,k0=2*np.pi/0.35e-6,Zi=1,A=1,f=6.5,figure=1, sigma = 1e10,nu=0.1):
    c  = 3e8
    km=k0/2./f
    k=kskm*km
    dk = (k[1]-k[0] )
    w0 = k0*c
    kk,tt = np.meshgrid(k,t)
    wplus = -0.5*kk**2*c**2/w0
    xii   = np.sqrt(1836.*511.e3/Ti/2.)*wplus/np.abs(kk)/c
    xie   = np.sqrt(511.e3/Te/2.)*wplus/np.abs(kk)/c
    cs=1./np.sqrt(1836.*511.e3/Ti)*(Zi*Te/Ti+1)**0.5*c
    ##print xii
    ##print xie
    def ZZ(x):
        res= 1j*np.sqrt(np.pi)*np.exp(-x**2)-1./x
        crit=np.abs(x)<20
        res[crit]=np.exp(-x[crit]**2)*np.pi**0.5*(1j-erfi(x[crit]))
        return res
    def ZZp(x):
        return -2*(1+x*Z(x))
    def alpha_kin(xie,xii,ztesti):
        return -0.5*ZZp(xie)*ZZp(xii) / (ZZp(xii)+ZZp(xie)/ztesti)
    akin  = -Fkin(xie,xii,Zi*Te/Ti) #* (np.abs(k)>kmin)
    expit = np.exp(1j*wplus*tt)
    if sigma<1e10:
        env =0*kk
        kp=np.linspace(-km,km,100)
        #for i in range(len(kp)):
        #	env += 2*sigma*np.sinc(sigma*(k-2*kp[i])/np.pi) / len(kp)
        #	H = env
        H = -2*(sici( (kk-2*km)*sigma)[0] -sici((kk+2*km)*sigma)[0]) /2./km
    else:
        env=1./2./km
        H = (np.abs(kk)<2*km) *env
    
    cv=0*tt
    #print np.shape(cv)
    cv2=0*tt
    fki=0*tt
    fkr=0*tt
    for it in range(len(t)):
        fki[it,:] = (1-np.exp(-nu*np.abs(kk[it,:]*cs*tt[it,:]))*np.cos(kk[it,:]*cs*tt[it,:]))*np.imag(akin[it,:]*expit[it,:])
        fkr[it,:] = (1-np.cos(kk[it,:]*cs*tt[it,:]))*np.real(akin[it,:]*expit[it,:])
        cv[it,:] = np.convolve(H[it,:],fkr[it,:],mode='same')*dk
        cv2[it,:] = np.trapz(fkr[it,:]*H[it,:],k)*H[it,:]*2*km 
    
    #print np.shape(cv)
    cvm = np.mean(cv *(np.abs(kk)<2*km),axis=1)
    #print np.shape(cv)
    sk=1
    #print np.shape(cv)
    if len(k)>1000:
        sk = int(len(k)/500.) 
    #print np.shape(cv)
    mesh = np.meshgrid(  range(0,len(t),1), range(0,len(k),sk ) )
    #print np.shape(cv)
    cvplot = cv[mesh]
    kplot = k[ range(0,len(k),sk )]/km
    indexmin = np.argmin(np.abs(k+0.1*km))
    indexmax = np.argmin(np.abs(k-0.1*km))
    mesh = np.meshgrid(  range(0,len(t),1),  range(indexmin,indexmax,1 )   )
    fkplot = fki[mesh]
    kplotfk= k[  range(indexmin,indexmax,1 )  ]/km
    #print np.shape(fkplot) , np.shape(kplotfk)
    ##print cv
    
    fig = plt.figure(figure)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.pcolor(t*1e12,kplot,cvplot)
    plt.colorbar()
    ax.set_xlabel("$t$ ps")
    ax.set_ylabel("$k/k_m$")
    #ax.set_xlim(np.min(x),np.max(x))
    #ax.set_ylim(np.min(y), np.max(y))
    fig.canvas.draw()
    plt.show()
    
    fig = plt.figure(figure+1)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.pcolor(t*1e12,kplotfk,fkplot)
    plt.colorbar()
    ax.set_xlabel("$t$ ps")
    ax.set_ylabel("$k/k_m$")    
    #ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(-0.1, 0.1)
    fig.canvas.draw()
    plt.show()

    fig = plt.figure(figure+2)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t*1e12,cvm)
    ax.set_xlabel("$t$ ps")
    ax.set_ylabel("$G_k$")
    #ax.set_xlim(np.min(x),np.max(x))
    #ax.set_ylim(np.min(y), np.max(y))  
    fig.canvas.draw()
    plt.show()




#Plot comparaison FFT RCF et alphakin
#fileRCF="/ccc/cont001/home/f8/ruyer4s/XP_Omega_EP/Lineout_26325H8.txt"
#pour le C5H12 : nhsne=1./(6*5./12. + 1.) et ncsne = 5/12* nesne
def comp_RCF_alphakin(Te,Ti,Z,A,nisne,fileRCF,figure=1,ylog=True,xlog=True,facteur_comp=1):
    c=3e8
    k0=2*np.pi/0.35e-6
    w0 = k0*c
    y, dose = np.loadtxt(fileRCF,unpack=True)
    dose0 = dose - np.mean(dose)
    y=y*1e-6
        
    dosefft = np.fft.fftshift(np.fft.fft( (dose0) ) ) *(y[1]-y[0])    
    dk = 2*np.pi/(np.max(y)-np.min(y))
    kydose = dk*np.linspace(-int(len(y)/2),int(len(y)/2), len(y)+1)
    kydose = kydose[np.abs(kydose)>0]
    #print 'RCF : '
    #print 'kmax / k0 = ',np.max(kydose/k0)
    #print 'kmin / k0 = ',np.min(np.abs(kydose/k0))

    ks = np.linspace(-np.max(kydose),np.max(kydose),1000)
    vphi = 0.5*ks*c**2/w0
    F_kin=np.imag(alpha_kin(Te=Te, Ti=Ti, Z=Z, A=A, nisne=nisne, vphi=vphi, k2lde2=0, ne=1))
        
    ne=0.05*9e27
    Ep=7.5*1e6
    Lz=1200e-6
    dnsn=0.02 #I= 3.10^14 W/cm^2 
    facteur_rcf=Te/(2*Ep)*500e-6 * dnsn* ks * facteur_comp
    #facteur_rcf=   facteur_comp
    
    r=taux_collisionnel(masse=[1., np.mean(A)*1836.],charge=[-1., np.mean(Z)],dens=[ne/1e+6, ne/1e+6/np.mean(Z)],temp=[Te, np.mean(Ti)],vd=[0., 0.])
    loglamb = r['log_coul']
    #print 'Log( Lambda ) = ' , loglamb   
    lei = r['lmfp']*1e-2 
    Ak = 2 * (0.5+ 0.074/(np.abs(ks)*lei) + 0.88*np.mean(Z)**(5./7.) /(np.abs(ks)*lei)**(5./7.) + 2.54*np.mean(Z) /( 1.+5.5*(np.abs(ks)*lei)**2 ) ) 


    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        ak, = ax.plot(ks/k0,facteur_rcf*Ak*np.abs(F_kin),'k', linewidth=2)              
        ak, = ax.plot(kydose/k0,np.abs(dosefft),'--k', linewidth=2)              
        #prop = fm.FontProperties(size=15)
        #ax.legend([h13,h14,h15],['$10^{13}$ W/cm$^{-2}$','$10^{14}$ W/cm$^{-2}$', '$10^{15}$ W/cm$^{-2}$' ], loc=9, bbox_to_anchor=(0.6, 0.35),prop = prop)#, borderaxespad=0.)
        ax.set_ylabel("FFT(dose)")
        ax.set_xlabel("$k_y/k_0$ ")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()

def betak(x, ZTesTi):
    res = np.imag(Zp(x)/(Zp(x)-2./ZTesTi))
    return res
def betakr(x, ZTesTi, zsa = 1):
    xe = x *np.sqrt( zsa / 1836./ ZTesTi )
    res = np.real( Zp(x)/(Zp(x) + Zp(xe)/ZTesTi))
    return res

def d_betak(x, ZTesTi):
    res = -2./ZTesTi * np.imag( Zpp(x) / (Zp(x)-2./ZTesTi)**2 )
    return res
def dd_betak(x, ZTesTi):
    res = -2./ZTesTi* np.imag(( (Zp(x)-2./ZTesTi)*Zppp(x)-2*Zpp(x)**2)/(Zp(x)-2./ZTesTi)**3)
    return res

#Calcul le max de alpha kin et la largeur du pic
def Deltak_ksm_FSBS(ztesti,figure=None,f=6.5,k0=2*np.pi/0.35e-6):
    xim = 0*ztesti
    Dxi = 0*ztesti
    betakrr = 0*ztesti
    xit =np.logspace(-3,0.5,100000) 
    for i in range(len(ztesti)):
        dbeta = d_betak(xit,ztesti[i])
        xim[i] = xit[np.argmin(np.abs(dbeta))]
        Dxit = 0#( np.sqrt(-betak(xim[i],ztesti[i])/dd_betak(xim[i],ztesti[i]) ) )
        #print -betak(xim[i],ztesti[i]), dd_betak(xim[i],ztesti[i])
        Dxi[i] = np.real(Dxit)
        #print 'Precision a ZTe/Ti = ',ztesti[i],' est de ', np.imag(Dxit)/np.real(Dxit) 
        betakrr[i] = np.trapz(betakr(xit, ztesti[i]), xit )
    G=4.*4.*2**0.5/3. * xim*Dxi*betak(xim,ztesti) 
    
    #Deuxieme calcul : int int Im(betak) 
    def int_betak(x,u):
        if np.size(x)==1:
            x=np.array([x])
        sol=0*x
        for l in range(len(x)):
            xxi = np.linspace(np.max([-10,-10+x[l]]),np.min([5,5+x[l]]),1000)
            sol[l] = np.trapz(betak(xxi,u),xxi)
        return np.squeeze( sol )
    def intint_betak(x,u):   
        if np.size(x)==1:
            x=np.array([x])    
        sol=0*x
        for j in range(len(x)):
            xxi = np.linspace(np.max([-10,-10+x[j]]),np.min([5,5+x[j]]),1000)
            #xxi = np.linspace(-10+x[i],10+x[i],10000)
            sol[j] = np.trapz(int_betak(xxi,u),xxi)
        return np.squeeze(sol )
    #G2=0*ztesti
    #x = np.linspace(0,5,100)
    #for i in range(len(ztesti)):
    #    #print "Calcul 2, i = ", i," up to ", len(ztesti)
    #    G2[i] = np.max( intint_betak(x,ztesti[i]) )
    #    #print 'G2 = ', G2[i]
        
    
    
    if figure is None:
        return {'ztesti':ztesti, 'xim':xim, 'Dxi':Dxi}
    else:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        x, = ax.plot(ztesti,xim,'k', linewidth=2)              
        dx, = ax.plot(ztesti,Dxi,'--k', linewidth=2)            	
        prop = fm.FontProperties(size=15)
        ax.legend([x,dx],['$x_\mathrm{max}$','$\Delta x$']
                  ,loc='best', #bbox_to_anchor=(1.35, 0.7), borderaxespad=0., 
                  prop=prop)
  
        #ax.set_ylabel("$\xi_\mathrm{max}$, $\Delta \xi$")
        ax.set_xlabel("$ZT_e/T_i$ ")
        #if ylog:
        ax.set_yscale("log")
        #if xlog:
        ax.set_xscale("log")
        ax.set_xlim(np.min(ztesti),np.max(ztesti))
        fig.canvas.draw()
        plt.show()
 
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        ak, = ax.plot(ztesti,G,'k', linewidth=2)                 
        #ak, = ax.plot(ztesti,G2,'--k', linewidth=2)              

        #ax.set_ylabel("$\xi_\mathrm{max}$, $\Delta \xi$")
        ax.set_xlabel("$ZT_e/T_i$ ")
        #if ylog:
        ax.set_yscale("log")
        #if xlog:
        ax.set_xscale("log")
        ax.set_xlim(np.min(ztesti),np.max(ztesti))
        fig.canvas.draw()
        plt.show()
 
 
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+2, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        ak, = ax.plot(ztesti,betakrr,'k', linewidth=2)                 
        #ak, = ax.plot(ztesti,G2,'--k', linewidth=2)              

        #ax.set_ylabel("$\xi_\mathrm{max}$, $\Delta \xi$")
        ax.set_xlabel("$ZT_e/T_i$ ")
        #if ylog:
        ax.set_yscale("log")
        #if xlog:
        ax.set_xscale("log")
        ax.set_xlim(np.min(ztesti),np.max(ztesti))
        fig.canvas.draw()
        plt.show()
 

def taux_collisionnel(masse=None,charge=None,dens=None,temp=None,vd=None,*aqrgs,**kwargs):
    #% Frequences de collision de Spitzer
    #% Ref. A. Decoster, Modeling of collisions (1998)
    #%
    #% Input : [beam, background]
    #% . masse(1:2)  : masse/me
    #% . charge(1:2) : charge/qe
    #% . dens(1:2)   : densite (cm^-3)
    #% . temp(1:2)   : temperature (eV)
    #% . vde(1:2)    : vitesses de derive (cm/s)
    #% Output :
    #% . nu_imp  : momentum transfer frequency (1/s)
    #% . nu_ener : Energy transfer frequency (1/s)
    #% . lambda  : mean-free-path (cm)
    
    masse = np.array(masse)
    charge = np.array(charge)
    dens = np.array(dens)
    temp = np.array(temp)
    vd = np.array(vd)
    
    #varargin = cellarray(args)
    #nargin = 5-[masse,charge,dens,temp,vd].count(None)+len(args)
    
    vt1=30000000000.0 * (temp[0] / (masse[0] * 511000.0)) ** 0.5
    vd1=np.abs(vd[0])
    if all(masse == 1):
        ##print 'All electrons'
        if temp[1] < 10:
            log_coul=23 - np.log(dens[1] ** 0.5 * temp[1] ** (- 1.5))
        else:
            log_coul=24 - np.log(dens[1] ** 0.5 * temp[1] ** (- 1))
        ##print log_coul, temp
    else:
        if any(masse == 1.):
            ##print 'electron-ion'
            if masse[0] ==1.:
                ielec=0
                ##print 'indice electron: ',ielec
                iion=1
                ##print 'indice ion: ',iion
            else:
                ielec=1
                ##print 'indice electron: ',ielec
                iion=0
                ##print 'indice ion: ',iion
            
            
            if (temp[iion] / masse[iion] < temp[ielec]) and (temp[ielec] < 10 * charge[iion] ** 2):
                log_coul=23 - np.log(dens[ielec] ** 0.5 * charge[iion] * temp[ielec] ** (- 1.5))
            else:
                if 10 * charge[iion] ** 2 < temp[ielec]:
                    log_coul=24 - np.log(dens[ielec] ** 0.5 * temp[ielec] ** (- 1))
                else:
                    if temp[ielec] < temp[iion] * charge[iion] / masse[iion]:
                        mu=masse[iion]/1836.
                        log_coul=30 - np.log(dens[iion] ** 0.5 * temp[iion] **(-1.5) * charge[iion] ** 2 / mu)
                    else:
                        #print( 'No Coulombien logarithm from Lee and Moore')
                        return  {"nup":None,"nuk":None,"log_coul":None,"lmfp":None}
            	##print 'Log Coulombien: ',log_coul
        else:
            log_coul=23 - np.log(charge[0] * charge[1] * (masse[0] + masse[1]) * (dens[0] * charge[0] ** 2 / temp[0] + dens[1] * charge[1] ** 2 / temp[1]) ** 0.5 / (masse[0] * temp[1] + masse[1] * temp[0]))

    qe=4.8032e-10
    temp=1.6022e-19 * 10000000.0 * temp
    masse=9.1094e-28 * masse
    m12=masse[0] * masse[1] / (masse[0] + masse[1])
    nu_imp=(4. / 3.) * (2 * np.pi) ** 0.5 * dens[1] * charge[0] ** 2 * charge[1] ** 2 * qe ** 4 * log_coul / (masse[0] * m12 * (temp[0] / masse[0] + temp[1] / masse[1] + (vd[0] - vd[1]) ** 2) ** 1.5)
    nu_ener=(8. / 3.) * (2 * np.pi) ** 0.5 * dens[1] * charge[0] ** 2 * charge[1] ** 2 * qe ** 4 * log_coul / (masse[0] * masse[1] * (temp[0] / masse[0] + temp[1] / masse[1] + (vd[0] - vd[1]) ** 2) ** 1.5)
    ##print np.max([vt1,vd1])
    _lambda=np.max([vt1,vd1]) / nu_imp
    ##print 'nu_imp = ',nu_imp,' Hz'
    ##print 'nu_ener = ',nu_ener,' Hz'
    ##print 'tau_imp = ',1/nu_imp,' s'
    ##print 'tau_ener = ',1/nu_ener,' s', dens, log_coul,masse
    ##print 'log_coul = ',log_coul
    ##print 'Mean-free-path: ',_lambda,' cm'
    result = {"nup":nu_imp,"nuk":nu_ener,"log_coul":log_coul,"lmfp":_lambda}
    return result


