#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Calcul de la constante

from mpmath import zeta,exp,log, sqrt
import primefac
from time import time
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

nbr_rec=100 #nombre de rectangles pour C6
varkappa=1/8
gammaa=gamma(1/4)/gamma(3/4)
nbre_theta=10000 #échantillon de theta à optimiser

def f1(p):
    return((3*p**2-3*p+1)/((p-1)**3*p))

def f2(p):
    #defini parge 33, C7
    return((5*p**5-6*p**4+5*p**2-4*p+1)/((p-1)**5*p*(p+1)))

def prods():
    l=list(primefac.primegen(limit=1000))
    P1=1
    P2=1
    for p in l:
        P1*=(1+f1(p))
        P2*=(1+f2(p))
    return(P1.n(),P2.n())

P1,P2=prods()

def rho(t):
    return(numerical_approx(find_root(-1+2*t*x+e^(x*(1-t))*(-1+2*x),0,1)))

def rho2(t,v):
    a=sqrt(pi*varkappa*v)
    b=gammaa
    def f(x):
        return(2*t*x*(a+b*sqrt(x))-2*a-b*sqrt(x)+exp(x*(1-t))*(2*x*(a+b*sqrt(x))-2*a-b*sqrt(x)))
    return(numerical_approx(find_root(f,0.1,1)))
    
def C2(x):
    c=C3(x)
    return(6*(c+1+2*sqrt(c))/(x*varkappa)**2)

def C3(x):
    r=rho(x)
    return((1/(8*varkappa)+3/2)*((exp(r)+exp(r*x))/((1-x)*sqrt(pi*r))*gammaa)**4*P1)

def C4(x):
    return(1/(3*(1-x)*sqrt(pi))*(9/5*(1+x**(5/2))+2*max(3*sqrt(1-x),min(max(1,abs(1-4*x))+3*x*sqrt(x),1-x+3*x*sqrt(1-x)))))

def C5(x):
    r=rho(x)
    return(1/(1-x)*(exp(r)+exp(r*x))/(2*sqrt(pi*varkappa*r))*gammaa)

def C6(t,v):
    r=rho2(t,v)
    return(numerical_approx(1/(1-t)*(exp(r)+exp(r*t))/(2*sqrt(pi*varkappa*r))*(sqrt(v/r)*sqrt(pi*varkappa)+gammaa)))
    
C7=32/3

def bornes_int(N,f,a,b): #majore et minore l'intégrale d'une fonction croissante f sur [a,b]. Méthode des rectangles (N rectangles)
    Sleq=0
    Sgeq=0
    pas=(b-a)/N
    for i in range(N):
        Sleq+=f(a+i*pas)*pas
        Sgeq+=f(a+(i+1)*pas)*pas
    return(numerical_approx(Sleq),numerical_approx(Sgeq))

def simpson(N,f,a,b):#methode de simpson 3/8, inutilisable car pas de majoration/minoration
    S=0
    pas=(b-a)/N
    for i in range(N):
        aa=i*pas
        bb=aa+pas
        S+=pas/8*(f(aa)+3*f((2*aa+bb)/3)+3*f((aa+2*bb)/3)+f(bb))
    return(numerical_approx(S))
    
def k1(x):
    return(P1*C7/(sqrt(pi)*(1-x)))
           
def k2(x,rec=nbr_rec):
    def f(t):
        return(C6(x,t)**2*(1/2+2*varkappa)+2*C4(x)*C6(x,t)*sqrt(varkappa))
    return(varkappa*P2*bornes_int(rec,f,0,1/varkappa)[1]+P1*(varkappa/(1/2-2*varkappa)+C7**2/(4*(1-x)**2*pi)+C7/(sqrt(pi)*(1-x))*(log(varkappa)-varkappa)))

def k3(x):
    return(-C7**2*P1/(4*(1-x)**2*pi*varkappa))

def k4(x,rec=nbr_rec):
    def f(t):
        return(t*(C6(x,t)**2*(1/2+2*varkappa)+2*C4(x)*C6(x,t)*sqrt(varkappa)))
    return(-varkappa*P2*bornes_int(rec,f,0,1/varkappa)[0]+P1*((4*varkappa-1/2)/(1/2-2*varkappa)-C7**2/(4*(1-x)**2*pi*varkappa)*(1+log(varkappa))+C7/(sqrt(pi)*(1-x)*varkappa)))

def C1(A,x):
    return(8*C5(x)**2*(k1(x)*A*log(A)+k2(x)*A+k3(x)*log(A)+k4(x)))

A=SR.symbol('A')

def find_opti(N,c5,K1,K2,K3,K4,c2):#optimise en A, à C1 et C2 fixés
    def c(A):
        return(8*c5**2*(K1*A*(A.log(hold=True))+K2*A+K3*(A.log(hold=True))+K4))
    
    if N==1:
        def mino1(A): #à part, le cas N=1 car la borne inf a une meilleure minoration
            return(2*pi*(1/(2*A)-(c(A).sqrt(hold=True)+numerical_approx(sqrt(c2)))**2/A**3))
        g1=derivative(mino1(A),A)
        i=10
        while(g1(i)*g1(10)>0):
            i*=10
        A0=find_root(g1(x),10,i)
        return(2*pi*(1/(2*A0)-(sqrt(8*c5**2*(K1*A0*log(A0)+K2*A0+K3*log(A0)+K4))+sqrt(c2))**2/A0**3),A0)
    
    def mino(A):
        return(2*pi*(1/(2*A)-4*N*(c(A)+c2)/A**3))
    g=derivative(mino(A),A)
    i=10
    while(g(i)*g(10)>0):
        i*=10
    A0=find_root(g(x),10,i)
    return(2*pi*(1/(2*A0)-4*N*(8*c5**2*(K1*A0*log(A0)+K2*A0+K3*log(A0)+K4)+c2)/A0**3),A0)

def auxiliaire(x,rec=nbr_rec):
    return(numerical_approx(C5(x)),numerical_approx(k1(x)),numerical_approx(k2(x,rec)),numerical_approx(k3(x)),numerical_approx(k4(x,rec)),numerical_approx(C2(x)))

def theta_opti(L=[1,2,3,4,5,10,100,1000],k=nbre_theta,A=0,B=1):#optimise le résultat en theta, en coupant [A,B] en k-1, pour une liste L de N
    thetas=[A+i/k*(B-A) for i in range(1,k)]
    
    t1=time()

    pool = mp.Pool(mp.cpu_count())
    coefficients = pool.map(auxiliaire, thetas)
    pool.close()
    
    t2=time()
    
    print(t2-t1)
    
    res_final=[]
    for N in L:
        def aux(tupple):
            a,b,c,d,e,f=tupple
            return(find_opti(N,a,b,c,d,e,f))
        results=list(map(aux,coefficients))
        i=0
        theta_max=A+(B-A)/k
        maxi=results[0]
        for (a,b) in results:
            i+=1
            if maxi[0]<a:
                maxi=(a,b)
                theta_max=A+i/k*(B-A)
        res_final.append((maxi[0],maxi[1],theta_max,N))#Densité, A pour lequel elle est atteinte, theta optimal et N
    return(res_final)

#Calcul quand N est grand

def C6p(v):
    return(e/(sqrt(pi*varkappa/2))*(sqrt(2*v*pi*varkappa)+gammaa))

def C5aux(x):
    r=rho(x)
    return((exp(r)+exp(r*x))/(2*sqrt(pi*varkappa*r))*gammaa)
    
def C5p():
    return(C5aux(1/4))

def C5m():
    return(C5aux(0))

def lambdam():
    return(128*C5m()**2*k1(0))

def lambdap():
    return(128*C5p()**2*k1(0))

def C3p():
    r=rho(1/4)
    return((1/(8*varkappa)+3/2)*((exp(r)+exp(r/4))/(sqrt(pi*r))*gammaa/sqrt(pi/2))**4*P1)

def C2p():
    c=C3p()
    return(6*(c+1+2*sqrt(c))/varkappa**2)

def N0():
    return(C2p()/lambdam()**3)

def C4p():
    return(13/(5*sqrt(pi)))

def k2p():
    def f(t):
        return(C6p(t)**2*(1/2+2*varkappa)+2*C4p()*C6p(t)*sqrt(varkappa))
    return(varkappa*P2*bornes_int(100,f,0,1/varkappa)[1]+P1*(varkappa/(1/2-2*varkappa)+C7**2/(4*pi)))

def k4p():
    return(P1*(C7**2/(4*pi*varkappa)*(log(1/varkappa)-1)+C7/(sqrt(pi)*varkappa)))

def largeN():
    NN=SR.var('NN')
    eps=SR.var('eps')
    l=lambdap()
    k=k1(0)
    lm=lambdam()
    def arrondi(x):
        return(numerical_approx(x, digits=4))
    a=2*pi/(4*l)
    b=l/lm
    c=(log(l)+1+k2p()/k)*l/lm
    d=(16+k4p()/(k*9*lm**2*N0()))*l
    a,b,c,d=tuple(map(arrondi,[a,b,c,d]))
    return(a/(NN*NN.log(hold=True))*(1-b*NN.log(hold=True).log(hold=True)/(NN.log(hold=True))-c/(NN.log(hold=True))-3*eps-d*eps/(NN.log(hold=True))**2))


# In[ ]:


#Mollification graphe, à ne pas compiler en même temps que le code du dessus

from scipy.special import zeta,gamma
from sympy import symbols, solve
from sympy import Symbol
x = Symbol('x')
from mpmath import zeta,exp,log, sqrt,pi
import numpy as np
import matplotlib.pyplot as plt
from sympy.ntheory.factor_ import totient
import primefac
from collections import Counter
i=complex(0,1)
from sympy import symbols, solve

kappa=1/8

def f1(p):
    return((3*p**2-3*p+1)/((p-1)**3*p))

def f2(p):
    return((5*p**5-6*p**4+5*p**2-4*p+1)/((p-1)**5*p*(p+1)))

def f3(p):
    return((4*p**3-6*p**2+4*p-1)/((p-1)**4*p))

def prods():
    l=list(primefac.primegen(limit=1000))
    P1=1
    P2=1
    P3=1
    for p in l:
        P1*=(1+f1(p))
        P2*=(1+f2(p))
        P3*=(1+f3(p))
    return(P1,P2,P3)

P1,P2,P3=prods()

def dichotomie():
    a=0
    b=0
    def f(x):
        return(-1+x+exp(x*1/2)*(2*x-1))
    e=0.001
    debut = a
    fin = b
    m=0
    #calcul de la longueur de [a,b]
    ecart = b-a
    while ecart>e:
        #calcul du milieu
        m = (debut+fin)/2
        if f(m)>0:
            #la solution est inférieure à m
            fin = m
        else:
            #la solution est supérieure à m
            debut = m
        ecart = fin-debut
    return m

def inter(A):
    return (-6.28318530717959*(1.72247853677322e7*A*log(A) + 1.85161314459044e6*A**(3/2) + 1.29916576565609e8*A + 2.03702872594442e9)/A**2 + 6.28318530717959)/A

def tracer():
    ab=[k*0.01 for k in range(1,100)]
    y=[inter(x) for x in ab]
    plt.plot(ab,y)
    plt.show()

def theta(t):
    return(1/2*t*log(t/2*pi)-1/2*t-pi/8-1/(48*t))

def X(t):
    return(np.real(exp(i*theta(t))*zeta(1/2+i*t)))

def create_count_list(lst):
    count_list = []
    count_dict = {}
    for element in lst:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1
    for element, count in count_dict.items():
        count_list.append((element, count))
    return count_list

def facto(n):
    l=list(primefac.primefac(n))
    return(create_count_list(l))

def tau(n):
    l=facto(n)
    P=1
    for (p,a) in l:
        P*=gamma(-1/2+a)/(gamma(-1/2)*gamma(a+1))
    return P

def M_sel(T,x):
    xi=T**(1/8)
    if 1<=x and x<=xi:
        return((1-log(x))/log(xi))
    return(0)

def M(T,x):
    theta=0.9
    return(1/(1-theta)*(M_sel(T,x)-theta*M_sel(sqrt(T),x)))

def eta_sel(t, T):
    S=0
    j=1
    while j<=T**(1/8):
        S+=tau(j)/j**(1/2+i*t)*M_sel(T,j)
        j+=1
    return(abs(S**2))

def eta(t,T):
    S=0
    j=1
    while j<=T**(1/8):
        S+=tau(j)/j**(1/2+i*t)*M(T,j)
        j+=1
    return(abs(S**2))

def eta_inte(t,T):
    S=0
    j=1
    while j<=T**(1/8):
        S+=tau(j)/j**(1/2+i*t)
        j+=1
    return(abs(S**2))

def tracer2(T):
    TT=int(T/10)
    def aux1(t):
        return(X(t)*eta(t,T))
    def aux2(t):
        return(X(t)*eta_sel(t,T))
    ab=np.linspace(T,T+3,500)
    y=[aux1(x) for x in ab]
    yy=[aux2(x) for x in ab]
    yyy=[X(x) for x in ab]
    plt.plot(ab,y, label="Our mollifer")
    plt.plot(ab,yy, label="Selberg's mollifer")
    plt.plot(ab,yyy, label="No mollifer")
    plt.legend()
    plt.savefig('figure1.eps', format='eps')
    plt.show()


# In[ ]:




