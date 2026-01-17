import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([x for x in range(0,101)])
y = np.array([1 if i > 49 else 0 for i in x])
#y = y * y  # squaring step

def grd(x,y):
   m=0
   c=-1
   iter=100
   n=len(x)
   lern=0.001
   lamda = 0.9
   plt.scatter(x, y,c="blue", label="actual points")
   plt.legend()
   for i in range(iter):
       yperr=1/(1+np.exp((-m*x)-c))
       #partial differentation of MSE
       md=-(2/n)*sum(((1/yperr)**-2)*(y-yperr)*(x*np.exp((-m*x)-c)))
       cd=-(2/n)*sum(((1/yperr)**-2)*(y-yperr)*(np.exp((-m*x)-c)))
       mse =np.mean((y - yperr)**2)
       m-=lern*md
       c-=lamda*cd


       print(f"c= {c} m= {m} itter= {i} cost= {mse}")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.01)

   plt.show()


def grd2(x,y):
   m=0
   c=-1
   iter=100
   n=len(x)
   lern=0.001
   lamda=0.9

   for i in range(iter):
       yperr=1/(1+np.exp((-m*x)-c))
       #partial differentation of MSE
       md = -(2 / n) * sum(((1 / yperr) ** -2) * (y - yperr) * (x * np.exp((-m * x) - c)))
       cd = -(2 / n) * sum(((1 / yperr) ** -2) * (y - yperr) * (np.exp((-m * x) - c)))
       mse = np.mean((y - yperr)**2)
       m -= lern * md
       c -= lamda * cd

       plt.cla()
       plt.scatter(x, y, c="blue", label="actual points")
       print(f"c= {c} m= {m} itter= {i} cost= {mse}")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.01)
       plt.legend()

   plt.show()


grd2(x,y)
grd(x,y)
