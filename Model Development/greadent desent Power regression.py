import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data2")
x=np.array(data['x'])
y=np.array(data['y'])
y=y**(2)

def grd(x,y):
    m = c = 0
    p = 1
    iter = 200
    n = len(x)
    lern = 0.0000001

    plt.scatter(x, y,c="blue", label="actual points")
    plt.legend()
    for i in range(iter):
       yperr = (m * (x ** p)) + c
       # partial differentation of MSE
       mse = np.mean((y - yperr) ** 2)
       md = -(2 / n) * sum(x * (y - yperr))
       cd = -(2 / n) * sum(y - yperr)
       pd = -(2 / n) * sum(m * p * x * (y - yperr))
       m -= lern * md
       c -= lern * cd
       p -= lern * pd


       print(f"c= {c} m= {m} p= {p} itter= {i} cost= {mse}")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.0000001)

    plt.show()


def grd2(x,y):
   m=c=0
   p=1
   iter=200
   n=len(x)
   lern = 0.0000001
   for i in range(iter):
       yperr=(m*(x**p))+c
       #partial differentation of MSE
       mse =  np.mean((y - yperr)**2)
       md = -(2 / n) * sum(x * (y - yperr))
       cd = -(2 / n) * sum(y - yperr)
       pd = -(2 / n) * sum(m*p*x*(y - yperr))
       m-=lern*md
       c-=lern*cd
       p-=lern*pd

       plt.cla()
       plt.scatter(x, y, c="blue", label="actual points")
       print(f"c= {c} m= {m} p= {p} itter= {i} cost= {mse}")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.000001)
       plt.legend()

   plt.show()



grd2(x,y)
grd(x,y)