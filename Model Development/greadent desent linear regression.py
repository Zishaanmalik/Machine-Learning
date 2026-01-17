import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
data=pd.read_csv("data2")
x=np.array(data['x'])
y=np.array(data['y'])
#y=y*y


def grd(x,y):
   mc=c=0
   iter=100
   n=len(x)
   lern=0.00001
   plt.scatter(x, y,c="blue", label="actual points")
   plt.legend()
   for i in range(iter):
       yperr=(mc*x)+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))
       cd=-(2/n)*sum(y-yperr)
       mse = np.mean((y - yperr)**2)
       mc -= lern * md
       c -= lern * cd

       print(f"c= {c} m= {mc} itter= {i} cost= {mse}")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.01)

   plt.show()


def grd2(x,y):
   mc=c=0
   iter=100
   n=len(x)
   lern=0.00001

   for i in range(iter):
       yperr=mc*x+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))
       cd=-(2/n)*sum(y-yperr)
       mse = np.mean((y - yperr)**2)
       mc -= lern * md
       c -= lern * cd
       print(f"c= {c} m= {mc} itter= {i} cost= {mse}")

       plt.cla()
       plt.scatter(x, y, c="blue", label="actual points")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.01)
       plt.legend()

   plt.show()


grd2(x,y)
grd(x,y)


