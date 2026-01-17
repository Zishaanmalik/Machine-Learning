import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data2")
x = np.arange(0,1000)
np.random.seed(0)
y = np.arange(0,1000)+(np.random.choice(np.arange(0,10),size=1))
#y = y * y  # squaring step

def grd(x,y):
   mc=c=0
   iter=100
   n=len(x)
   lern=np.linspace(0.0001,1e-11,7)
   lamda = 0.02
   plt.scatter(x, y,c="blue", label="actual points")
   plt.legend()
   for i in range(iter):
       yperr=(mc*x)+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))+ (lamda * mc * mc)
       cd=-(2/n)*sum(y-yperr)
       mse =np.mean((y - yperr)**2)
       mse_int = int(mse)
       n = len(str(mse_int))
       print(n)

       lern =10**-n

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
   iter=50
   n=len(x)
   lamda=0.02

   for i in range(iter):
       yperr=mc*x+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))+ (lamda * mc * mc)
       cd=-(2/n)*sum(y-yperr)
       mse = np.mean((y - yperr)**2)
       mse_int = int(mse)
       n = len(str(mse_int))
       print(n)

       #lern = [10**(-n-2),10**(-n-3),10**(-n-4),10**(-n-5),10**(-n-6),10**(-n-7),10**(-n-8)]
       lern = np.linspace(  10**-n,10**-(n*6),7)
       print(lern)
       if mse > 10000:
           mc -= lern[0] * md
           c -= lern[0] * cd
           print("lern[0]")
       elif mse > 1000:
           mc -= lern[1] * md
           c -= lern[1] * cd
           print("lern[1]")
       elif mse > 100:
           mc -= lern[2] * md
           c -= lern[2] * cd
           print("lern[2]")
       elif mse > 10:
           mc -= lern[3] * md
           c -= lern[3] * cd
           print("lern[3]")
       elif mse > 1:
           mc -= lern[4] * md
           c -= lern[4] * cd
           print("lern[4]")
       elif mse > 0.1:
           mc -= lern[5] * md
           c -= lern[5] * cd
           print("lern[5]")
       elif mse < 0.1:
           mc -= lern[6] * md
           c -= lern[6] * cd
           print("lern[6]")

       plt.cla()
       plt.scatter(x, y, c="blue", label="actual points")
       print(f"c= {c} m= {mc} itter= {i} cost= {mse}")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.01)
       plt.legend()

   plt.show()
   print('\n\n\n\n')


#grd2(x,y)
grd(x,y)
