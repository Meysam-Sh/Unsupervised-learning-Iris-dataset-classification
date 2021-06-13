# imports
import numpy as np
from math import sqrt, pi, exp
from sklearn.datasets import load_iris

# loading the data
iris = load_iris()
x = iris.data
x = x[:,3]
N = 150

Gauss = np.zeros((3, 3))
Gauss[0,:] = [.3,.3,1]
Gauss[1,:] = [1.5,.6,1]
Gauss[2,:] = [1.9,.5,1]

def EM3():
  responsibility1 = []
  responsibility2 = []
  responsibility3 = []
  log_likelihood = []
  
#========================= E STEP ==========================================
  for i in range(0,N):
    
    a = (1/sqrt(2*pi*Gauss[0,1]))*exp(-(x[i]-Gauss[0,0])**2/(2*Gauss[0,1]))
    b = (1/sqrt(2*pi*Gauss[1,1]))*exp(-(x[i]-Gauss[1,0])**2/(2*Gauss[1,1]))
    c = (1/sqrt(2*pi*Gauss[2,1]))*exp(-(x[i]-Gauss[2,0])**2/(2*Gauss[2,1]))
  
    probability_x = a*Gauss[0,2]+b*Gauss[1,2]+c*Gauss[2,2]
    responsibility1.append(a*Gauss[0,2]/probability_x)
    responsibility2.append(b*Gauss[1,2]/probability_x)
    responsibility3.append(b*Gauss[2,2]/probability_x)
    log_likelihood.append(np.log(probability_x))
    
#========================= M STEP ===========================================
  log_likelihood = sum(log_likelihood)
  N1 = sum(responsibility1)
  N2 = sum(responsibility2)
  N3 = sum(responsibility3)
  sum1 = 0
  sum2 = 0
  sum3 = 0
  for i in range(0,N):
    sum1 += responsibility1[i]*x[i]
    sum2 += responsibility2[i]*x[i]
    sum3 += responsibility3[i]*x[i]
  Gauss[0,0] = sum1/N1
  Gauss[1,0] = sum2/N2
  Gauss[2,0] = sum3/N3

  sum1 = 0
  sum2 = 0
  sum3 = 0
  for i in range(0,N):
    sum1 += responsibility1[i]*((x[i]-Gauss[0,0])**2)
    sum2 += responsibility2[i]*((x[i]-Gauss[1,0])**2)
    sum3 += responsibility3[i]*((x[i]-Gauss[2,0])**2)
  Gauss[0,1] = sum1/N1
  Gauss[1,1] = sum2/N2
  Gauss[2,1] = sum3/N3
  
  Gauss[0,2] = N1/N
  Gauss[1,2] = N2/N
  Gauss[2,2] = N3/N
  
  return Gauss, log_likelihood

 
for j in range(0,10):
  Gauss, log_likelihood = EM3()
  print('====================================================================') 
  print("**PETAL WIDTH**              mean         std_dev        mix_coeff \n")
  print("                 cluster1:","  ",round(Gauss[0,0],4),"     ",round(Gauss[0,1],4),"      ",round(Gauss[0,2],4))
  print("                 cluster2:","  ",round(Gauss[1,0],4),"     ",round(Gauss[1,1],4),"      ",round(Gauss[1,2],4))
  print("                 cluster3:","  ",round(Gauss[2,0],4),"     ",round(Gauss[2,1],4),"      ",round(Gauss[2,2],4))
  print("\n                 log likelihood:",log_likelihood)
