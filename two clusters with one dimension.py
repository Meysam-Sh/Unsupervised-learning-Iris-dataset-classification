# imports
from math import sqrt, pi, exp
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
plt.plot?

# loading the dataset
iris = load_iris()
x = iris.data
x = x[:,2]
N = 150

# number of clusters
Gauss1 = [.5,.5,1]
Gauss2 = [2.5,.7,1]

def EM1():
  responsibility1 = []
  responsibility2 = []
  log_likelihood = []
  
#========================= E STEP ==========================================
  for i in range(0,N):
    
    a = (1/sqrt(2*pi*Gauss1[1]))*exp(-(x[i]-Gauss1[0])**2/(2*Gauss1[1]))
    b = (1/sqrt(2*pi*Gauss2[1]))*exp(-(x[i]-Gauss2[0])**2/(2*Gauss2[1]))
  
    probability_x = a*Gauss1[2]+b*Gauss2[2]
    responsibility1.append(a*Gauss1[2]/probability_x)
    responsibility2.append(b*Gauss2[2]/probability_x)
    log_likelihood.append(np.log(probability_x))
    
#========================= M STEP ===========================================
  log_likelihood = sum(log_likelihood)
  N1 = sum(responsibility1)
  N2 = sum(responsibility2)
  sum1 = 0
  sum2 = 0
  for i in range(0,N):
    sum1 += responsibility1[i]*x[i]
    sum2 += responsibility2[i]*x[i]
  Gauss1[0] = sum1/N1
  Gauss2[0] = sum2/N2

  sum1 = 0
  sum2 = 0
  for i in range(0,N):
    sum1 += responsibility1[i]*((x[i]-Gauss1[0])**2)
    sum2 += responsibility2[i]*((x[i]-Gauss2[0])**2)
  Gauss1[1] = sum1/N1
  Gauss2[1] = sum2/N2

  Gauss1[2] = N1/N
  Gauss2[2] = N2/N 
  
  return Gauss1, Gauss2, log_likelihood

 
for j in range(0,10):
  Gauss1, Gauss2, log_likelihood = EM1()
  print('====================================================================') 
  print("**PETAL LENGTH**              mean         std_dev        mix_coeff \n")
  print("                 cluster1:","  ",round(Gauss1[0],4),"     ",round(Gauss1[1],4),"      ",round(Gauss1[2],4))
  print("                 cluster2:","  ",round(Gauss2[0],4),"     ",round(Gauss2[1],4),"      ",round(Gauss2[2],4))
  print("\n                 log likelihood:",log_likelihood)
