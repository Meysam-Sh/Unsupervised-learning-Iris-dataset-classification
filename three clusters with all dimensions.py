# imports 
import numpy as np
from math import sqrt, pi, exp
from sklearn.datasets import load_iris

# imports
iris = load_iris()
x = iris.data
N = 150

Gauss = np.zeros((3, 3, 4))
Gauss[0,:,:] = [[2.5,3.7,1.7,.3],       [.6,.7,.4,.3],    [1,1,1,1]]
Gauss[1,:,:] = [[3,2.8,4.4,1.5],        [.7,.5,.6,.6],    [1,1,1,1]]
Gauss[2,:,:] = [[3.3,3.2,5.6,1.9],      [.5,.5,.5,.5],    [1,1,1,1]]

def EM2(i):
  responsibility1 = []
  responsibility2 = []
  responsibility3 = []
  log_likelihood = []
  for j in range(0,N):
      
    a = (1/sqrt(2*pi*Gauss[0,1,i]))*exp(-(x[j,i]-Gauss[0,0,i])**2/(2*Gauss[0,1,i]))
    b = (1/sqrt(2*pi*Gauss[1,1,i]))*exp(-(x[j,i]-Gauss[1,0,i])**2/(2*Gauss[1,1,i]))
    c = (1/sqrt(2*pi*Gauss[2,1,i]))*exp(-(x[j,i]-Gauss[2,0,i])**2/(2*Gauss[2,1,i]))
      
  
    probability_x = a*Gauss[0,2,i]+b*Gauss[1,2,i]+c*Gauss[2,2,i]
    responsibility1.append(a*Gauss[0,2,i]/probability_x)
    responsibility2.append(b*Gauss[1,2,i]/probability_x)
    responsibility3.append(c*Gauss[2,2,i]/probability_x)
    log_likelihood.append(np.log(probability_x))
    
#===========================================================================
  log_likelihood = sum(log_likelihood)
  N1 = sum(responsibility1)
  N2 = sum(responsibility2)
  N3 = sum(responsibility3)
  sum1 = 0
  sum2 = 0
  sum3 = 0
  for k in range(0,N):
    sum1 += responsibility1[k]*x[k,i]
    sum2 += responsibility2[k]*x[k,i]
    sum3 += responsibility3[k]*x[k,i]
  Gauss[0,0,i] = sum1/N1
  Gauss[1,0,i] = sum2/N2
  Gauss[2,0,i] = sum3/N3

  sum1 = 0
  sum2 = 0
  sum3 = 0
  for l in range(0,N):
    sum1 += responsibility1[l]*((x[l,i]-Gauss[0,0,i])**2)
    sum2 += responsibility2[l]*((x[l,i]-Gauss[1,0,i])**2)
    sum3 += responsibility3[l]*((x[l,i]-Gauss[2,0,i])**2)
  Gauss[0,1,i] = sum1/N1
  Gauss[1,1,i] = sum2/N2
  Gauss[2,1,i] = sum3/N3

  Gauss[0,2,i] = N1/N
  Gauss[1,2,i] = N2/N
  Gauss[2,2,i] = N3/N
   
  return Gauss, log_likelihood

for i in range(0,4):
  if i == 0:
    dim = "SEPAL LENGTH"
  elif i ==1:
    dim = "SEPAL WIDTH"
  elif i == 2:
    dim = "PETAL LENGTH"
  else:
    dim = "PETAL WIDTH"
  print('\n \n***********************************************************************************************************************')
  print('**************************************************  %s  *****************************************************'%dim)
  print('***********************************************************************************************************************\n \n')
  for p in range(0,20):
    Gauss, log_likelihood = EM2(i)
    print('====================================================================') 
    print("**%s**              mean         std_dev        mix_coeff \n"%dim)
    print("                 cluster1:","  ",round(Gauss[0,0,i],4),"     ",round(Gauss[0,1,i],4),"      ",round(Gauss[0,2,i],4))
    print("                 cluster2:","  ",round(Gauss[1,0,i],4),"     ",round(Gauss[1,1,i],4),"      ",round(Gauss[1,2,i],4))
    print("                 cluster3:","  ",round(Gauss[2,0,i],4),"     ",round(Gauss[2,1,i],4),"      ",round(Gauss[2,2,i],4))
    print("\n                 log likelihood:",log_likelihood)
