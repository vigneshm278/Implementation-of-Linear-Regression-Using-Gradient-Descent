# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: HARINI V
RegisterNumber:212222230044  
```
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header = None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function of using the in a linear regression model
  """
  m=len(y) # length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning J

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function

from matplotlib.container import ErrorbarContainer
from IPython.core.interactiveshell import error
def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take the numpy array X,y,theta and update theta by taking the num_tiers gradient with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """

    m=len(y)
    J_history=[]

    for i in range(num_iters):
      predictions=X.dot(theta)
      error=np.dot(X.transpose(),(predictions -y))
      descent=alpha *1/m*error
      theta-=descent
      J_history.append(computeCost(X,y,theta))

    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000")
plt.title("Profit Prediction"

def predict(x,theta):
  """
  Tkes in numpy array of x and theta and return the predicted value of y base
  """

  predictions=np.dot(theta.transpose(),x)

  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population =35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
### Profit Prediction Graph :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/d64e3ca6-c94d-49b5-9116-a817b1d6d623)

![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/92625829-e1c6-473f-8f6a-f00d6209bdd6)
### Compute Cost Value :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/9e384697-fc9f-4277-92c1-841b285cd101)
### h(x) Value :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/d8b272d5-104d-4cdb-942c-b849e8b54300)
### Cost function using Gradient Descent Graph :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/c1ac8e0b-f252-4aac-8984-2c9f00da624a)
### Profit for the Population 35,000 :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/61aad46d-b2d2-47d7-a7d7-ece05043cf30)
### Profit for the Population 70,000 :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/70f9f953-a1da-4225-be06-19b89e9b42fe)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
