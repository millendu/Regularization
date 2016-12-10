__author__ = 'manideep'

#REFERENCES
# http://www.hongliangjie.com/notes/lr.pdf
# http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf

# ----------------------------------------------------------------------------------------

#imports for the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D

#global variables
alpha = 0.001
num_iters = 4000

#parameters for normalizing the data
mean_X = 0.0 #mean of X(input data)
mean_y = 0.0 #mean of y(class lables)
std_X = 0.0

#Initializing the w ndarray with 0's
w = np.ones
error_list = []
w0 = []
w1 =[]
w2 = []

#Function for fitting the data
def fit(X, y, beta):
    global w
    global mean_y
    global mean_X
    global std_X

    #print(type(X))
    Xn = np.ndarray.copy(X)
    #print(Xn)
    yn = np.ndarray.copy(y)

    #initializing the values of w to be zero(size of w will be 1+num_of_features)
    w = np.ones((Xn.shape[1] + 1, 1)) * 3.5

    #Normalizing the input data
    mean_X = np.mean(Xn , axis = 0)
    Xn = Xn - mean_X
    std_X = np.std(Xn, axis = 0)
    std_X[std_X == 0] = 1
    Xn = Xn / std_X

    #Normalizing the class labels
    mean_y = yn.mean(axis = 0)

    #Estimating the values of w using gradient descent
    gradient_descent(Xn, yn, beta)


def cost(Xn, y, wIter, beta):

    m = X.shape[0]
    # for L2 put J = (1. / (2. * m)) * (np.sum((np.dot(X, wIter) - y) ** 2.) + beta * np.dot(wIter.T, wIter)
    # for L1 put J = (1. / (2. * m)) * (np.sum((np.dot(X, wIter) - y) ** 2.) + beta * np.dot(wIter,np.ones((wIter.shape[0], 1))))
    # for Lp put J = (1. / (2. * m)) * (np.sum((np.dot(X, wIter) - y) ** 2.))
    J = (1. / (2. * m)) * (np.sum((np.dot(X, wIter) - y) ** 2.) + beta * np.dot(wIter,np.ones((wIter.shape[0], 1))))

    # for L2 return J
    # for L1 return J[0]
    # for Lp return J + beta * (math.sqrt(w1) + math.sqrt(w2))
    return J[0]


#Gradient descent function for estimating w
def gradient_descent(X, y, beta):
    global w
    global error_list
    global w1
    global w2
    global w0
    #adding the intercept term as 1 for the data so that it can be used easily for dot product and calculation later
    X = np.hstack((np.ones(X.shape[0])[np.newaxis].T, X))

    m = X.shape[0]

    x = w.tolist()
    w0.append(x[0][0])
    w1.append(x[1][0])
    w2.append(x[2][0])




    #gradient descent for estimating w
    for i in range(num_iters):
        # print(i)
        # for L1 w = w - (alpha / m) * (np.dot(X.T, (X.dot(w) - y[:, np.newaxis])) + beta )
        # for L2 w = w - (alpha / m) * (np.dot(X.T, (X.dot(w) - y[:, np.newaxis])) + beta * w)
        # for Lp w = w - (alpha / m) * (np.dot(X.T, (X.dot(w) - y[:, np.newaxis])) + beta * ((1/math.sqrt(w[0])) + (1/math.sqrt(w[1])))
        w = w - (alpha / m) * (np.dot(X.T, (X.dot(w) - y[:, np.newaxis])) + beta )
        #error_list.append(sumerrors(X, y))
        x = w.tolist()
        w0.append(x[0][0])
        w1.append(x[1][0])
        w2.append(x[2][0])




#Function for predicting the class label based on the w calculated using gradient descent using L2 norm
def predict(X):
    global w
    global mean_y

    Xn = np.ndarray.copy(X)

    Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

    return Xn.dot(w) + mean_y

def sumerrors(X, y):
    global w
    global mean_y
    sum_error = 0.0

    Xn = np.ndarray.copy(X)
    #
    Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

    error = (Xn.dot(w) + mean_y) - y[:, np.newaxis]
    error = error * error

    sum_error = np.sum(error)
    return sum_error

if __name__ == '__main__':
    data = pd.read_csv('Dataset.csv', header = None)
    y = data[3]
    X = data.drop([0,3], axis = 1)
    X = X.values
    y = y.values
    beta = 0.5 #this is lambda(regularization constant)

    #Fitting and predicting the labels based on prediction of the model implemented using L2 norm
    fit(X,y, beta)
    out = predict(X)

    #Varying the value of beta with the error encountered by prediction
    xAxis = []
    yAxis = []

    betaAxis = np.asarray(xAxis)
    errorAxis = np.asarray(yAxis)

    errorAxis = np.asarray(error_list)
    w1Axis = np.asarray(w1)
    w2Axis = np.asarray(w2)

    plotX, plotY = np.meshgrid(np.linspace(-5,5, 100), np.linspace(-5, 5, 100))
    for i in range(100):
        tempList = []
        for j in range(100):
            temp = []
            temp.append(plotX[i][j])
            temp.append(plotY[i][j])
            tempNp = np.asarray(temp)
            tempList.append(cost(X, y, tempNp, beta))
        error_list.append(tempList)

    errorNp = np.asarray(error_list)
    plt.contour(plotX, plotY, errorNp)
    plt.plot(w2Axis,w1Axis,color = 'g', alpha = 0.8)

    plt.legend()
    plt.tight_layout()
    plt.show()
