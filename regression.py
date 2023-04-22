import numpy as np
from matplotlib import pyplot as plt
import csv
import math
from numpy.linalg import inv
import random


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    i = 0
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            i+=1
            if i==1:
                continue
            dataset.append(row[1:])
    dataset = np.array(dataset, dtype = float)
    return dataset

def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    sum = 0.0
    count = 0
    for entry in dataset[:, col]:
        sum += entry
        count+=1
    print(count)
    mean = sum/count
    
    squaredSum = 0
    for number in dataset[:, col]:
        s = math.pow((number-mean), 2)
        squaredSum += s
    coef = (1/(count- 1))
    standardDev = math.sqrt(coef * squaredSum)
    mean = round(mean, 2)
    print("%.2f"%mean)
    standardDev = round(standardDev, 2)
    print("%.2f"%standardDev)


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    #x = np.zeros(shape = (252, 2))
    
    #go through all desired data points in each of 252 rows

    
    x = []
    for col in cols:
        x.append(dataset[:, col])
    x = np.array(x, dtype = float)
    x = np.transpose(x)

    totalSum = 0
    n = 0
    for i in range(len(x)):
        n+=1
        currSum = 0
        for beta in range(len(betas)):
            if beta == 0:
                currSum += betas[beta]
            else:
                currSum += betas[beta] * x[i, beta - 1]
        currSum = currSum - float(dataset[i, 0])
        currSum = currSum*currSum
        totalSum +=currSum
    mse = totalSum/n
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    gradDescent = np.empty(shape = len(betas))
    x = []
    for col in cols:
        x.append(dataset[:, col])
    x = np.array(x, dtype = float)
    x = np.transpose(x)

    n = 0
    count = 0
    for b in range(len(betas)): #for each element in array beta
        totalSum = 0
        for i in range(len(x)): # for each i in sum i->n
            n+=1 #keeping track of total
            currSum = 0
            for beta in range(len(betas)):
                if beta == 0:
                    currSum += betas[beta]
                else:
                    currSum += betas[beta] * x[i, beta - 1]
            count+=1
            currSum = currSum - float(dataset[i, 0])
            if b == 0:
                currSum = currSum
            else:
                currSum = currSum * x[i, b - 1]
            totalSum+=currSum
        totalSum = totalSum * (2/252)
        gradDescent[b] = totalSum
    return gradDescent


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    entries = np.zeros(shape = (T+1, len(betas)))
    mse = np.zeros(shape = (T+1))
    for time in range(T + 1):
        if time == 0:
            mse[0] = regression(dataset, cols, betas)
            for b in range(len(betas)):
                entries[0][b] = betas[b]
            continue
        for beta in range(len(betas)):
            grad = gradient_descent(dataset, cols, entries[time - 1])
            newBeta = entries[time - 1][beta] - eta*grad[beta]
            entries[time][beta] = newBeta
        mse[time] = regression(dataset, cols, entries[time])
    #printing
    for time in range(T):
        print(time+1, end = " ")
        toPrint1 = (round(mse[time + 1], 2))
        print("%.2f"%toPrint1, end = " ")
        for entry in entries[time + 1]:
            toPrint2 = round(entry,2)
            print("%.2f"%toPrint2, end = " ")
        print("")
            

    

def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    
    x = []
    ones = []
    for i in range(len(dataset)):
        ones.append(1)
    x.append(ones)
    
    for col in cols:
        x.append(dataset[:, col])
    x = np.array(x, dtype = float)
    
    y = dataset[:, 0]
    y = np.array(y, dtype = float)
    transposeX = np.transpose(x)

    beta_1 = np.matmul(x, transposeX)
    beta_1 = inv(beta_1)
    beta_2 = np.matmul(beta_1, x)
    betas = np.matmul(beta_2, y)
    mse = regression(dataset, cols, betas)
    return (mse, *betas)

def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    formula = 0
    for f in range(len(betas)):
        if f == 0:
            continue
        if f == 1:
            formula += betas[1]
        else:
            formula += betas[f] * features[f - 2]
    return formula

def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linearDataset = np.zeros(shape = (len(X), 2))
    for i in range(len(X)):
        y = 0.0
        y+= betas[0] + betas[1] * X[i]
        y += np.random.normal(loc = 0.0, scale = sigma) #z term
        linearDataset[i][0] = y
    for i in range(len(X)):
        linearDataset[i][1] = X[i][0]

    quadraticDataset = np.zeros(shape = (len(X), 2))
    for i in range(len(X)):
        y = 0.0
        y += np.random.normal(loc = 0.0, scale = sigma) #z term
        y += alphas[0] + alphas[1] * (math.pow(X[i], 2))
        quadraticDataset[i][0] = y
    for i in range(len(X)):
        quadraticDataset[i][1] = X[i][0]
    
    returnTup = (linearDataset, quadraticDataset)

    return returnTup


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = np.zeros(shape = (1000, 1))
    for i in range(len(X)):
        X[i] = random.randint(-100, 100)
    X = np.array(X, dtype = float)
    betas = [float(np.random.uniform(low = 0, high = 10)), float(np.random.uniform(low = 0, high = 10))]
    alphas = [float(np.random.uniform(low = 0, high = 10)), float(np.random.uniform(low = 0, high = 10))]
    sigmas = [0.0010, 0.010, 0.1, 1, 10, 100, 1000, 10000, 100000]
    datasets = {}
    for sigma in range(len(sigmas)):
        datasets[sigmas[sigma]] = synthetic_datasets(betas, alphas, X, sigmas[sigma])
    mses = {}

    for data in datasets:
        d = np.array(datasets[data][0], dtype = float)
        betaInfoLinear = compute_betas(d, [1])
        d2 = np.array(datasets[data][1], dtype = float)
        betaInfoQuad = compute_betas(d2, [1])
        mses[data] = [betaInfoLinear[0], betaInfoQuad[0]]
    xVals = []
    yValsLin = []
    yValsQuad = []
    for mse in mses:
        xVals.append(mse)
        yValsLin.append(mses[mse][0])
        yValsQuad.append(mses[mse][1])
    
    plt.xlabel("Sigma")
    plt.ylabel("MSE")
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(xVals, yValsLin)
    plt.scatter(xVals, yValsQuad)
    plt.plot(xVals, yValsLin)
    plt.plot(xVals, yValsQuad)
    plt.legend(['Linear', 'Quadratic'], loc = 'lower right')
    plt.savefig("mse.pdf")
    plt.show()

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
