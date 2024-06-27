import numpy as np
import pandas as pd


train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv("./data/test.csv")


Y_train = train_data['label'].values
X_train = train_data.drop(columns=['label'])
X_train = X_train.values/255
X_test = test_data.values/255


def relu(x):
    return np.maximum(0, x)


def predict(X,W,b):
    Z1 = np.matmul(X, W[0]) + b[0]
    A2 = relu(Z1)
    A3 = np.matmul(A2, W[1])


    S = np.exp(A3)
    total = np.sum(S, axis=1).reshape(-1,1)
    S = S/total
    return S


def backprop(W,b,X,Y,alpha=1e-4):
    '''
    Step 1: explicit forward pass h(X;W,b)
    Step 2: backpropagation for dW and db
    '''
    K = 10
    N = X.shape[0]
   
    '''
    Step 1: feed forward
    '''
   
    #second (hidden) layer
    Z1 = np.matmul(X, W[0]) + b[0]
    A2 = relu(Z1)


    #third (output) layer
    A3 = np.matmul(A2, W[1])
   
    '''
    We apply a softmax function on the final layer to create a probability distribution
    over the 10 different digit cases.
    The use of exp allows the nn to make clearer predictions between cases.
    '''


    S = np.exp(A3)
    total = np.sum(S, axis=1).reshape(-1,1)
    S = S/total
   
    '''
    Step 2: Backpropagate
    '''
   
    Y = (Y[:, np.newaxis] == np.arange(K)).astype(int)
   
    delta2 = S - Y
    grad_W1 = np.matmul(A2.T, delta2)


    delta1 = np.matmul(delta2, W[1].T)*(Z1>0)
    grad_W0 = np.matmul(X.T, delta1)    
   
    dW = [grad_W0/N + alpha*W[0], grad_W1/N + alpha*W[1]]
    db = [np.mean(delta1, axis=0)]


    return dW, db


alpha = 1e-6
iterations = 1000
hidden_neurons = 256
first_layer_neurons = X_train.shape[1]
K = 10


#parameters for gradient descent
eta = 5e-1
gamma = 0.99
eps = 1e-3


# initialization
np.random
W = [1e-1*np.random.randn(first_layer_neurons, hidden_neurons), 1e-1*np.random.randn(hidden_neurons, K)]
b = [np.random.randn(hidden_neurons)]


gW0 = gW1 = gb0 = 1


for i in range(iterations):
    dW, db = backprop(W,b,X_train,Y_train,alpha)
   
    gW0 = gamma*gW0 + (1-gamma)*np.sum(dW[0]**2)
    etaW0 = eta/np.sqrt(gW0 + eps)
    W[0] -= etaW0 * dW[0]
   
    gW1 = gamma*gW1 + (1-gamma)*np.sum(dW[1]**2)
    etaW1 = eta/np.sqrt(gW1 + eps)
    W[1] -= etaW1 * dW[1]
   
    gb0 = gamma*gb0 + (1-gamma)*np.sum(db[0]**2)
    etab0 = eta/np.sqrt(gb0 + eps)
    b[0] -= etab0 * db[0]


Y_predict = predict(X_train,W,b)


correct_predictions = 0
total_predictions = Y_predict.shape[0]


for j in range(total_predictions):
    if np.argmax(Y_predict[j]) == Y_train[j]:
        correct_predictions += 1


accuracy = correct_predictions / total_predictions


print("Model Accuracy {:.4%}".format(accuracy))



