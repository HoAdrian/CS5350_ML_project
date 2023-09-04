import numpy as np
import csv
import argparse
import copy
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("../utils")
from data_util import *

import math

def get_examples(data):
    '''
    returns the matrix of feature vectors and a row of labels
    (assume label_idx=-1)
    '''
    labels = data[:, -1]
    instances = data[:, :-1]
    return instances, labels


class LogisticRegression:
    def __init__(self, init_weights):
        self.weights = copy.deepcopy(init_weights)
        self.dev_accuracies = []
        self.train_accuracies = []
        self.losses = []
        
    def train(self, train_data, dev_data, reg, lr, num_epochs=20, lr_decay=True, verbose=True):
        # lr is the learning rate
        init_lr = lr
        t = 0
        num_data = train_data.shape[0]
        k = 1/10000
        
        for epoch in range(num_epochs):
            np.random.shuffle(train_data)
            instances, labels = get_examples(train_data)
            for i in range(num_data):
                x = instances[i]
                y = labels[i]
                grad = -y*k/(1*k+np.exp(np.log(k)+y*np.matmul(self.weights, x)))*x + 2/reg*self.weights
                self.weights -= lr * grad
                
                # if np.matmul(self.weights, self.weights)>=500**2:
                #     self.weights /= np.matmul(self.weights, self.weights)**0.5

            t+=1
            if lr_decay:
                lr = init_lr/(1+t)
                
            logistic_loss = 0
            for i in range(num_data):
                x = instances[i]
                y = labels[i]
                logistic_loss += np.log(1+np.exp(-y*np.matmul(self.weights, x)))
            self.losses.append(logistic_loss)
            
            
            # save accuracies every epoch
            dev_accuracy = self.accuracy(dev_data)
            self.dev_accuracies.append(dev_accuracy)
            train_accuracy = self.accuracy(train_data)
            self.train_accuracies.append(train_accuracy)
            if verbose:
                print(f"epoch:{epoch}/{num_epochs}\ttrain accuracy:{train_accuracy}\tdev accuracy: {dev_accuracy}")
    
        return self.weights
    
    def predict(self, x):
        out = 1/(1+math.exp(-np.matmul(self.weights, x)))
        if out<0.5:
            return -1
        else:
            return 1
        
        
    def accuracy(self, data):
        instances, labels = get_examples(data)
        num_data = data.shape[0]
        accuracy = 0
        for i in range(num_data):
            x = instances[i]
            y = labels[i]
            y_pred = self.predict(x)
            if y_pred==y:
                accuracy+=1
        accuracy = accuracy/num_data
        return accuracy
    
    def predict_batch(self, X):
        #y = 1/(1+np.exp(-np.matmul(X, self.weights)))
        y = np.matmul(X, self.weights)
        mask_positive = (y>0).astype(int)
        mask_negative = (y<=0).astype(int)
        mask_negative = mask_negative*-1
        y = mask_positive + mask_negative
        return y
    
    def accuracy_batch(self, data):
        instances, labels = get_examples(data)
        num_data = data.shape[0]
        y = self.predict_batch(instances)
        return sum(y==labels)/num_data

def plot_trainTest_curves(train_accuracies, test_accuracies, title):
    epochs = [i for i in range(len(train_accuracies))]
    fig, ax = plt.subplots()
    ax.plot(epochs, train_accuracies, label="train accuracy")
    ax.plot(epochs, test_accuracies, label="test accuracy")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracies")
    os.makedirs(f"./figures", exist_ok=True)
    fig.savefig(f"./figures/{title}.png")
    
def plot_loss_curve(losses, title):
    time_steps = [i for i in range(len(losses))]
    fig, ax = plt.subplots()
    ax.plot(time_steps, losses, label="loss")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("losses")
    os.makedirs(f"./figures", exist_ok=True)
    fig.savefig(f"./figures/{title}.png")
    
    
    

if __name__ == "__main__":
    feature_type = "roberta"
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--traindp', default=f"../data/{feature_type}/{feature_type}.train.csv", type=str, help="training data path")
    parser.add_argument('--testdp', default=f"../data/{feature_type}/{feature_type}.test.csv", type=str, help="testing data path")
    parser.add_argument('--evaldp', default=f"../data/{feature_type}/{feature_type}.eval.anon.csv", type=str, help="evaluation data path for submission")
    parser.add_argument('--evalidsp', default=f"../data/eval.ids", type=str, help="path to the evaluation ids")
    
    args = parser.parse_args()
    train_data_path = args.traindp
    test_data_path = args.testdp
    eval_data_path = args.evaldp
    eval_ids_path = args.evalidsp
    
    print("================train on selected hyperparameters================")
    train_data = load_csv_data_perceptron(train_data_path)
    test_data = load_csv_data_perceptron(test_data_path)
    instances, labels = get_examples(train_data)
    np.random.seed(2021)
    init_weights = np.random.uniform(low=-0.01, high=0.01, size=(instances.shape[1], ))
    logistic = LogisticRegression(init_weights)
    logistic.train(train_data, dev_data=test_data, reg=5000, lr=0.1, num_epochs=40, lr_decay=True, verbose=True)
    print(f"final test acc: {logistic.accuracy(test_data)}")
    
    print(logistic.accuracy_batch(train_data))
    
    # plot_trainTest_curves(logistic.train_accuracies, logistic.dev_accuracies, f"{feature_type}_traintest")
    # plot_loss_curve(logistic.losses, f"{feature_type}_loss")
    # os.makedirs("submission", exist_ok=True)
    # submission_path = f"submission/submit_{feature_type}.csv"
    
    # eval_ids = load_eval_ids(eval_ids_path)
    # eval_data = load_csv_data_perceptron(eval_data_path)
    # eval_data = eval_data[:, :-1]
    
    # write_csv_row(submission_path, ["example_id", "label"])
    # pairs = []
    # for id in eval_ids:
    #     out = logistic.predict(eval_data[id])
    #     if out==-1:
    #         out=0
    #     pairs.append([id, out])
    # write_csv_rows(submission_path, pairs)
        
        