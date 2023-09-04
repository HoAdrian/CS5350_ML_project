import numpy as np
import csv
import argparse
import copy
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("../utils")
from data_util import *
from perceptron import *

np.random.seed(2021)

def split_folds(data, num_folds=5):
    np.random.shuffle(data)
    num_data = len(data)
    fraction = num_data//num_folds
    folds = []
    for i in range(num_folds-1):
        fold = data[i*fraction : (i+1)*fraction]
        folds.append(fold)
    folds.append(data[(num_folds-1)*fraction :])
    return folds
    
    
def cross_validation(data, num_folds=5):
    print("================CROSS VALIDATION================")
    num_features = data.shape[1] - 1
    folds = split_folds(data, num_folds)
    # hyperparameters
    learn_rates = [1,0.1,0.01]
    margins = [1,0.1,0.01]
    # accuracies across folds for each param
    accuracy_params = {}
    
    for margin in margins:
        for learn_rate in learn_rates:
            # accuracies across different validation folds
            accuracy_folds = []
            for valid_fold_idx in range(num_folds):
                # get training data
                train_fold_idxs = [i for i in range(num_folds)]
                train_fold_idxs.remove(valid_fold_idx)
                train_datas = [folds[i] for i in train_fold_idxs]
                train_data_combine = np.concatenate(train_datas, axis=0)
                
                # train and evaluate accuracy
                init_weights = np.random.uniform(low=-0.01, high=0.01, size=(num_features, ))
                perceptron = AveragePerceptron(init_weights)
                perceptron.train(train_data_combine, dev_data=train_data, lr=learn_rate, margin=margin, num_epochs=10, lr_decay=True, verbose=False)
                valid_fold_data = folds[valid_fold_idx]
                accuracy = perceptron.accuracy(valid_fold_data)
                accuracy_folds.append(accuracy)
            
            # compute statistical measures across folds for the hyperparameter  
            accuracy_params[(margin,learn_rate)] = accuracy_folds  
        
    # printing the result
    for margin in margins:
        for lr in learn_rates:
            avg_accuracy = sum(accuracy_params[(margin,lr)])/len(accuracy_params[(margin,lr)])
            print(f"margin: {margin}\tlearning rate: {lr}\tmean accuracy: {avg_accuracy}")
    
    # print the best hyperparameter based on mean accuracy
    best_hyperparam = max(accuracy_params.keys(), key=lambda x: sum(accuracy_params[x])/len(accuracy_params[x]))
    best_margin, best_learn_rate = best_hyperparam
    print(f"best margin = {best_margin}")
    print(f"best learning rate = {best_learn_rate}")
    print(f"mean accuracy of best param: {sum(accuracy_params[best_hyperparam])/len(accuracy_params[best_hyperparam])}")
    print("===============================================")
    return best_hyperparam
    
    
if __name__ == "__main__":
    feature_type = "roberta"
    print(f"##################### feature type: {feature_type} #######################")
    
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--traindp', default=f"../data/{feature_type}/{feature_type}.train.csv", type=str, help="training data path")
    parser.add_argument('--testdp', default=f"../data/{feature_type}/{feature_type}.test.csv", type=str, help="testing data path")
    parser.add_argument('--evaldp', default=f"../data/{feature_type}/{feature_type}.eval.anon.csv", type=str, help="evaluation data path for submission")
    args = parser.parse_args()
    train_data_path = args.traindp
    test_data_path = args.testdp
    eval_data_path = args.evaldp
    
    train_data = load_csv_data_perceptron(train_data_path)
    test_data = load_csv_data_perceptron(test_data_path)
    
    ################ statistics of labels ##################
    # train
    print("\n")
    print("============= train data statistics===============")
    print(label_counts_dict(train_data[:,-1]))
    # test
    print("============= test data statistics===============")
    print(label_counts_dict(test_data[:,-1]))
    print("\n")
    
    # print("num_train_data: ", len(train_data))
    # folds = split_folds(train_data, num_folds=5)
    # print(f"num_folds: {len(folds)}")
    # print(f"length of each folds: {len(folds[0])}, {len(folds[1])}, {len(folds[2])}, {len(folds[3])}, {len(folds[4])}")
    
    best_margin, best_lr = cross_validation(train_data, num_folds=5)
    
    # print("================train on best hyperparameters================")
    # instances, labels = get_examples(train_data)
    # np.random.seed(2021)
    # init_weights = np.random.uniform(low=-0.01, high=0.01, size=(instances.shape[1], ))
    # perceptron = AveragePerceptron(init_weights)
    # perceptron.train(train_data, dev_data=train_data, lr=best_lr, margin=best_margin, num_epochs=20, lr_decay=True, verbose=True)
    # print(f"final test accuracy: {perceptron.accuracy(test_data)}")