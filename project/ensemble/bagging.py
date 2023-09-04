import numpy as np
import csv
import argparse
import copy
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("../utils")
sys.path.append("../perceptron")
from data_util import *
from perceptron import *

class PerceptronEnsemble:
    def __init__(self, num_learner):
        self.learners = []
        self.num_learner = num_learner

    def get_weak_learners(self, train_data, test_data):
        learners = []
        for i in range(self.num_learner):
            print(f"########## learner {i} ##########")
            np.random.shuffle(train_data)
            num_data = len(train_data)//2
            train_data_subset = train_data[:num_data]
            instances, labels = get_examples(train_data)
            init_weights = np.random.uniform(low=-0.01, high=0.01, size=(instances.shape[1], ))
            perceptron = AveragePerceptron(init_weights)
            perceptron.train(train_data_subset, dev_data=test_data, lr=0.1, margin=0.01, num_epochs=30, lr_decay=True, verbose=False)
            print(f"train accuracy: {perceptron.accuracy(train_data)}")
            print(f"test accuracy: {perceptron.accuracy(test_data)}")
            print("##################################")
            learners.append(perceptron)
        self.learners = learners
        return learners

    def predict(self, x):
        preds = []
        for learner in self.learners:
            out = learner.predict(x)
            preds.append(out)
        count = {}
        for pred in preds:
            count[pred] = count.get(pred, 0) + 1
            
        return max(count.keys(), key=lambda x: count[x])

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
    #print(len(train_data))
    np.random.seed(2021)
    ensemble = PerceptronEnsemble(num_learner=40)
    ensemble.get_weak_learners(train_data, test_data)
    print("train accuracy: ", ensemble.accuracy(train_data))
    print("test accuracy: ", ensemble.accuracy(test_data))
    
    
    # os.makedirs("submission", exist_ok=True)
    # submission_path = f"submission/submit_bagging_{feature_type}.csv"
    
    # eval_ids = load_eval_ids(eval_ids_path)
    # eval_data = load_csv_data_perceptron(eval_data_path)
    # eval_data = eval_data[:, :-1]
    
    # write_csv_row(submission_path, ["example_id", "label"])
    # pairs = []
    # for id in eval_ids:
    #     out = ensemble.predict(eval_data[id])
    #     if out==-1:
    #         out=0
    #     pairs.append([id, out])
    # write_csv_rows(submission_path, pairs)
        
        