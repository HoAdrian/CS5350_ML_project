import numpy as np
import math
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


class Adaboost:
    def __init__(self, train_data, num_learner):
        self.learners = []
        self.num_learner = num_learner
        self.train_data = train_data
        self.D_t = np.array([1/len(train_data) for i in range(len(train_data))]).astype(np.double)
        self.alphas = []
        self.hypotheses = []
        self.train_accuracies = []
        self.dev_accuracies = []
        #assert(np.sum(self.D_t)==1)

    def get_weak_learners(self, train_data, test_data):
        learners = []
        for i in range(self.num_learner):
            np.random.shuffle(train_data)
            num_data = len(train_data)//self.num_learner
            train_data_subset = train_data[:num_data]
            instances, _ = get_examples(train_data)
            init_weights = np.random.uniform(low=-0.01, high=0.01, size=(instances.shape[1], ))
            perceptron = AveragePerceptron(init_weights)
            perceptron.train(train_data_subset, dev_data=test_data, lr=0.1, margin=0.01, num_epochs=30, lr_decay=True, verbose=False)
            print(f"########## weak learner {i} ##########")
            print(f"train accuracy: {perceptron.accuracy(train_data)}")
            print(f"test accuracy: {perceptron.accuracy(test_data)}")
            print("##################################")
            learners.append(perceptron)
        self.learners = learners
        return learners
    
    def weighted_error(self, learner):
        instances, labels = get_examples(self.train_data)
        y_pred = learner.predict_batch(instances)
        error_arr = (labels != y_pred).astype(int)
        weighted_arr = np.multiply(error_arr.astype(np.double), self.D_t)
        return np.sum(weighted_arr)
    
    def adaboost(self, num_iter, dev_data, verbose=True):
        print("============= adaboost ==============")
        for t in range(num_iter):
            candidates = {}
            for i, learner in enumerate(self.learners):
                weighted_error = self.weighted_error(learner)
                if weighted_error<0.5:
                    candidates[weighted_error] = (i,learner)
            if len(candidates)!=0:
                epsilon = min(candidates.keys())
                learner_id, hypothesis = candidates[epsilon]
            # else:
            #     hypothesis = learner[0]
            #     epsilon = self.weighted_error(hypothesis)
            alpha = 0.5*math.log((1-epsilon)/epsilon)
            self.alphas.append(alpha)
            self.hypotheses.append(hypothesis)
            
            instances, labels = get_examples(self.train_data)
            y_preds = hypothesis.predict_batch(instances)
            update = np.exp(-alpha*np.multiply(y_preds, labels))
            self.D_t = np.multiply(update, self.D_t)
            self.D_t = np.divide(self.D_t, np.sum(self.D_t)).astype(np.double)
            
            if verbose:
                print("########################################")
                print(f"iteration {t}: weighted error: {epsilon} alpha: {alpha}")
                print(f"hypothesis chosen: {learner_id}")
                print(f"predictions: {y_preds}")
                print(f"ground truths: {labels}")
                print(f"probabilities: {self.D_t}")
                print(f"sum of probability: {sum(self.D_t)}")
                print(f"dev accuracy: {self.accuracy(dev_data)}")
                print(f"train accuracy: {self.accuracy(self.train_data)}")
                print("########################################")
                
            self.train_accuracies.append(self.accuracy(self.train_data))
            self.dev_accuracies.append(self.accuracy(dev_data))
            
    def adaboost_online(self, num_iter, dev_data, verbose=True):
        print("============= adaboost online ==============")
        for t in range(num_iter):
            instances, labels = get_examples(self.train_data)
            epsilon = 1
            while epsilon >= 0.5:
                init_weights = np.random.uniform(low=-0.01, high=0.01, size=(instances.shape[1], ))
                new_train_data_idx = np.random.choice([i for i in range(self.train_data.shape[0])], size=self.train_data.shape[0], p=self.D_t)
                new_train_data = copy.deepcopy(self.train_data[new_train_data_idx, :])
                hypothesis= AveragePerceptron(init_weights)
                hypothesis.train(new_train_data, dev_data=test_data, lr=0.1, margin=0.01, num_epochs=40, lr_decay=True, verbose=False)
                epsilon = self.weighted_error(hypothesis)
                print("========== generating weak learner")
                print(f"weighted error: {epsilon}")
                print("=================================")
                #assert(epsilon<0.5)
            alpha = 0.5*math.log((1-epsilon)/epsilon)
            self.alphas.append(alpha)
            self.hypotheses.append(hypothesis)
            
            instances, labels = get_examples(self.train_data)
            y_preds = hypothesis.predict_batch(instances)
            update = np.exp(-alpha*np.multiply(y_preds, labels))
            self.D_t = np.multiply(update, self.D_t)
            self.D_t = np.divide(self.D_t, np.sum(self.D_t)).astype(np.double)
            
            if verbose:
                print("########################################")
                print(f"iteration {t}: weighted error: {epsilon} alpha: {alpha}")
                print(f"predictions: {y_preds}")
                print(f"ground truths: {labels}")
                print(f"probabilities: {self.D_t}")
                print(f"sum of probability: {sum(self.D_t)}")
                print(f"dev accuracy: {self.accuracy(dev_data)}")
                print(f"train accuracy: {self.accuracy(self.train_data)}")
                print("########################################")
                
            self.train_accuracies.append(self.accuracy(self.train_data))
            self.dev_accuracies.append(self.accuracy(dev_data))
            
    def predict(self, x):
        pred = 0
        for i,hypothesis in enumerate(self.hypotheses):
            pred += self.alphas[i]*hypothesis.predict(x)
            
        if pred >= 0:
            return 1
        else:
            return -1
        
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
    #np.random.seed(2021)
    
    ensemble = Adaboost(train_data, num_learner=40)
    # ensemble.get_weak_learners(train_data, test_data)
    #ensemble.adaboost(num_iter=100, dev_data=test_data)
    
    num_iter = 200
    ensemble.adaboost_online(num_iter=num_iter, dev_data=test_data)
    
    # plot_trainTest_curves(ensemble.train_accuracies, ensemble.dev_accuracies, f"Adaboost_{feature_type}_online_{num_iter}")
    
    # os.makedirs("submission", exist_ok=True)
    # submission_path = f"submission/submit_adaboost_online_{feature_type}_{num_iter}.csv"
    
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
        
        