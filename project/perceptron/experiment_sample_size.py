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
    
    train_accuracies = []
    test_accuracies = []
    np.random.seed(2021)
    for size in range(500, len(train_data), 500):
        init_weights = np.random.uniform(low=-0.01, high=0.01, size=(instances.shape[1], ))
        perceptron = AveragePerceptron(init_weights)
        np.random.shuffle(train_data)
        train_data_subset = train_data[:size]
        perceptron.train(train_data_subset, dev_data=test_data, lr=0.1, margin=0.01, num_epochs=30, lr_decay=True, verbose=False)
        
        test_acc = perceptron.accuracy(test_data)
        train_acc = perceptron.accuracy(train_data)
        print(f"============ sample size: {size} =============")
        print(f"final test acc: {test_acc}")
        print(f"final train accuracy: {train_acc}")
        print("=================================================")
        print(perceptron.accuracy_batch(train_data))
        train_accuracies.append(train_acc)
        test_accuracies.append( test_acc)
        
    sizes = [size for size in range(500, len(train_data), 500)]
    fig, ax = plt.subplots()
    ax.plot(sizes, train_accuracies, label="train accuracy")
    ax.plot(sizes, test_accuracies, label="test accuracy")
    ax.legend()
    ax.set_title("sample_size_effect")
    ax.set_xlabel("training set size")
    ax.set_ylabel("accuracies")
    os.makedirs(f"./figures", exist_ok=True)
    fig.savefig(f"./figures/sample_size_effect.png")
        
    
        
        
        
    # plot_trainTest_curves(perceptron.train_accuracies, perceptron.dev_accuracies, f"{feature_type}_traintest")
    
    # #plot_learning_curve(perceptron.dev_accuracies, feature_type)
    
    # os.makedirs("submission", exist_ok=True)
    # submission_path = f"submission/submit_{feature_type}30epoch.csv"
    
    # eval_ids = load_eval_ids(eval_ids_path)
    # eval_data = load_csv_data_perceptron(eval_data_path)
    # eval_data = eval_data[:, :-1]
    
    # write_csv_row(submission_path, ["example_id", "label"])
    # pairs = []
    # for id in eval_ids:
    #     out = perceptron.predict(eval_data[id])
    #     if out==-1:
    #         out=0
    #     pairs.append([id, out])
    # write_csv_rows(submission_path, pairs)
        
        