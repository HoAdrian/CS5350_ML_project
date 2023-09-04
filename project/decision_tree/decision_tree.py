import numpy as np
import csv
import argparse
import copy
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("../utils")
import math
from data_util import *

'''
Each feature feat in the node will be represented as
(feature index, a). where a is a threshold for testing. The feature has value True iff
the value x we are testing satisfies x>=a.
'''


def get_feats(data):
        
    features = []

    sort_data = list(copy.deepcopy(data))
    for j in range(data.shape[1]):
        sort_data.sort(key=lambda x:x[j])
        current_label = sort_data[0][-1]
        for i in range(data.shape[0]):
            label = sort_data[i][-1]
            if label!=current_label:
                features.append((j, sort_data[i][j]))
                
    return features

def get_feats_to_vals(data):
    feats = get_feats(data)
    feat_vals = [{True, False} for i in feats]
    feats_to_vals = {feats[i]:feat_vals[i] for i in range(len(feats))}
    return feats_to_vals

def get_data_subset(data, feat, value):
    '''
    return a subset of data where each instance's specified feature satisfies the threshold test
    '''
    idx, threshold = feat
    if value:
        mask = (data[:, idx]>=threshold)
        data_v = data[mask, :]
    else:
        mask = (data[:, idx]<threshold)
        data_v = data[mask, :]
    return data_v

def entropy(labels):
    '''
    calculate the entropy of a set of examples according to the list of labels belonging to the dataset
    '''
    label_counts = label_counts_dict(labels)
    entropy = 0
    total_size = len(labels)
    for _, count in label_counts.items():
        prob = count/total_size
        entropy -= prob*math.log(prob, 2)
    return entropy

def expected_entropy(data, feats_to_vals, feat, label_idx):
    '''
    return the expected entropy of taking a path from the feature on the decision tree
    '''
    expected_entropy = 0
    feat_values = feats_to_vals[feat]
    for v in feat_values:
        data_v = get_data_subset(data, feat, v)
        labels_v = data_v[:,label_idx]
        expected_entropy += len(data_v)/len(data)*entropy(labels_v)
    return expected_entropy

def info_gain(data, feats_to_vals, feat, label_idx):
    labels = data[:,label_idx]
    info_gain = entropy(labels)-expected_entropy(data, feats_to_vals, feat, label_idx)
    return info_gain

def get_best_feature(data, feats_to_vals, label_idx):
    best_feat = 0
    labels = data[:,label_idx]
    best_info_gain = -1
    for feat in feats_to_vals.keys():
        info_gain = entropy(labels)-expected_entropy(data, feats_to_vals, feat, label_idx)
        if info_gain > best_info_gain:
            best_feat = feat
            best_info_gain = info_gain
    return best_feat
        
class Node:
    '''
    represents a node in the decision tree
    '''
    def __init__(self, feature, label=None):
        self.feature = feature
        self.children = {}
        self.label = label
        
    def add_child(self, feature_val, node):
        '''
        add the root node of a subtree corresponding to a feature value branch
        '''
        self.children[feature_val] = node
        
    def is_leaf(self):
        return self.label!=None

def numerical_ID3(data, feats_to_vals, info_gain_low = 0, label_idx=-1):
    '''
    Learn a decision tree from data (can be continuous values)
    - each row in data is an instance [feat0, feat1,..., featm, label]
    - label_idx is the index of the label in each instance
    - features are represented by their index in each instance
    - feat_to_vals is the mapping from each feature to its set of values
    '''
    labels = data[:,label_idx]
    label_counts = label_counts_dict(labels)
    if len(label_counts) == 1 or len(feats_to_vals)==0:
        # all labels are the same, or no more featrues to consider
        return Node(feature=None, label=[label for label in label_counts.keys()][0])
    
    best_feature = get_best_feature(data, feats_to_vals, label_idx)
    print(len(feats_to_vals))
    root_node = Node(feature=best_feature)
    for v in feats_to_vals[best_feature]:
        data_v = get_data_subset(data, best_feature, v)
        if len(data_v)==0:
            root_node.add_child(v, Node(feature=None, label=get_most_common_label(label_counts)))
        elif info_gain(data, feats_to_vals, best_feature, label_idx)<=info_gain_low:
            root_node.add_child(v, Node(feature=None, label=get_most_common_label(label_counts)))
        else:
            new_feats_to_val = {feat:vals for feat, vals in feats_to_vals.items()}
            new_feats_to_val.pop(best_feature)
            child_node = numerical_ID3(data_v, new_feats_to_val, label_idx)
            root_node.add_child(v, child_node)
    
    return root_node

def predict(root_node, feature_vector):
    '''
    predict the label of the given feature vector by using the decision tree rooted at root_node
    '''
    current_node = root_node
    while(not current_node.is_leaf()):
        feature = current_node.feature
        idx, threshold = feature
        test_value = feature_vector[idx]
        
        branch = None
        
        if test_value >= threshold:
            branch = True
        else:
            branch = False
                
        current_node = current_node.children[branch]
        
    return current_node.label

def get_height(node):
    '''
    returns the height of the tree rooted at the given node.
    i.e. returns the depth of the deepest node from the given node
    '''
    if node.is_leaf():
        return 0
    subtree_heights = [get_height(child) for child in node.children.values()]
    return max(subtree_heights) + 1
        
def get_accuracy(data, root_node, label_idx=-1):
    accuracy = 0
    for row in data:
        label = row[label_idx]
        vec = row[:label_idx]
        predicted_label = predict(root_node, vec)
        accuracy += (predicted_label==label)
        #print(f"predicted{predicted_label}\tactual:{label}")
    accuracy /= len(data)
    return accuracy


if __name__ == "__main__":
    feature_type = "tfidf"
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
    train_data = load_csv_data_decision_tree(train_data_path)
    print("loaded train data")
    test_data = load_csv_data_decision_tree(test_data_path)
    print("loaded test data")
    feats_to_vals = get_feats_to_vals(train_data[:, :])
    print("got features")
    print(len(feats_to_vals))
    root_node = numerical_ID3(train_data[:2000, :], feats_to_vals, label_idx=-1)
    print(f"test acc: {get_accuracy(test_data, root_node, label_idx=-1)}")
    
    os.makedirs("submission", exist_ok=True)
    submission_path = f"submission/submit_{feature_type}.csv"
    
    eval_ids = load_eval_ids(eval_ids_path)
    eval_data = load_csv_data_decision_tree(eval_data_path)
    eval_data = eval_data[:, :-1]
    
    write_csv_row(submission_path, ["example_id", "label"])
    pairs = []
    for id in eval_ids:
        out = predict(root_node, eval_data[id])
        pairs.append([id, out])
    write_csv_rows(submission_path, pairs)
        
        