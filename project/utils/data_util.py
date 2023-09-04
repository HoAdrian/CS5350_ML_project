import csv
import numpy as np
import argparse

def label_counts_dict(labels):
    '''
    returns a dictionary mapping each label to number of its occurence
    '''
    label_counts_dict = {}
    for label in labels:
        label_counts_dict[label] = label_counts_dict.get(label, 0) + 1
    return label_counts_dict

def get_most_common_label(label_counts_dict):
    '''
    returns the most common label from a dictionary mapping each label to number of its occurence
    '''
    return max(label_counts_dict.keys(), key=lambda x: label_counts_dict[x])

def load_csv_data(data_path, preppend_one=True):
    '''
    load csv data into a np array of type float, excluding the column names (first row)
    '''
    data = []
    with open(data_path, mode ='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            data.append(line)
    data.pop(0)
    data = np.array(data)
    data = data.astype(float)
    # preppend 1 to the end of each feature vector (bias trick)
    if preppend_one:
        one = np.ones((len(data), 1), dtype=float)
        data = np.concatenate((one, data), axis=1)
    return data

def load_csv_data_decision_tree(data_path):
    data = load_csv_data(data_path, preppend_one=False)
    return data

def load_csv_data_perceptron(data_path):
    '''
    load csv data with converting label 0 to label -1 and preppending 1 to each feature vector
    (Assume label_idx is -1)
    '''
    data = load_csv_data(data_path)
    
    # make label value 0 to be -1
    data_labels = data[:, -1]
    neg_labels = np.array([-1 for i in range(len(data))])
    processed_labels = np.where(data_labels==0, neg_labels, data_labels)
    data[:,-1] = processed_labels
    
    return data

def load_eval_ids(file_path):
    data = []
    with open(file_path, mode ='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            data.append(int(line[0]))
        return data


def write_csv_row(file_path, row):
    '''
    fill out a row for the csv file
    '''
    with open(file_path, 'a', newline="") as csvfile: 
    # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(row)
        
def write_csv_rows(file_path, rows):
    '''
    fill out multiple rows for the csv file
    '''
    with open(file_path, 'a', newline="") as csvfile: 
    # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerows(rows) 


if __name__ == "__main__":
    feature_type = "spacy-embeddings"
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--traindp', default=f"../data/{feature_type}/{feature_type}.train.csv", type=str, help="training data path")
    parser.add_argument('--testdp', default=f"../data/{feature_type}/{feature_type}.test.csv", type=str, help="testing data path")
    parser.add_argument('--evaldp', default=f"../data/{feature_type}/{feature_type}.eval.anon.csv", type=str, help="evaluation data path for submission")
    args = parser.parse_args()
    train_data_path = args.traindp
    test_data_path = args.testdp
    eval_data_path = args.evaldp
    
    # data = load_csv_data(train_data_path)
    # print(data)
    
    data = load_csv_data_perceptron(train_data_path)
    print(data)