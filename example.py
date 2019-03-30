import random
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas
from tqdm import tqdm
# from tut
import ujson as json
import node2vec
import networkx as nx
from gensim.models import Word2Vec
import logging
import random
import numpy as np
from sklearn.metrics import roc_auc_score

# from tut
def divide_data(input_list, group_number):
    local_division = len(input_list) / float(group_number)
    random.shuffle(input_list)
    return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
            range(group_number)]

def get_G_from_edges(edges):
    edge_dict = dict()
    # calculate the count for all the edges
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.DiGraph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        # add edges to the graph
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        # add weights for all the edges
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight
    return tmp_G

def randomly_choose_false_edges(nodes, true_edges):
    tmp_list = list()
    all_edges = list()
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            all_edges.append((i, j))
    random.shuffle(all_edges)
    for edge in all_edges:
        if edge[0] == edge[1]:
            continue
        if (nodes[edge[0]], nodes[edge[1]]) not in true_edges and (nodes[edge[1]], nodes[edge[0]]) not in true_edges:
            tmp_list.append((nodes[edge[0]], nodes[edge[1]]))
    return tmp_list


def get_neighbourhood_score(local_model, node1, node2):
    # Provide the plausibility score for a pair of nodes based on your own model.
    # from tut
    try:
        vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return random.random()


def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)


if __name__ == "__main__":
    directed = True
    p = 1
    q = 1
    num_walks = 10
    walk_length = 80
    dimension = 128
    window_size = 50
    num_workers = 8
    iterations = 20
    # number_of_groups = 2

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    # Start to load the train data

    train_edges = list()
    raw_train_data = pandas.read_csv('train.csv')
    for i, record in raw_train_data.iterrows():
        train_edges.append((str(record['head']), str(record['tail'])))
    train_edges = list(set(train_edges))
    # from tut
    # edges_by_group = divide_data(train_edges, number_of_groups)

    print('finish loading the train data.')

    # Start to load the valid/test data

    valid_positive_edges = list()
    valid_negative_edges = list()
    raw_valid_data = pandas.read_csv('valid.csv')
    for i, record in raw_valid_data.iterrows():
        if record['label']:
            valid_positive_edges.append((str(record['head']), str(record['tail'])))
        else:
            valid_negative_edges.append((str(record['head']), str(record['tail'])))
    valid_positive_edges = list(set(valid_positive_edges))
    valid_negative_edges = list(set(valid_negative_edges))

    # Start to load the test data
    test_edges = list()
    raw_test_data = pandas.read_csv('test.csv')
    for i, record in raw_test_data.iterrows():
        test_edges.append((str(int(record['head'])), str(int(record['tail']))))
    test_edges = list(set(test_edges))
    print( test_edges)

    print('finish loading the valid/test data.')
    print("train: ",len(train_edges),"test: ",len(test_edges),"difference: ",len(set(train_edges).difference(set(test_edges))))
    print("train: ",len(train_edges),"valid: ",len(valid_positive_edges+valid_negative_edges),"difference: ",len(set(train_edges).difference(set(valid_positive_edges+valid_negative_edges))))
    print("test: ",len(test_edges),"valid: ",len(valid_positive_edges+valid_negative_edges),"difference: ",len(set(test_edges).difference(set(valid_positive_edges+valid_negative_edges))))
    cycle=0
    while True:
        # write code to train the model here
        G = node2vec.Graph(get_G_from_edges(train_edges), directed, p, q)
        # Calculate the probability for the random walk process
        G.preprocess_transition_probs()
        # Conduct the random walk process
        walks = G.simulate_walks(num_walks, walk_length)
        # Train the node embeddings with gensim word2vec package
        model = Word2Vec(walks, size=dimension, window=window_size, min_count=0, sg=1, workers=num_workers, iter=iterations)
        # Save the resulted embeddings (you can use any format you like)
        resulted_embeddings = dict()
        # for i, w in enumerate(model.wv.index2word):
        #     resulted_embeddings[w] = model.wv.syn0[i]
        # replace 'your_model' with your own model and use the provided evaluation code to evaluate.
        tmp_AUC_score = get_AUC(model, valid_positive_edges, valid_negative_edges)

        print('tmp_accuracy:', tmp_AUC_score)

        print('end')

        # predicting
        prediction_list = list()
        label_list=list()
        for edge in test_edges:
                tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
                prediction_list.append(tmp_score)
                label_list.append(("True","False")[tmp_score<=0.5])
        df = raw_test_data
        df["score"] = prediction_list
        df["label"]= label_list
        filename="test_"+str(cycle)+".csv"
        df.to_csv(filename, index=False)
        with open("acc_record_for_test.txt", "a") as myfile:
            myfile.write("acc for "+filename+" is "+str(tmp_AUC_score)+"\n")
        cycle+=1