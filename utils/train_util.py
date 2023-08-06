import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader

from models.deep_neural_network_model import *
from models.machine_learning_models import *
from utils.dataset_util import get_dataset


def train_machine_learning_model(path):
    con_data_set = get_dataset(path, 'con', 3)
    dep_data_set = get_dataset(path, 'dep', 3)
    con_x_train = con_data_set[0]
    con_y_train = con_data_set[1]
    dep_x_train = dep_data_set[0]
    dep_y_train = dep_data_set[1]
    model_name = 'con_decision_tree_model_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_decision_tree_model(con_x_train, con_y_train, '../runs/models/' + model_name + '.sav')
    model_name = 'dep_decision_tree_model_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_decision_tree_model(dep_x_train, dep_y_train, '../runs/models/' + model_name + '.sav')
    model_name = 'con_random_forest_model_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_random_forest_model(con_x_train, con_y_train, '../runs/models/' + model_name + '.sav')
    model_name = 'dep_random_forest_model_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_random_forest_model(dep_x_train, dep_y_train, '../runs/models/' + model_name + '.sav')
    model_name = 'con_svm_model_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_svm_model(con_x_train, con_y_train, '../runs/models/' + model_name + '.sav')
    model_name = 'dep_svm_model_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_svm_model(dep_x_train, dep_y_train, '../runs/models/' + model_name + '.sav')
    model_name = 'con_logistic_regression_model_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_logistic_regression_model(con_x_train, con_y_train, '../runs/models/' + model_name + '.sav')
    model_name = 'dep_logistic_regression_model_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_logistic_regression_model(dep_x_train, dep_y_train, '../runs/models/' + model_name + '.sav')


def dnn_train(train_loader, test_loader):
    model = DNN()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = 0.0
    best_model = None
    for run in range(100):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                indices = (labels == 1).nonzero(as_tuple=True)[0]

                if len(indices) == 0:
                    continue

                inputs = inputs[indices]
                labels = labels[indices]

                outputs = model(inputs.float())
                predicted = torch.round(outputs)
                correct += (predicted == labels).sum().item()
                total += len(indices)
            accuracy = 100 * correct / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model.state_dict()

    return [best_accuracy, best_model]


def train_dnn_model(path):
    con_data_set = get_dataset(path, 'con', 3)
    dep_data_set = get_dataset(path, 'dep', 3)
    con_x = con_data_set[0]
    con_y = [[x] for x in con_data_set[1]]
    dep_x = dep_data_set[0]
    dep_y = [[x] for x in dep_data_set[1]]
    con_x_train = con_x[:int(len(con_x) * 0.8)]
    con_y_train = con_y[:int(len(con_y) * 0.8)]
    con_x_test = con_x[int(len(con_x) * 0.8):]
    con_y_test = con_y[int(len(con_y) * 0.8):]
    dep_x_train = dep_x[:int(len(dep_x) * 0.8)]
    dep_y_train = dep_y[:int(len(dep_y) * 0.8)]
    dep_x_test = dep_x[int(len(dep_x) * 0.8):]
    dep_y_test = dep_y[int(len(dep_y) * 0.8):]

    train_dataset = TensorDataset(torch.tensor(con_x_train), torch.tensor(con_y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(con_x_test), torch.tensor(con_y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    result = dnn_train(train_loader, test_loader)
    torch.save(result[1], '../runs/models/con_dnn_model_' + str(result[0]) + '_'
               + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pt')

    train_dataset = TensorDataset(torch.tensor(dep_x_train), torch.tensor(dep_y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(dep_x_test), torch.tensor(dep_y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    result = dnn_train(train_loader, test_loader)
    torch.save(result[1], '../runs/models/dep_dnn_model_' + str(result[0]) + '_'
               + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pt')
