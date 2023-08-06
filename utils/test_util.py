import datetime

import joblib
import torch
from torch.utils.data import DataLoader


def test_machine_learning_model(dataset, model_path):
    model = joblib.load(model_path)
    vectors = dataset[0]
    sentences = dataset[2]
    originals = dataset[3]
    file = open('./runs/logs/' + model_path.split('/')[-1] + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H-%M-%S') + '.log', 'w+')

    predict = model.predict(vectors)
    file.write('Model: ' + model_path + '\n')
    error_count = 1
    for i in range(len(predict)):
        if predict[i] == 0:
            file.write('Error ' + str(error_count) + ': ' + 'Metamorphic sentence -> ' + str(sentences[i]) +
                       'Metamorphic sentence -> ' + str(originals[i]) + '\n')
            error_count += 1
    file.write('Error count: ' + str(error_count - 1) + '\n')
    file.close()


def test_dnn_model(dataset, model_path):
    vectors = dataset[0]
    sentences = dataset[2]
    originals = dataset[3]
    model = torch.load(model_path)
    model.eval()
    data_loader = DataLoader(torch.tensor(vectors), batch_size=1, shuffle=True)
    predict = []
    with torch.no_grad():
        for inputs in data_loader:
            output = model(inputs.float())
            predict.append(round(output.item()))

    file = open('./runs/logs/' + model_path.split('/')[-1] + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H-%M-%S') + '.log', 'w+')
    file.write('Model: ' + model_path + '\n')
    error_count = 1
    for i in range(len(predict)):
        if predict[i] == 0:
            file.write('Error ' + str(error_count) + ': ' + 'Metamorphic sentence -> ' + str(sentences[i]) +
                       'Original sentence -> ' + str(originals[i]) + '\n')
            error_count += 1
    file.write('Error count: ' + str(error_count - 1) + '\n')
    file.close()
    torch.save(model, model_path)
