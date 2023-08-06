import argparse
import os

from utils.dataset_util import get_dataset, get_dataset_for_label
from utils.test_util import test_machine_learning_model, test_dnn_model
from utils.train_util import train_machine_learning_model


def main():
    parser = argparse.ArgumentParser(description="StructureTester")
    parser.add_argument("--mode", type=str, help="StructureTester runs mode")
    args = parser.parse_args()
    mode = args.mode
    current_dir = os.getcwd()

    if mode == 'demo':
        print('Running on demo mode')
        con_dataset = get_dataset(current_dir + '/datasets/demo.xls', 'con', 3)
        dep_dataset = get_dataset(current_dir + '/datasets/demo.xls', 'dep', 3)
        con_test_models = ['con_decision_tree_demo.sav',
                           'con_random_forest_demo.sav',
                           'con_svm_demo.sav',
                           'con_logistic_regression_demo.sav']
        dep_test_models = ['dep_decision_tree_demo.sav',
                           'dep_random_forest_demo.sav',
                           'dep_svm_demo.sav',
                           'dep_logistic_regression_demo.sav']

        for model in con_test_models:
            test_machine_learning_model(con_dataset, current_dir + '/runs/models/' + model)
            print('Demo model complete: ' + model)
        for model in dep_test_models:
            test_machine_learning_model(dep_dataset, current_dir + '/runs/models/' + model)
        test_dnn_model(con_dataset, current_dir + '/runs/models/con_tree_dnn_demo.pt')
        print('Demo model complete: con_tree_dnn_demo.pt')
        test_dnn_model(dep_dataset, current_dir + '/runs/models/dep_tree_dnn_demo.pt')
        print('Demo model complete: dep_tree_dnn_demo.pt')

        print('Demo complete! Please check the results in current_dir/runs/logs/!')
    else:
        print('Running on real mode')

        print('Step 1: Prepare your authentication for DeepL and Google if needed.')
        input('Press ENTER to continue')

        print('Step 2: Prepare datasets and train model. Please prepare txt format data in datasets folder.')
        command = input('Filename: (Press ENTER to skip)')
        if len(command) > 1:
            translator_type = input('Translater type: (google or deepl)')
            get_dataset_for_label(current_dir + '/datasets/' + command, translator_type)

        print('Step 3: Please label the data generated or '
              'provide your own dataset in similar format and store it in datasets folder.')
        command = input('Filename: (Press ENTER to skip)')
        if len(command) > 1:
            train_machine_learning_model(current_dir + '/datasets/' + command)
            train_machine_learning_model(current_dir + '/datasets/' + command)

        print('Step 4: Prepare testing sentences in txt format or '
              'provide your own test datasets in similar format and store it in datasets folder. .')
        command = input('Filename: (Press ENTER to skip)')
        if len(command) > 1:
            translator_type = input('Translater type: (google or deepl)')
            get_dataset_for_label(current_dir + '/datasets/' + command, translator_type)

        print('Step 5: Starting to detect translation errors. The results will be stored in runs/logs/.')
        command = input('Model name: (Press ENTER to skip)')
        if len(command) > 1:
            dataset_name = input('Dataset name: (Press ENTER to skip)')
            if command.__contains__('con'):
                dataset = get_dataset(current_dir + '/datasets/' + dataset_name, 'con', 3)
            else:
                dataset = get_dataset(current_dir + '/datasets/' + dataset_name, 'dep', 3)
            if command.__contains__('dnn'):
                test_dnn_model(dataset, current_dir + '/runs/models/' + command)
            else:
                test_machine_learning_model(dataset, current_dir + '/runs/models/' + command)

        print('All steps are done! Good Bye! :)')


if __name__ == "__main__":
    print('Welcome to StructureTester!')
    main()
