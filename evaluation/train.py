from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import datasets

import pickle
import os
import numpy as np

from classifier import Classifier
from optimizer import Optimizer
from dataset import DataSet

if __name__ == '__main__':

    data_dir = '../data'
    model_name = 'densenet'
    output_classes = 2
    feature_extract = True
    batch_size = 8
    num_epochs = 20

    learningRate = 0.001
    momentum = 0.9

    run_id = 'l_' + str(learningRate) + '_m_' + str(momentum)

    saved_data_structures_dir = 'saved_data_structures/'

    densenetClassifier = Classifier(model_name, output_classes, batch_size, num_epochs, feature_extract)
    model = densenetClassifier.initPretrainedModel(224)

    dataloaders_dict, dataset_sizes, class_names = DataSet.initDataLoaders(data_dir, batch_size)
    data_transforms = DataSet.setUpDataLoaderTransformers()

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val', 'test']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sgdOptimizer = Optimizer(device)
    optimizer_ft = sgdOptimizer.optimize(model, feature_extract, learningRate, momentum)

    criterion = nn.CrossEntropyLoss()

    model, val_acc_history, per_epoch_loss, per_epoch_accuracy = densenetClassifier.train_model(model,
                                                criterion,
                                                optimizer_ft,
                                                dataloaders_dict,
                                                dataset_sizes)

    print(f"Validation Accuracy History:\n{val_acc_history}")
    print(f"\nPer epoch loss:\n{per_epoch_loss}")
    print(f"\nPer epoch accuracy:\n{per_epoch_accuracy}")

    print("Saving final model to disk...")
    save_as_name = '../trained_models/densenetFeatureExtraction_' + run_id + '.pt'
    torch.save({
        'name': 'densenet_feature_extraction_' + run_id,
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_ft.state_dict(),
    }, save_as_name)

    # testing
    print("Running independent test...")
    state = torch.load(save_as_name)
    model.load_state_dict(state['model_state_dict'])
    predictions = densenetClassifier.testModel(dataloaders_dict, model, class_names, dataset_sizes, batch_size=8)

    # save predicted values
    save_as_name = '../trained_models/predicted_labels/predictedLabelsDensenet_' + run_id + '.csv'
    np.savetxt(save_as_name, predictions, fmt='%s')
