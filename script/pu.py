#!/usr/bin/env python

import argparse
import os 
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GRU
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Ex command line : python pu.py --test binary --learning_rate 0.0001 --batch_size 4 --path_data /path/to/root/dir --path_save /path/to/save/dir --epoch 2
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required = True, dest = "test")
    parser.add_argument("--learning_rate", required = True, dest = "lrate")
    parser.add_argument("--batch_size", required = True, dest = "batch_size")
    parser.add_argument("--path_data", required = True, dest = "path_data")
    parser.add_argument("--path_save", required = True, dest = "path_save")
    parser.add_argument("--epoch", required = True, dest = "epoch")
    parser.add_argument("--transf", required = True, dest = "transf")

    args = parser.parse_args()

    test = args.test
    lrate = float(args.lrate)
    path_data = args.path_data
    path_save = args.path_save
    batch_size = int(args.batch_size)
    epoch = int(args.epoch)
    transf_model = args.transf

    return test, lrate, batch_size, path_data, path_save, epoch, transf_model

def split_data(split_coeff, data_list, train_data, test_data, val_data):
    '''Split data in 64/20/16 for training, test and validation dataset'''
    data_split = int((len(data_list))*split_coeff)
    test_data += data_list[data_split:]

    tmp_train_data = data_list[:data_split] # 80% of initial data
    train_split = int(len(tmp_train_data)*split_coeff)

    train_data += tmp_train_data[:train_split]
    val_data += tmp_train_data[train_split:]
    return train_data, test_data, val_data

def data_generator(files, batch_size, num_classes, method, class_classification = None, class_encoding = None):
    '''Give batch of data according to batch size. For binary classification, the encoding is done here'''
    while True:
        np.random.shuffle(files)
        for i in range(0, len(files) - batch_size + 1, batch_size):
            batch_files = files[i:i+batch_size]
            batch_data = []
            batch_labels = []
            for file in batch_files:
                data = pd.read_csv(file, header=None).values
                label = file.split('/')[-2]
                if method == "binary" or method is None:
                    if label == class_classification or label is None:
                        batch_labels.append(0)
                        batch_data.append(data)
                    else:
                        batch_labels.append(1)
                        batch_data.append(data)
                elif method == "categorical":
                    batch_labels.append(class_encoding[label])
                    batch_data.append(data)
                           
            batch_data = np.array(batch_data)
            if method == "binary":
                batch_labels = np.array(batch_labels)
            else:
                batch_labels = np.array(batch_labels)
                batch_labels = to_categorical(batch_labels, num_classes)
                print(type(batch_labels),batch_labels)

            yield batch_data, batch_labels

def gru_binary(lrate, data_shape):
    """Binary classification using a GRU model
    First layer also gives back sequences for the second layer
    Defining the optimizer to allow control on learning rate
    """
    model = Sequential()
    model.add(GRU(64, input_shape=(data_shape[0], data_shape[1]), return_sequences=True))
    model.add(GRU(64))
    model.add(Dense(1, activation='softmax'))

    optimizer = Adam(learning_rate=lrate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def gru_cat(transf_model, lrate, data_shape):
    """Categorical classification using GRU model and transfer learning from a binary classification model
    Last layer of GRU binary model goes into dense layer with 64 units
    Can't use GRU model because the binary model layer does not return sequences"""
    optimizer = Adam(learning_rate=lrate)

    denselay = Dense(64)(transf_model.layers[-2].output)
    outputlay = Dense(num_classes, activation='softmax')(denselay)
    cat_model = Model(transf_model.input, outputlay)

    cat_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return cat_model
    
def plot_loss(history, name, batch_size, lrate):
    """Plot the training and validation loss"""
    print("Start plotting loss")

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title(f'Training and Validation Loss, batch-size : {batch_size} / learning-rate : {lrate}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    print("Saving loss plot !")
    plt.savefig(f'./save/{name}_loss.png')
    plt.close()

def plot_acc(history, name, batch_size, lrate):
    """Plot the training and validation accuracy"""
    print("Start plotting accuracy")
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy, batch-size : {batch_size} / learning-rate : {lrate}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    print("Saving accuracy plot !")
    plt.savefig(f'{save_name}_accuracy.png')
    plt.close()

if __name__ == "__main__":
    test, lrate, batch_size, path_data, path_save, epoch, transf_model = get_args()  
    save_name = f"{path_save}/{test}_adjusted_batch{batch_size}_lr{str(lrate).split('e')[-1]}" #Name for plot and model saving

    file_shape = (60, 1280)
    data_count = {}
    train_data, test_data, val_data = [], [], []

    # Generate a dictionary with key for PU class and value for the count of samples inside this class
    core_list = os.listdir(path_data)
    split_coeff = .8
    for core in core_list:
        tmp_list = [f"{path_data}/{core}/{data}" for data in os.listdir(os.path.join(path_data, core))]
        random.shuffle(tmp_list)
        split_data(split_coeff, tmp_list, train_data, test_data, val_data)
        data_count[core] = len(tmp_list)

    random.shuffle(train_data)
    random.shuffle(val_data)

    # Binary doesn't need encoding because it is done inside the data generator
    if test == "binary":
        num_classes = 2
        binary_class = "noncore"
        train_generator = data_generator(train_data, batch_size, test, num_classes, class_classification = binary_class)
        val_generator = data_generator(val_data, batch_size, test, num_classes, class_classification = binary_class)

        model = gru_binary(lrate, data_shape = file_shape)

    elif test == "categorical":
        tmodel = load_model(transf_model)

        num_classes = len(data_count.keys())

        #Encoding label and adding class weights
        class_encoding = {}
        class_weights = {}
        for i, (key, value) in enumerate(data_count.items()):
            class_encoding[key] = i
            class_weights[i] = round(12000/value, 2)

        train_generator = data_generator(train_data, batch_size, test, num_classes, class_encoding = class_encoding)
        val_generator = data_generator(val_data, batch_size, test, num_classes, class_encoding = class_encoding)
        model = gru_cat(tmodel, lrate, file_shape)
    
    # Define training and validation steps (because of the use of a generator)
    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size
    
    # Define a early stopping function, to stop model from overfitting. Use val_loss as a monitoring function
    early_stopping_callback = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    mode = 'min',
    verbose = 1)
    
    # Prepare .csv file for saving training log
    output_file = f"{save_name}.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(output_file)
    
    if test == "binary":
        hist = model.fit(
            train_generator, 
            validation_data = val_generator, 
            steps_per_epoch = train_steps, 
            validation_steps = val_steps, 
            epochs = epoch, 
            callbacks = [early_stopping_callback, csv_logger])
    elif test == "categorical":
        hist = model.fit(
            train_generator,
            validation_data = val_generator,
            steps_per_epoch = train_steps,
            validation_steps = val_steps,
            class_weight = class_weights,
            epochs = epoch,
            callbacks = [early_stopping_callback, csv_logger]) 

    print("Training finished !")

    print("Saving model !")
    model.save(f"{save_name}.h5")

    print("Start plotting !")
    plot_loss(hist, save_name,  batch_size, lrate)
    plot_acc(hist, save_name,  batch_size, lrate)

