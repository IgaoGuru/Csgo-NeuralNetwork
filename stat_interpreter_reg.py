import pickle
import matplotlib.pyplot as plt
import numpy as np

dirpath = '/home/igor/mlprojects/modelsave/'
filename = 'model#10-train'
filepath = dirpath + filename

def interpreter(filepath=None, loss_dict=None, mode=2):
    if mode != 1 and mode != 2:
        raise ValueError('please select either "1" for training, or "2" for testing')
    if filepath==None and loss_dict==None:
        raise ValueError('please input either a filepath, or a loss_dict!')
    if loss_dict == None:
        with open(filepath, 'rb') as filezin:
            loss_dict = pickle.load(filezin)
    print(loss_dict)

    plt.plot(loss_dict['loss_sum'], label='sum')
    plt.plot(loss_dict['loss_classifier'], label='classifier')
    plt.plot(loss_dict['loss_box_reg'], label='box_reg')
    plt.plot(loss_dict['loss_objectness'], label='objectness')
    plt.plot(loss_dict['loss_rpn_box_reg'], label='rpn_box_reg')
    if mode == 1:
        plt.plot(loss_dict['loss_sum_val'], label='sum_val')
        plt.plot(loss_dict['loss_classifier_val'], label='classifier_val')
        plt.plot(loss_dict['loss_box_reg_val'], label='box_reg_val')
        # plt.plot(loss_dict['loss_objectness_val'], label='objectness_val')
        plt.plot(loss_dict['loss_rpn_box_reg_val'], label='rpn_box_reg_val')
    plt.legend()
    plt.show()

def stdev_mean(filepath=None, loss_dict=None):
    if filepath==None and loss_dict==None:
        raise ValueError('please input either a filepath, or a loss_dict!')
    if loss_dict == None:
        with open(filepath, 'rb') as filezin:
            loss_dict = pickle.load(filezin)

    sum_stdev = np.std(loss_dict['loss_sum'])
    sum_mean = np.mean(loss_dict['loss_sum'])

    print(f'loss_sum mean = {sum_stdev} \n sum_mean = {sum_mean}')
    

def train_val_comparison(filepath):
    with open(filepath, 'rb') as filezin:
        loss_dict = pickle.load(filezin)

    plt.plot(loss_dict['loss_sum'], label='training_loss')
    plt.plot(loss_dict['loss_sum_val'], label='validation_loss')
    plt.legend()
    plt.show()

# interpreter(filepath=filepath, mode=2)
# train_val_comparison(filepath)