import pickle
import matplotlib.pyplot as plt

dirpath = '/home/igor/mlprojects/modelsave/'
filename = 'model#2-train'
filepath = dirpath + filename

def interpreter(filepath=None, loss_dict=None):
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
    plt.legend()
    plt.show()

interpreter(filepath=filepath)