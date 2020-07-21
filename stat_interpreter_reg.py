import pickle
import matplotlib.pyplot as plt

dirpath = '/home/igor/mlprojects/modelsave/'
filename = 'model#999-train'
filepath = dirpath + filename
print(filepath)

def interpreter(filepath):
    with open(filepath, 'rb') as filezin:
        loss_dict = pickle.load(filezin)
        print(loss_dict)

interpreter(filepath)