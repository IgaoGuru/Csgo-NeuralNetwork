import matplotlib.pyplot as plt
import pickle

file_path = '/home/igor/mlprojects/Csgo-NeuralNetwork/modelsave/model#0r'

with open(file_path, 'rb') as file:
    stats = pickle.load(file)

    print('STATS: ----------------\n')
    inferences = stats['inferences']
    print('median inference = %s \n'%(sum(inferences)/len(inferences)))

    losses = stats['losses']
    plt.plot(losses)

    accs = stats['accuracy']
    plt.plot(accs)
    print('median accuracy: %s \n'%(sum(accs)/len(accs)))
    
    tp = stats['tp']
    tn = stats['tn']
    fp = stats['fp']
    fn = stats['fn']
    TPFN = tp+tn+fp+fn

    print('tp: %s, tn: %s, fp: %s, fn: %s \n'%(tp, tn, fp, fn))
    tp = int((tp*100) / TPFN)
    tn = int((tn*100) / TPFN)
    fp = int((fp*100) / TPFN)
    fn = int((fn*100) / TPFN)
    print('tp: %s, tn: %s, fp: %s, fn: %s (percentages) \n'%(tp, tn, fp, fn))
    print('correct = %s, incorrect = %s \n'%((tp + tn), (fp + fn)))

    print(accs[30000:])

    plt.show()
