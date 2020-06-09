import matplotlib.pyplot as plt
import pickle

file_path = '/home/igor/mlprojects/Csgo-NeuralNetwork/modelsave/model#2r'

with open(file_path, 'rb') as file:
    stats = pickle.load(file)

    #training
    print('STATS: ----------------\n')
    inferences = stats['inferences']
    print('median inference = %s \n'%(sum(inferences)/len(inferences)))

    losses = stats['losses']
    print('median loss: %s \n'%(sum(losses)/len(losses)))
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

    #validation
    if stats['type'] == 'training':
        
        val_loss = stats['val_losses']
        print('median validation loss: %s \n'%(sum(accs)/len(accs)))
        plt.plot(val_loss)

        val_accs = stats['val_accs']
        print('median validation accuracy: %s \n'%(sum(accs)/len(accs)))
        plt.plot(val_accs)
        
    # plt.legend((losses, accs, val_loss, val_accs), ('losses', 'accs', 'val_loss', 'val_accs'))
    plt.show()
