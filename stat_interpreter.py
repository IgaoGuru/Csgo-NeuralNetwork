import matplotlib.pyplot as plt
import pickle

file_path = '/home/igor/mlprojects/Csgo-NeuralNetwork/modelsave/model#1r'


def legacy(filepath):
    
    with open(file_path, 'rb') as file:
        stats = pickle.load(file)
        fig, ax = plt.subplots()

        #training
        print('STATS: ----------------\n')
        inferences = stats['inferences']
        print('median inference = %s \n'%(sum(inferences)/len(inferences)))

        runtime = stats['runtime']
        print('runtime: %s \n'%(runtime))

        losses = stats['losses']
        print('median loss: %s \n'%(sum(losses)/len(losses)))
        ax.plot(0, 1, losses, label='training losses')

        accs = stats['accuracy']
        ax.plot(0, 100, accs, label='training accuracy')
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
            print('median validation loss: %s \n'%(sum(val_loss)/len(val_loss)))
            print('best val loss: %s \n'%(min(val_loss)))
            ax.plot(0, 1, val_loss, label='validation losses')

            val_accs = stats['val_accs']
            print('median validation accuracy: %s \n'%(sum(accs)/len(accs)))
            ax.plot(0, 100, val_accs, label='validation accuracy')      
            
        legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
        plt.show()

def train_interpreter(filepath):
    
    with open(file_path, 'rb') as file:
        stats = pickle.load(file)
        fig, ax = plt.subplots()

        #training
        print('STATS: ----------------\n')
        inferences = stats['inferences']
        print('median inference = %s \n'%(sum(inferences)/len(inferences)))

        runtime = stats['runtime']
        print('runtime: %s \n'%(runtime))

        losses = stats['losses']
        print('median loss: %s \n'%(sum(losses)/len(losses)))
        print('last loss: %s \n')%(losses[len(losses)])
        print('best loss: %s \n')%(min(losses))
        ax.plot(0, 1, losses, label='training losses')

        accs = stats['accuracy']
        ax.plot(0, 100, accs, label='training accuracy')
        print('median accuracy: %s \n'%(sum(accs)/len(accs)))
        
        TPFN = stats['train_tpfn']
        last_tpfn = TPFN[len(TPFN)]
        tp = last_tpfn[0]
        tn = last_tpfn[1]
        fp = last_tpfn[2]
        fn = last_tpfn[3]

        print('(training) tp: %s, tn: %s, fp: %s, fn: %s \n'%(tp, tn, fp, fn))
        tp = int((tp*100) / sum(last_tpfn))
        tn = int((tn*100) / sum(last_tpfn))
        fp = int((fp*100) / sum(last_tpfn))
        fn = int((fn*100) / sum(last_tpfn))
        print('tp: %s, tn: %s, fp: %s, fn: %s (percentages) \n'%(tp, tn, fp, fn))
        print('correct = %s, incorrect = %s \n'%((tp + tn), (fp + fn)))

        #validation
        if stats['type'] == 'training':
            
            val_loss = stats['val_losses']
            print('median validation loss: %s \n'%(sum(val_loss)/len(val_loss)))
            print('best val loss: %s \n'%(min(val_loss)))
            ax.plot(0, 1, val_loss, label='validation losses')

            val_accs = stats['val_accs']
            print('median validation accuracy: %s \n'%(sum(accs)/len(accs)))
            ax.plot(0, 100, val_accs, label='validation accuracy')      

            TPFN = stats['val_tpfn']
            last_tpfn = TPFN[len(TPFN)]
            tp = last_tpfn[0]
            tn = last_tpfn[1]
            fp = last_tpfn[2]
            fn = last_tpfn[3]

            print('(validation) tp: %s, tn: %s, fp: %s, fn: %s \n'%(tp, tn, fp, fn))
            tp = int((tp*100) / sum(last_tpfn))
            tn = int((tn*100) / sum(last_tpfn))
            fp = int((fp*100) / sum(last_tpfn))
            fn = int((fn*100) / sum(last_tpfn))
            print('tp: %s, tn: %s, fp: %s, fn: %s (percentages) \n'%(tp, tn, fp, fn))
            print('correct = %s, incorrect = %s \n'%((tp + tn), (fp + fn)))
            
        legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
        plt.show()

legacy(file_path)