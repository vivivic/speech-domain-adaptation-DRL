import os
import sys
from os.path import join
import numpy as np
from hparams import get_hparams



hparams = get_hparams()

def train(root):

    mel_all_clean = []
    mel_all_noisy = []
    if not os.path.exists('./etc'):
        os.mkdir('./etc')

    speaker = os.listdir(root)
    print(speaker)

    for sp in speaker : 
        out_dir = os.path.join(root, sp)
        lists = os.listdir(out_dir)
          

        if sp == 'tr05_orig_clean':
        
            
            for t in lists :
        
                print(os.path.join(out_dir, t))
                
                mel_filename = t
                mel_spectrogram = np.load( os.path.join(out_dir, t) )
                mel_all_clean.append(mel_spectrogram)
        
                p = int(1)
        
                for n in range(mel_spectrogram.shape[0]):
                    with open('./etc/{}_{}_list.csv'.format('Train', sp), 'a') as li:
                        li.write('{}\t{}\t{}\t{}'.format(sp, mel_filename,  str(int(n)), str(p) )  + '\n')

        elif sp == 'tr05_real_noisy':

            for t in lists :
        
                print(os.path.join(out_dir, t))
                
                mel_filename = t
                mel_spectrogram = np.load( os.path.join(out_dir, t) )
                noise_type = mel_filename.split('.')[0].split('_')[-1]

                mel_all_noisy.append(mel_spectrogram)
        
                p = int(1)
        
                for n in range(mel_spectrogram.shape[0]):
                    with open('./etc/{}_{}_list.csv'.format('Train', sp), 'a') as li:
                        li.write('{}\t{}\t{}\t{}'.format(sp, mel_filename,  str(int(n)), str(p) )  + '\n')

        

    print(len(mel_all_clean))
    mel_all_clean = np.concatenate(mel_all_clean, axis=0)
    mel_all_noisy = np.concatenate(mel_all_noisy, axis=0)

    q001_clean = np.percentile(mel_all_clean, 0.01, axis = 0)
    q999_clean = np.percentile(mel_all_clean, 99.9, axis =  0)

    q001_noisy = np.percentile(mel_all_noisy, 0.01, axis = 0)
    q999_noisy = np.percentile(mel_all_noisy, 99.9, axis =  0)

    np.save('./etc/min_clean.npy', q001_clean)
    np.save('./etc/max_clean.npy', q999_clean)
    np.save('./etc/min_noisy.npy', q001_noisy)
    np.save('./etc/max_noisy.npy', q999_noisy)

def test(root):

    if not os.path.exists('./etc'):
        os.mkdir('./etc')

    speaker = sorted(os.listdir(root))

    speaker = speaker[:2]
    print(speaker)

    for sp in speaker : 

        out_dir = os.path.join(root, sp)
        lists = os.listdir(out_dir)
          
        mel_all = []
        
            
        for t in lists :
    
            print(os.path.join(out_dir, t))
            
            mel_filename = t
            mel_spectrogram = np.load( os.path.join(out_dir, t) )
            p = int(1)
    
    
            for n in range(mel_spectrogram.shape[0]):
                with open('./etc/{}_{}_list.csv'.format('Test', sp), 'a') as li:
                    li.write('{}\t{}\t{}\t{}'.format(sp, mel_filename,  str(int(n)), str(p) )  + '\n')
        


if __name__ == "__main__":

    root_train = '../dataset/feat/train'
    train(root_train) 

    root_test = '../dataset/feat/test'
    test(root_test) 
