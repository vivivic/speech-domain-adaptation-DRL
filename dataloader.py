#data load from dataset, using WORLD vocoder
import os
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
import torch
from utils import Tanhize
from hparams import get_hparams


hparams = get_hparams()




def testset_list_classifier(root,list_dir):

    testset_list = []
    set_list = open(list_dir,'r').read().split('\n')

    for i in range(len(set_list)-1):
        speaker_name, mel_name, frame_num, phone_num = set_list[i].split('\t')

        testset_list.append(mel_name)

    testset_list = sorted(list(set(testset_list)))


    return testset_list, speaker_name

def testset_loader(root,audio_name, speaker, normalizer):



    bin_data = np.load(root + '/' + speaker + '/'+ audio_name)

    feature = bin_data # with hard coding 
    feature = normalizer.forward_process(feature)
    feature = torch.FloatTensor(feature)

    return {
        'sp' : feature,
        'audio_name': audio_name,
    } 



class npyDataset2d(torch.utils.data.Dataset):
    def __init__(self, root, list_dir,frame_len, normalizer):
        self.normalizer = normalizer

        set_list = open(list_dir,'r').read().split('\n')[:-1]

        set_list_tmp = []
        set_phone_tmp = []


        for i, data in enumerate(set_list):
            speaker_name, mel_name, frame_num, phone_num  = data.split('\t')

            phone_num = int(phone_num) 

            if int(frame_num) > frame_len + 1 and int(phone_num) != 0:

                set_list_tmp.append(data)
                set_phone_tmp.append(phone_num)

        self.set_list = set_list_tmp
        self.phone_list = set_phone_tmp

        self.frame_len = frame_len
        self.root = root

    def __getitem__(self, idx):

        speaker_name , mel_name, frame_num, phone = self.set_list[idx].split('\t')

        bin_data = np.load(os.path.join(self.root, speaker_name, mel_name))

        feature = bin_data[int(frame_num)-self.frame_len : int(frame_num), :] # with hard coding 
        feature = self.normalizer.forward_process(feature)
        
        return feature

    def __len__(self):
        return len(self.set_list)



if __name__ == '__main__':
    main()
