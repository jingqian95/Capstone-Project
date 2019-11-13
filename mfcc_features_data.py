###This is the file that extract the mfcc features of the data. Eventually, this py file saves all the
###mfcc feature and the label of the audio data to a dictionary in the form 
####{'MFCC':(mfcc features in),'name':file_name,'label':label}.

from scipy.misc import comb
import soundfile as sf
from pylab import*
from scipy.io import wavfile
import os
import wave
import contextlib
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
import gzip
import pickle as pkl
import platform
import sys
print(platform.python_version())

####### This is the file that extract the mfcc features and save the data to json files#########


label = pd.read_csv('all_labels.csv')
def save_mfcc_data(rootdir):
    ##max_len means the max number of the features. This is calculated by the fact that the longest file is
    # 4000 seconds and the sample rate is 22050. So the max len is 4000*22050
    max_len = 88200000
    for root, dirs, files in os.walk(rootdir):
        for fname in files:
            ####this is the two files that I found inconsistent since they have a sample rate of 40000
            if fname == '2010_09_1476.wav' or fname == '2010_09_1454.wav':
                continue
            if fname.endswith('.wav'):
                path=os.path.join(root,fname)
                f_prefix = ''.join(fname.split('.')[:-1])
                try:
                    lab = label.loc[label['docket_id']==f_prefix]['label'].item()
                    print('yes')
                except:
                    print('pass')
                    continue
                f = sf.SoundFile(path)
                duration=len(f) / f.samplerate
                if duration<4000 and duration>1500:
                    rate,sig = wav.read(path)
                    ####pad all the data for computational efficient in the future when training model##
                    padded_sig= np.pad(sig,(0,max_len-len(sig)),'constant')
                    mfcc_feat = mfcc(padded_sig,rate,numcep=20,nfft=552)
                    file_dict = {"MFCC":mfcc_feat.astype(np.float64),"name":fname.split('/')[-1].split('.')[0],'label':lab}
                    with open(os.path.join(root,f_prefix+'_mfcc'+'.json'),'wb') as f:
                        pkl.dump(file_dict,f,protocol=pkl.HIGHEST_PROTOCOL)
save_mfcc_data('/scratch/jq689/wavfile_1999')