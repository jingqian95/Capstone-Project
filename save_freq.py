
####Through this file, we want to save three things: all the freq data in a dictionary 
####form {'freq': (freq),'label': label} for later training use; the list called 'no_label' 
####which saves all the wavfile data with no labels; another list called 'log_files' which logs
###the files whose duration is longer than 4000 seconds or shorter than 1500 seconds.###
from scipy.misc import comb
from scipy.io import wavfile
import soundfile as sf
import wave
import contextlib
import pandas as pd
import json
import numpy as np
import scipy.io.wavfile as wav
import os
import argparse
import pickle as pkl

####Log the files which have no labels###
no_label = []

######log the files whose duration is longer than 4000 seconds or shorter than 1500 seconds###
log_files = []

#####'all_labels.csv' is the csv file contains all the labels of all the audio files###
label = pd.read_csv('all_labels.csv')

def extract_features(rootdir):
    ###audio_dict is created so that we can save our freq in a dictionary form {'freq': (freq),'label': label}
    audio_dict = np.array([])
    for root, dirs, files in os.walk(rootdir):
            for fname in files:
                if fname.endswith('.wav'):
                    path=os.path.join(root,fname)
                    f_prefix = ''.join(fname.split('.')[:-1])
                    #if not os.path.exists(os.path.join(root,f_prefix+'.json')):
                    try:
                        lab = label.loc[label['docket_id']==f_prefix]['label'].item()
                    except:
                        no_label.append(path)
                        continue
                    f = sf.SoundFile(path)
                    duration=len(f) / f.samplerate
                    if duration<4000 and duration>1500:
                        sampFreq, snd=wavfile.read(path)
                        snd = snd / (2.**15)
                        freq_dict = {'freq':snd.astype(np.float64), 'label':lab}
                        audio_dict = np.unique(np.append(freq_dict['freq'],audio_dict))
                        print(audio_dict.shape)
                        with open(os.path.join(root,f_prefix+'.json'),'wb') as out:
                            pkl.dump(freq_dict,out,protocol=pkl.HIGHEST_PROTOCOL)
                    else:
                        log_files.append(path)
                        continue
    with open('no_label.pkl', 'wb') as f:
        pkl.dump(no_label, f)
    with open('log_files.pkl', 'wb') as fout:
        pkl.dump(log_files, fout)   
extract_features(os.getcwd())
