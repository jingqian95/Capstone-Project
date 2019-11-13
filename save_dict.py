import os
import pickle as pkl
import numpy as np

#######get the freq dictionary code##########
###contains all the amplitute numbers that occured in the training data
audio_dict = np.array([])

rootdir = '/scratch/jq689/wavfile_1998'
for root, dirs, files in os.walk(rootdir):
    for fname in files:
        if fname.endswith('.json'):
            if fname.endswith('mfcc.json'):
                continue
            else:
                path=os.path.join(root,fname)
                pickle_in = open(path,"rb")
                file_dict = pkl.load(pickle_in)
                audio_dict = np.unique(np.append(file_dict['freq'],audio_dict))

with open('audio_dict.pkl','wb') as out:
    pkl.dump(audio_dict,out)

