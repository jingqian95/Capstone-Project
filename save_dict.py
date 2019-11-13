import os
import pickle as pkl
import numpy as np

###Split the data into train and test set in the ratio of 8:2 in chronological order.##
data_list = []
for root, dirs, files in os.walk(args.data_dir):
    for fname in files:
        if fname.endswith('.json'):
            path=os.path.join(root,fname)
            data_list.append(path)
data_year = [f.split('/')[-1].split('_')[0] for f in data_list]
data_sorted = [x for _,x in sorted(zip(data_year,data_list))]
data_len = len(data_sorted)
train_per = 0.8
train = data_sorted[:int(data_len * train_per)]
# test = data_sorted[int(data_len * train_per):]


#######get the freq dictionary code##########
###contains all the amplitute numbers that occured in the training data
audio_dict = np.array([])
for root, dirs, files in os.walk(train):
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

