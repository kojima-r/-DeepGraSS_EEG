
import numpy as np
import pandas as pd
import mne
from mne.io import RawArray
from mne.channels import read_custom_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs
from mne.decoding import CSP

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
#from sklearn.cross_validation import cross_val_score, LeaveOneLabelOut
import glob

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.signal import welch
from mne import pick_types

import pickle
import json
import os
import scipy

def process(mode="train"):
    y=[]
    g=[]
    epochs_tot=[]
    info_list=[]
    filename_list=glob.glob("{}/*_data.csv".format(mode))
    for i, filename in enumerate(filename_list[:10]):
        print(filename)
        ev_fname = filename.replace('_data','_events')
        df=pd.read_csv(filename)
        if not os.path.exists(ev_fname):
            continue
        df_event=pd.read_csv(ev_fname)
        
        h=df.columns.values
        X=df[h[1:]].values
        X=X[:-3000,:]
        plt.plot(X)

        ch_names = list(df.columns[1:])
        montage = mne.channels.make_standard_montage('standard_1005')
        #ch_names
        ch_type = ['eeg']*len(ch_names) + ['stim']*6
        events_names = df_event.columns[1:]

        all_names=ch_names+events_names.tolist()
        info = create_info(all_names,sfreq=500.0, ch_types=ch_type)

        events_data = np.array(df_event[events_names])
        ch_data=df.values[:-2,1:]
        l=ch_data.shape[0]
        data = np.concatenate((ch_data[:l,:].T,events_data[:l,:].T))   

        raw = RawArray(data,info,verbose=False)
        picks = pick_types(raw.info,eeg=True)
        
        raw.filter(2,100, picks=picks, method='iir', n_jobs=-1, verbose=False)

        events = find_events(raw,stim_channel='Replace', verbose=False)

        epochs = Epochs(raw, events, {'during' : 1}, -2, -0.5, proj=False,
                                picks=picks, baseline=None, preload=True,)

        epochs_rest = Epochs(raw, events, {'after' : 1}, 0.5, 2, proj=False,
                                picks=picks, baseline=None, preload=True,)
        
        dt = epochs_rest.times[0] - epochs.times[0]
        epochs.shift_time(dt)
        

        epochs_tot.append(epochs)
        y.extend([1]*len(epochs))
        epochs_tot.append(epochs_rest)
        y.extend([-1]*len(epochs_rest))
        info_list.extend( [filename] * (len(epochs)+len(epochs_rest)) )
        g.extend( [i] * (len(epochs)+len(epochs_rest)) )
       
    all_epochs = concatenate_epochs(epochs_tot)
    print(epochs_tot)
    X = all_epochs.get_data()
    X = (X-np.mean(X.ravel()))/np.std(X.ravel())
    XX=[]
    for i in range(X.shape[0]):
        S=[]
        for j in range(X.shape[1]):
            f, t, Sxx=scipy.signal.spectrogram(X[i,j,:],500,nperseg=128,nfft=128,noverlap=128*3/4)
            S.append(Sxx)
            print(Sxx.shape)
        XX.append(np.concatenate(S,axis=0))
    X=np.array(XX)
    print(X.shape)
    X=X.transpose([0,2,1])
    y = np.array(y)
    g = np.array(g)
    print(X.shape)
    np.save("dataset/eeg.{}.obs.npy".format(mode),X)
    np.save("dataset/eeg.{}.y.npy".format(mode),y)
    np.save("dataset/eeg.{}.g.npy".format(mode),g)
    with open('dataset/eeg.{}_info.json'.format(mode), 'w') as fp:
        json.dump(info_list, fp, indent=4)

    with open("dataset/eeg.{}.pkl".format(mode), mode="wb") as fp:
        pickle.dump(epochs_tot, fp)
    
if __name__ == '__main__':
    process(mode="train")
    #process(mode="test")

