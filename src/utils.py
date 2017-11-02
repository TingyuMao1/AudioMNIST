import librosa
import os
import random
import numpy as np
import torch


def lrelu(x, leaky=0.01):
    return torch.max(x, leaky * x)


def acc(output, target):
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct = pred.eq(target.data).cpu().sum()
    return correct


def load_dataset(path, sr):
    label_mapping = {
            "z" : 0, 
            "1" : 1, 
            "2" : 2, 
            "3" : 3, 
            "4" : 4, 
            "5" : 5, 
            "6" : 6, 
            "7" : 7, 
            "8" : 8, 
            "9" : 9, 
            "o" : 10
            }
    dataset = []
    for f in os.listdir(path):
        audio, _ = librosa.load(os.path.join(path, f), sr)
        dataset.append((audio, label_mapping[f[0]]))
    return dataset

def Divide(data, sr, fs=0.025, ws=0.01)
    """
    Divide the data into overlappling frames, return the number of frames and the matrix
    """
    number_of_frames = data.size/(sr*ws)
    A = np.zeros((number_of_frames, sr*fs))
    j = 0
    while j*sr*ws + sr*fs <= data.size:
        A[j, :] = np.array(data)[j*sr*ws : j*sr*ws + sr*fs]
        j = j + 1
    A = A[0:j, :].transpose()
    return(A.shape[1], A)

def extract_features(audio, sr, n_mfcc):
    """
    Extract features from @audio

    The features are 
        * mfcc
        * chroma
        * mel
        * spectral_contrast
        * tonnetz

    Refer to http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

    @param
        audio: audio file loaded from librosa
        sr: sample rate
        n_mfcc: number of mfcc components

    @return
        tuple of five extracted features
    """
    stft = np.abs(librosa.stft(audio))
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(audio, sr=sr).T,axis=0)
    sc = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(audio), sr=sr).T,axis=0)

    return (mfcc, chroma, mel, sc, tonnetz)


def batchify(data, label, batch_size=None, shuffle=False):
    if not batch_size:
        batch_size = len(data)

    data_size = len(data)
    order = list(range(data_size))
    if shuffle:
        random.shuffle(order)

    batches = int(np.ceil(1.*data_size/batch_size))
    for i in range(batches):
        start = i * batch_size
        indices = order[start:start+batch_size]
        yield (data[indices], label[indices])
