import librosa
import librosa.filters
import math
import numpy as np
import os
import struct
from scipy import signal
from hparams import get_hparams
from scipy.io import wavfile

import lws

hparams = get_hparams()

def load_wav(path):
    ext = path.split('.')[-1]
    if ext == 'wav':
        return librosa.core.load(path, sr=hparams.fs)[0]
    if ext == 'pcm':
        return pcm_read(path)

def pcm_read(filename,dtype = np.float32):

    statinfo = os.stat(filename)
    length = statinfo.st_size
    x = []
    with open(filename, 'rb') as f1:
        chunksize = 2
        while length > 0 : 
            data = f1.read(chunksize)
            x.append(struct.unpack("<h",data))
            length -= chunksize
    f1.close()

    x = np.asarray(x, dtype = dtype).squeeze()
    x = x / 32768.0

    return x

def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hparams.fs, wav.astype(np.int16))



def melspectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    if not hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    return _normalize(S)



def _preemphasis(y):
    return signal.lfilter([1, -hparams.preemphasis], [1], y)


def librosa_pad_lr(x, fsize):

    pad = int( fsize // 2)

    if pad % 2 == 0:
        return int(pad//2), int(pad//2 )
    else:
        return int(pad//2), int(pad//2 + 1)

# Conversions:


_mel_basis = None

def _stft(y):
    return librosa.stft(y = y, n_fft = hparams.fft_size, hop_length =  hparams.hop_size, win_length = hparams.win_size) 

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    assert hparams.fmax <= hparams.fs // 2
    return librosa.filters.mel(hparams.fs, hparams.fft_size,
                               fmin=hparams.fmin, fmax=hparams.fmax,
                               n_mels=hparams.num_mels)


def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
