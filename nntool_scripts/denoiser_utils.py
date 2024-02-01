
import librosa
import numpy as np
from pesq import pesq
from pystoi import stoi

WIN_LENGTH = 400
HOP_LENGTH = 100
N_FFT = 512
SAMPLERATE = 16000
WIN_FUNC = "hann"

def open_wav(file, expected_sr=SAMPLERATE, verbose=False):
    data, sr = librosa.load(file, sr=expected_sr)
    if sr != expected_sr:
        if verbose:
            print(f"expected sr: {expected_sr} real: {sr} -> resampling")
        data = librosa.resample(data, orig_sr=sr, target_sr=expected_sr)
    return data

def preprocessing(input_file, frame_size=400, frame_step=100, n_fft=512, win_func="hann"):
    if isinstance(input_file, str):
        data = open_wav(input_file)
    else:
        data = input_file
    
    #This is to get same size as input
    global pad_to_remove
    pad_to_remove = 400 - ((np.squeeze(data.shape)-512)%100)
    pad = np.zeros((pad_to_remove),dtype=data.dtype)
    data = np.concatenate((data,pad),axis=0)
    stft = librosa.stft(data, win_length=frame_size, n_fft=n_fft, hop_length=frame_step, window=win_func, center=False)
    #This is to remove a strange clicking on the beginning of the output
    pad = np.zeros((257,3))
    stft = np.concatenate((pad,stft),axis=1)
    
    return stft

def postprocessing(stfts, frame_size=400, frame_step=100, n_fft=512, win_func="hann"):
    
    data = librosa.istft(stfts, win_length=frame_size, n_fft=n_fft, hop_length=frame_step, window=win_func, center=False)
    #remove pad for clicking
    data = data[300:]
    #Remove the pad for size
    data = data[:-pad_to_remove]
    return data

def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val

def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val

def pesq_stoi(clean, estimate, samplerate):
    pesq_i = pesq(samplerate, clean, estimate, 'wb')
    stoi_i = stoi(clean, estimate, samplerate, extended=False)
    return pesq_i, stoi_i
