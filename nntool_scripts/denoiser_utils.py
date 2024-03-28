
import os
import librosa
import numpy as np

TRACK_METRICS = ["ovrl_mos", "sig_mos", "bak_mos", "p808_mos", "pesq", "stoi", "meanerr"]

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

    stft = librosa.stft(data, win_length=frame_size, n_fft=n_fft, hop_length=frame_step, window=win_func, center=True)    
    return stft

def postprocessing(stfts, frame_size=400, frame_step=100, n_fft=512, win_func="hann"):
    data = librosa.istft(stfts, win_length=frame_size, n_fft=n_fft, hop_length=frame_step, window=win_func, center=True)
    # clip for problems in the dnsmos
    data = np.clip(data, -1.0, 1.0)
    return data

def gather_results(row_list, noisy_row_list, csv_file=None, csv_file_allfiles=None, model_name="no_name"):
    import pandas as pd

    df = pd.DataFrame(row_list, columns=["filename"] + TRACK_METRICS)
    df_noisy = pd.DataFrame(noisy_row_list, columns=["filename"] + TRACK_METRICS)
    if csv_file_allfiles:
        df.to_csv(csv_file_allfiles, index=False, encoding='utf-8')
        df_noisy.to_csv(csv_file_allfiles, index=False, encoding='utf-8', mode="a")

    df_mean = df[TRACK_METRICS].mean().to_frame().transpose()
    df_noisy_mean = df_noisy[TRACK_METRICS].mean().to_frame().transpose()
    col_order = []
    for col in df_mean.columns.values:
        df_mean[f"d{col}"] = df_mean[col] - df_noisy_mean[col]
        df_noisy_mean[f"d{col}"] = df_noisy_mean[col] - df_noisy_mean[col]
        col_order += [col, f"d{col}"]
    df_mean = df_mean[col_order]

    if csv_file:
        print(f"Writing summary results to {csv_file}")
        df_mean["name"] = model_name
        df_mean = df_mean[["name"] + col_order]
        if not os.path.exists(csv_file):
            df_noisy_mean["name"] = "Noisy"
            df_noisy_mean = df_noisy_mean[["name"] + col_order]
            df_noisy_mean.to_csv(csv_file, mode="w", index=False, header=True)
        df_mean.to_csv(csv_file, mode="a", index=False, header=False)

    return df_mean, df_noisy_mean
