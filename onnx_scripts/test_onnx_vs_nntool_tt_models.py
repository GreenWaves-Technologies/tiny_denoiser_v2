#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import onnxruntime
import numpy as np
import torch
import os
import torchaudio
from conv_stft import ConvSTFT, ConviSTFT
import librosa
import sys
from tqdm import tqdm
import warnings
import argparse
from argparse import Namespace
from nntool.api import NNGraph
from nntool.api.utils import model_settings, quantization_options, tensor_plot
from nntool.execution.graph_executer import GraphExecuter
from nntool.stats.activation_ranges_collector import ActivationRangesCollector
warnings.filterwarnings("ignore")

def reset_states(batch_size, rnn_units, dtype='float32', tensor_type="np"):
  rnn_h_state = np.zeros((batch_size, rnn_units)).astype('float32')
  rnn_c_state = np.zeros((batch_size, rnn_units)).astype('float32')
  if tensor_type =="torch":
    rnn_h_state = torch.from_numpy(rnn_h_state).to("cpu")
    rnn_c_state = torch.from_numpy(rnn_c_state).to("cpu")
  return rnn_h_state, rnn_c_state

def save_wavs(estimates, noisy_sigs, filenames, out_dir, sr=16_000):
    # Write result
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        write(noisy, filename + "_noisy.wav", sr=sr)
        write(estimate, filename + "_enhanced.wav", sr=sr)

def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--models_dir', type=str, help='Directory for models')
    parser.add_argument('--samples_dir', type=str, help='Directory of noisy samples')
    parser.add_argument('--noisy_file_path', type=str, help='Path to a noisy sample')
    parser.add_argument('--output_dir', type=str, default='./test_onnx_tt_denoiser', help='Output directory')
    parser.add_argument('--rank', type=int, default=2, help='Rank')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], help='RNN type (lstm or gru)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--offset', type=int, default=0, help='Offset')
    parser.add_argument('--win_len', type=int, default=400, help='Window length')
    parser.add_argument('--hop_length', type=int, default=100, help='Hop length')
    parser.add_argument('--fft_len', type=int, default=512, help='FFT length')
    parser.add_argument('--rnn_units', type=int, default=256, help='RNN units')
    parser.add_argument('--win_type', type=str, default='hann', help='Window type')
    parser.add_argument('--librosa_stft_center', action='store_true', help='Use Librosa STFT center')
    return parser

def test_onnx_runner_with_conv(args):
  stft = ConvSTFT(args.win_len, args.hop_length, args.fft_len, args.win_type, 'real')
  istft = ConviSTFT(args.win_len, args.hop_length, args.fft_len, args.win_type, 'real')
  
  # Load the ONNX model
  onnx_session = onnxruntime.InferenceSession(args.onnx_model_path)
  
  # Get the names of input tensors
  input_tensor_names = [input.name for input in onnx_session.get_inputs()]
  # print("Input Tensor Names:", input_tensor_names)
  # print("")
  
  # Get the names of output tensors
  output_tensor_names = [output.name for output in onnx_session.get_outputs()]
  # print("Output Tensor Names:", output_tensor_names)

  
  for counter in tqdm(range(args.num_wav_files), desc="Denoising Progress"):
    noisy_filename = args.wav_files[counter]
    noisy_filename_with_no_extention = ".".join(noisy_filename.split(".")[:-1])
    noisy_file_path = os.path.join(args.samples_dir, noisy_filename)
  
    
    # Load the noisy signal
    noisy_signal, sr = torchaudio.load(noisy_file_path, frame_offset=args.offset)
    if sr != args.sample_rate:
        downsample_resample = torchaudio.transforms.Resample(
            sr, args.sample_rate, resampling_method='sinc_interpolation')
        noisy_signal = downsample_resample(noisy_signal)
    
    noisy_signal = torch.reshape(noisy_signal, (1, 1, -1))
    noisy_mags, phase = stft(noisy_signal)
    
    # Init the states
    batch_size = 1
    # rnn1_h_state, rnn1_c_state = reset_states(batch_size, rnn_units, tensor_type="torch")
    # rnn2_h_state, rnn2_c_state = reset_states(batch_size, rnn_units, tensor_type="torch")
    rnn1_h_state, rnn1_c_state = reset_states(batch_size, args.rnn_units)
    rnn2_h_state, rnn2_c_state = reset_states(batch_size, args.rnn_units)
    
    # Loop over all mags (timesteps) to creat denoising masks
    all_ts_masks = []
    for i in range(noisy_mags.shape[-1]):
      ts = noisy_mags[:,:,i:i+1]
      inputs = {'input': ts.detach().numpy(),
                'rnn1_h_state_in':rnn1_h_state,
                'rnn1_c_state_in':rnn1_c_state,
                'rnn2_h_state_in':rnn2_h_state,
                'rnn2_c_state_in':rnn2_c_state}
      outputs = onnx_session.run(output_tensor_names, inputs)
      ts_est_mask, rnn1_h_state, rnn1_c_state, rnn2_h_state, rnn2_c_state = outputs  
      all_ts_masks.append(ts_est_mask)
      
    # stack all timesteps masks
    all_ts_masks = torch.Tensor(np.concatenate(all_ts_masks,axis=-1))  
      
    # apply masks to the noisy_mags
    denoised_mags = noisy_mags * all_ts_masks
    denoised_signal = istft(denoised_mags, phase=phase)
    
    # export
    write(noisy_signal[0,:,:], os.path.join(args.output_dir, noisy_filename_with_no_extention + "_noisy.wav"), sr=args.sample_rate)
    write(denoised_signal[0,:,:], os.path.join(args.output_dir, noisy_filename_with_no_extention + "_enhanced_onnx_runner__conv.wav"), sr=args.sample_rate)
    
    
def test_onnx_runner_with_librosa(args):
      
  # Load the ONNX model
  onnx_session = onnxruntime.InferenceSession(args.onnx_model_path)
  
  # Get the names of input tensors
  input_tensor_names = [input.name for input in onnx_session.get_inputs()]
  # print("Input Tensor Names:", input_tensor_names)
  # print("")
  
  # Get the names of output tensors
  output_tensor_names = [output.name for output in onnx_session.get_outputs()]
  # print("Output Tensor Names:", output_tensor_names)

  
  for counter in tqdm(range(args.num_wav_files), desc="Denoising Progress"):
      noisy_filename = args.wav_files[counter]
      noisy_filename_with_no_extention = ".".join(noisy_filename.split(".")[:-1])
      noisy_file_path = os.path.join(args.samples_dir, noisy_filename)
      
      noisy_signal, _ = librosa.load(noisy_file_path, sr=args.sample_rate)
      if args.win_type == "hann":librosa_window_type='hamming'
      
      spec_data = librosa.stft(noisy_signal, n_fft=args.fft_len, hop_length=args.hop_length,
                               win_length=args.win_len, window=librosa_window_type,
                               center=args.librosa_stft_center)
      
      # Get magnitude and phase
      noisy_mags = np.abs(spec_data)
      noisy_mags = torch.Tensor(np.reshape(noisy_mags, (1, noisy_mags.shape[0], noisy_mags.shape[1])))
      
      # Init the states
      batch_size = 1
      rnn1_h_state, rnn1_c_state = reset_states(batch_size, args.rnn_units)
      rnn2_h_state, rnn2_c_state = reset_states(batch_size, args.rnn_units)
      
      # Loop over all mags (timesteps) to creat denoising masks
      all_ts_masks = []
      for i in range(noisy_mags.shape[-1]):
        ts = noisy_mags[:,:,i:i+1]
        inputs = {'input': ts.detach().numpy(),
                  'rnn1_h_state_in':rnn1_h_state,
                  'rnn1_c_state_in':rnn1_c_state,
                  'rnn2_h_state_in':rnn2_h_state,
                  'rnn2_c_state_in':rnn2_c_state}
        outputs = onnx_session.run(output_tensor_names, inputs)
        ts_est_mask, rnn1_h_state, rnn1_c_state, rnn2_h_state, rnn2_c_state = outputs  
        all_ts_masks.append(ts_est_mask)
        
      # stack all timesteps masks
      all_ts_masks = torch.Tensor(np.concatenate(all_ts_masks,axis=-1))  
      
      # apply masks to the noisy_mags
      denoised_mags = noisy_mags * all_ts_masks
      denoised_mags = denoised_mags.detach().numpy()
      
      # reconstruct back to time domain
      phase = np.angle(spec_data)
      denoised_signal = librosa.istft(denoised_mags[0,:,:] * np.exp(1j * phase),
                                      n_fft=args.fft_len, hop_length=args.hop_length, win_length=args.win_len,
                                      window=librosa_window_type, center=args.librosa_stft_center)
      
      # export
      denoised_signal = torch.reshape(torch.Tensor(denoised_signal), (1, -1))
      noisy_signal = torch.reshape(torch.Tensor(noisy_signal), (1, -1))
      # write(noisy_signal, os.path.join(args.output_dir, noisy_filename_with_no_extention + "_noisy__onnx_runner__librosa.wav"), sr=sr)
      write(denoised_signal, os.path.join(args.output_dir, noisy_filename_with_no_extention + "_enhanced_onnx__runner__librosa.wav"), sr=args.sample_rate)
      
      
def test_onnx_runner_with_nntool(args):
      
  G = NNGraph.load_graph(args.onnx_model_path, old_dsp_lib=False)
  # print(model.show())
  # model.draw()
  use_ema = False
  executer = GraphExecuter(G, qrecs=None)
  stats_collector = ActivationRangesCollector(use_ema=use_ema)
  if args.win_type == "hann":librosa_window_type='hamming'
  
  for counter in tqdm(range(args.num_wav_files), desc="Denoising Progress"):
      noisy_filename = args.wav_files[counter]
      noisy_filename_with_no_extention = ".".join(noisy_filename.split(".")[:-1])
      noisy_file_path = os.path.join(args.samples_dir, noisy_filename)
      
      noisy_signal, _ = librosa.load(noisy_file_path, sr=args.sample_rate)
      spec_data = librosa.stft(noisy_signal, n_fft=args.fft_len, hop_length=args.hop_length,
                               win_length=args.win_len, window=librosa_window_type,
                               center=args.librosa_stft_center)
      
      noisy_mags  = np.abs(spec_data)
      phase = np.angle(spec_data)
      
      # reset states for new audio file
      denoising_masks = []
      seq_length = noisy_mags.shape[-1]
      batch_size = 1
      rnn1_h_state, rnn1_c_state = reset_states(batch_size, args.rnn_units)
      rnn2_h_state, rnn2_c_state = reset_states(batch_size, args.rnn_units)
      for ts in range(seq_length):
        noisy_ts = np.reshape(noisy_mags[:,ts], (1, -1, 1))
        noisy_ts_data = [noisy_ts, rnn1_h_state, rnn1_c_state, rnn2_h_state, rnn2_c_state]
        stats_collector.collect_stats(G, noisy_ts_data) # => export if needed at the end by dumping stats_collector.stats
      
        ts_outputs = executer.execute(noisy_ts_data, qmode=None, silent=True, output_dict=True)
        denoising_mask = ts_outputs["output_1"][0]
        rnn1_h_state = ts_outputs["output_2"][0]
        rnn1_c_state = ts_outputs["output_3"][0]
        rnn2_h_state = ts_outputs["output_4"][0]
        rnn2_c_state = ts_outputs["output_5"][0]
        denoising_masks.append(denoising_mask)
      
      # stack all timesteps masks
      denoising_masks = torch.Tensor(np.concatenate(denoising_masks,axis=-1))  
      
      # apply masks to the noisy_mags
      noisy_mags = torch.Tensor(np.reshape(noisy_mags, (1, noisy_mags.shape[0], noisy_mags.shape[1])))
      denoised_mags = noisy_mags * denoising_masks
      denoised_mags = denoised_mags.detach().numpy()
      
      denoised_signal = librosa.istft(denoised_mags * np.exp(1j * phase),
                                      n_fft=args.fft_len, hop_length=args.hop_length, win_length=args.win_len,
                                      window=librosa_window_type, center=args.librosa_stft_center)
      # export
      denoised_signal = torch.from_numpy(denoised_signal).to("cpu")
      noisy_signal = torch.reshape(torch.from_numpy(noisy_signal), (1, -1))
      # write(noisy_signal, os.path.join(args.output_dir, noisy_filename_with_no_extention + "_noisy__nntool_executer__librosa.wav"), sr=sr)
      write(denoised_signal, os.path.join(args.output_dir, noisy_filename_with_no_extention + "_enhanced__nntool_executer___librosa.wav"), sr=args.sample_rate)


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    
    if args.samples_dir is None and args.noisy_file_path is None:
      parser.error("\n\nAt least one of 'samples_dir' and 'noisy_file_path' should be assigned.")

    if args.models_dir is None:
      parser.error("\n\nPlease provide directory to the onnx models.")
    
    if args.samples_dir is None:
      sp = args.noisy_file_path.split("/")
      args.samples_dir = "/".join(sp[:-1])
      wav_files = [sp[-1]]
    else:
      files = os.listdir(args.samples_dir)
      wav_files = [".".join(file.split(".")[:-1]) for file in files if file.endswith('.wav')]
    
    num_wav_files = len(wav_files)

    print(f"Found {num_wav_files} wav files to denoise!\n")
    
    args = Namespace(**vars(args), wav_files=wav_files) 
    args = Namespace(**vars(args), num_wav_files=num_wav_files) 
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = f"tt_denoiser_rank_{args.rank}.onnx"
    args.output_dir = os.path.join(args.output_dir, model_name.split(".")[0])
    os.makedirs(args.output_dir, exist_ok=True)

    onnx_model_path = os.path.join(args.models_dir, model_name)
    args = Namespace(**vars(args), onnx_model_path=onnx_model_path) 


    # onnx_runner (model) + ConvSTFT/iConvSTFT
    test_onnx_runner_with_conv(args)

    # onnx_runner (model) + librosaSTFT/iSTFT
    test_onnx_runner_with_librosa(args)
    
    # nntool_executer (model) + librosaSTFT/iSTFT
    test_onnx_runner_with_nntool(args)
    
    

if __name__ == "__main__":
    main()
