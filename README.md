# TinyDenoiser on GAP9

This project demonstrates a Recurrent Neural Network (RNN) based method for Speech Enhamencement on GAP9.  
The main loop of the application continuosly samples data from the microphone at 16kHz, applies the RNN filter and reconstruct the cleaned signal via overlap-and-add.
As depitcted in the Figure below, the nosiy signal is windowed (frame size of 25 msec with an hop lenght of 6.25 msec and Hanning windowing) and the STFT is computed. 
The RNN is fed with the magnitude of the STFT components and return a suppression mask. After weighting, the inverse STFT returns a cleaned audio clip.

![alt text](imgs/TinyDenoiser.png?raw=true "Title")

## Demo Getting Started
The demo runs on the GAP9 Audio EVK, using the microphone of the GAPmod board.
```
cmake -B build
cmake --build build --target run
```

Optionally, the application can run on GVSOC (or board) to denoise a custom audio file (.wav).
```
cmake -B build
cmake --build build --target menuconfig # Select the options DenoiseWav in the DENOISER APP -> Application mode menu
cmake --build build --target run
```
Output wav file will be written to test_gap.wav inside the project folder.

## Project Structure
* `main_demo.c` is the main file, including the application code
* `main_wav.c` is the app file when using the DenoiseWav option
* `model/` includes the necessary files to feed GAPflow for NN model code generation: 
    * the _onnx_ denoiser files
        * `denoiser_dns.onnx` is a GRU based models trained on the [DNS][dns] dataset. It is used for demo purpose.
        * `denoiser.onnx` and `denoiser_GRU.onnx` are respectively LSTM and GRU models trained on the [Valentini][valentini]. they are used for testing purpose.
* `nntool_scripts/` includes the nntool recipes to quantize the LSTM or GRU models, generate the Autotiler code and test the performance og the deployable models. You can refer to the [quantization section](#nn-quantization-settings) for more details. It also contains the scripts to generate the Autotiler preprocessing models for the FFT/iFFT.
* `dataset/` contains the audio samples for testing and quantization claibration
* `SFUGraph.src` is the configuation file for Audio IO. It is used only for board target.

## NN Quantization Settings
The Post-Training quantization process of the RNN model is operated by the GAPflow.
Both LSTM and GRU models can be quantized using one of the different options:
* `FP16`: quantizing both activations and weights to _float16_ format. This does not require any calibration samples.
* `INT8`: quantizing both activations and weights to _int\_8_ format. A calibration step is required to quantize the activation functions. Samples included within `samples/quant/` are used to this aim. This option is currently not suggested because of the not-negligible accuracy degradation.
* `FP16MIXED`: only RNN layers are quantized to 8 bits, while the rest is kept to FP16. This option achives the **best** trade-off between accuracy degration and inference speed.
* `NE16`: currently not supported. 

## Application Mode Configuration
In addition to individual settings, some application mode are made available to simplify the APP code configuration. This is done by setting the Application Mode in the `make menuconfig` DENOISER APP menu

### GVSoC - gvcontrol
To run the `Demo` mode on GVSoC you can use the `gvcontrol` file.
the gvcontrol is used to send/read data to/from the i2s interface of the gap9 gvsoc.
You can chose the input noisy wav file you want to process. Since gap is waiting for pdm data, the pcm/pdm convertion module of gvsoc is used. To learn more about this please refer to the following example in the sdk : `basic/interface/sai/pcm2pdm_pdm2pcm`.


## Python Utilities
In the `nntool_scripts` folder there are several python utilities to test the quality and the performance of the deployable model in full python environment through NNTool.

### To denoise a wav file
```
python nntool_scripts/test_nntool_model.py --mode test_sample --trained_model model/denoiser_dns.onnx --quant_type fp16 --test_sample dataset/test/noisy/p232_050.wav --out_wav output_nntool.wav
```
The output is saved in a file called `output_nntool.wav` in the home of the repository

### To test on dataset
```
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_dns.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/
```

[dns]: https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2020/
[valentini]: https://datashare.ed.ac.uk/handle/10283/2791

Results:
| Path                  | Type | Trained on  | Pytorch     | NNTool fp32 | NNTool fp16 | NNTool Mixedne16Fp16 |
|-----------------------|------|-------------|-------------|-------------|-------------|------------------|
| denoiser_LSTM_Valentini.onnx | GRU  | DNS  | PESQ= STOI= | PESQ= STOI= | PESQ=1.9442 STOI=0.9091 | PESQ=1.9154 STOI=0.9064 | 
| denoiser_GRU_dns.onnx | GRU  | DNS         | PESQ= STOI= | PESQ= STOI= | PESQ=1.7684 STOI=0.9010 | PESQ=1.7435 STOI=0.8983 | 
| FullRankGRU.onnx      | GRU  | DNS         | PESQ= STOI= | PESQ= STOI= | PESQ=1.7684 STOI=0.9010 | PESQ=1.7435 STOI=0.8983 | 
| FullRankLSTM.onnx     | LSTM | DNS         | PESQ= STOI= | PESQ= STOI= | PESQ=1.9442 STOI=0.9091 | PESQ=1.9154 STOI=0.9064 |