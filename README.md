# TinyDenoiser on GAP9

This project demonstrates a Recurrent Neural Network (RNN) based method for Speech Enhamencement on GAP9. Refer to the [original paper](https://arxiv.org/abs/2210.07692) for more details.

The main loop of the application continuosly samples data from the microphone at 16kHz, applies the RNN filter and reconstruct the cleaned signal via overlap-and-add. As depitcted in the Figure below, the nosiy signal is windowed (frame size of 25 msec with an hop lenght of 6.25 msec and Hanning windowing) and the STFT is computed. The RNN is fed with the magnitude of the STFT components and return a suppression mask. After weighting, the inverse STFT returns a cleaned audio clip.

![alt text](imgs/TinyDenoiser.png?raw=true "Title")

## Demo Getting Started
**Demo Mode**
The demo runs on the **GAP9 Audio EVK**, using the microphone of the GAPmod board.
```
cmake -B build
cmake --build build --target run
```

**GvSoc - gvcontrol**
To run the **Demo mode** on GVSoC you can use the `gvcontrol` file. The gvcontrol is used to send/read data to/from the i2s interface of the gap9 gvsoc. You can chose the input noisy wav file you want to process. Since gap is waiting for pdm data, the pcm/pdm convertion module of gvsoc is used. Beware that this mode is very slow (~10 minutes to run the whole execution on a small file). Select GVSoc as target platform with the **proxy option enabled**, then, in another terminal:
```
./gvcontrol --wav_in ../dataset/test_48kHz/noisy/p232_170.wav --wav_out ../output.wav
```

**NOTE**:
1. the wav_in/wav_out paths passed to gvcontrol script are relative to the build directory not your current directory.
2. the passed wav files must be in 48kHz because the main_demo.c will open the i2s interfaces with 48kHz settings and the SFU will be responsible to down/up-sample.

**DenoiseWav Mode**
Optionally, the application can run on GVSOC (or board) to denoise a custom audio file (.wav).
```
cmake -B build
cmake --build build --target menuconfig # Select the options DenoiseWav in the DENOISER APP -> Application mode menu
cmake --build build --target run
```
Output wav file will be written to denoised_output.wav inside the project folder.

## Project Structure
* `main_demo.c`: is the main file, including the application code (*Demo Mode*)
* `main_wav.c`: is the app file when using the DenoiseWav option (*DenoiseWav*)
* `cluster_fn.c`: contains the src code of the effective algorithm running on the cluster of GAP9. Both `main_demo.c` and `main_wav.c` use this src functions.
* `model/` includes the pretrained models converted to onnx and ready to be fed to GAPflow for NN model code generation: 
    * `denoiser_GRU_dns.onnx` is a GRU based models trained on the [DNS][dns] dataset. (Used by default)
    * `denoiser_LSTM_valentini.onnx` and `denoiser_GRU_valentini.onnx` are respectively LSTM and GRU models trained on the [Valentini][valentini]. they are used for testing purpose.
* `nntool_scripts/`: for more details refer to [Python Utilities](#python-utilities) 
    * `nn_nntool_script.py`: includes the nntool recipes to quantize the LSTM or GRU models and prepare for deployment (`build_nntool_graph` fucntion) then generate the Autotiler code. You can refer to the [quantization section](#quantization) for more details.
    * `fft_nntool_script.py`: it contains the scripts to generate the Autotiler pre/post-processing models for the FFT/iFFT.
    * `test_onnx.py`: test the quality of the original onnx models on a single sample or an entire dataset with onnxruntime.
    * `test_nntool_model.py`: test the quality of the deployable models on a single sample or an entire dataset with NNTool bit-accurate backend wrt the target computation.
    * `test_nntool_model_perf.py`: test the performance (cycles) of the deployable models (used for fastly prototype new architectures / quantization schemes and see real target performance).
* `dataset/`: contains the audio samples for testing and quantization claibration
* `SFUGraph.src`: is the configuation file for Audio IO. It is used only for board target.

## Quantization
The Post-Training quantization process of the RNN model is operated by the GAPflow.
Both LSTM and GRU models can be quantized using one of the different options:
* `FP16`: quantizing both activations and weights to _float16_ format. This does not require any calibration samples.
* `FP16MIXED`: only RNN layers are quantized to 8 bits, while the rest is kept to FP16. This option achives the **best** trade-off between accuracy degration and inference speed.
* `FP16NE16MIXED`: equal to `FP16MIXED` but the 8bits layers are deployed to the HW accelerator available in GAP9.

## Python Utilities
In the `nntool_scripts` folder there are several python utilities to test the quality and the performance of the deployable model in full python environment through NNTool. All scripts can be run with --help argument to check available options.

NOTE: `nntool_scripts/test_nntool_model.py` can be run with --float_exec_test, in this case, whatever quantization scheme selected, the execution on NNTool backend will be run with fp32 precision and yields to almost identical results wrt onnx execution.

### Denoise a wav file

You can test the quality of the deployable models with python scripting:

**NNTOOL**:
```
python nntool_scripts/test_nntool_model.py --mode test_sample --trained_model model/denoiser_dns.onnx --quant_type fp16 --test_sample dataset/test/noisy/p232_050.wav --out_wav output_nntool.wav
```

**ONNX** (note that the models must be stateful to work on onnx):
```
python nntool_scripts/test_onnx.py --mode test_sample --trained_model model/denoiser_dns.onnx --test_sample dataset/test/noisy/p232_050.wav --out_wav output_nntool.wav
```

The output is saved in a file called `output_nntool.wav` in the home of the repository

### Test on dataset

**NNTOOL**:
```
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_dns.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/
```

**ONNX** (note that the models must be stateful to work on onnx):
```
python nntool_scripts/test_onnx.py --mode test_dataset --trained_model model/denoiser_dns.onnx --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/
```


## Results:
| Path                         | Type        | Trained on  | Pytorch                 | NNTool fp32             | NNTool fp16             | NNTool MixedFp16        | NNTool Mixedne16Fp16    |
|------------------------------|-------------|-------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| denoiser_LSTM_Valentini.onnx | LSTM        | Valentini   | PESQ=       STOI=       | PESQ=2.2189 STOI=0.9091 | PESQ=2.2175 STOI=0.9091 | PESQ=2.1887 STOI=0.9054 | PESQ=2.2196 STOI=0.9064 |
| denoiser_GRU_dns.onnx        | GRU         | DNS         | PESQ=       STOI=       | PESQ=2.0486 STOI=0.9007 | PESQ=2.0468 STOI=0.9010 | PESQ=1.9590 STOI=0.8922 | PESQ=2.0158 STOI=0.8983 |
| tt_denoiser_rank_80.onnx     | LSTM TT R80 | DNS         | PESQ=       STOI=       | PESQ=2.5511 STOI=0.8878 | PESQ=2.5961 STOI=0.8825 | PESQ=2.4559 STOI=0.8908 | PESQ= STOI=             |
| tt_denoiser_rank_48.onnx     | LSTM TT R48 | DNS         | PESQ=       STOI=       | PESQ=2.2710 STOI=0.8893 | PESQ=2.2712 STOI=0.8881 | PESQ=1.8841 STOI=0.8917 | PESQ=1.9436 STOI=0.8913 |
| tt_denoiser_rank_16.onnx     | LSTM TT R16 | DNS         | PESQ=       STOI=       | PESQ=2.4287 STOI=0.8914 | PESQ=2.4137 STOI=0.8908 | PESQ=2.2953 STOI=0.8901 | PESQ=2.2886 STOI=0.8962|
| tt_denoiser_rank_8.onnx      | LSTM TT R8  | DNS         | PESQ=       STOI=       | PESQ=2.5557 STOI=0.8925 | PESQ=2.5539 STOI=0.8917 | PESQ=2.3030 STOI=0.8904 | PESQ=2.3674 STOI=0.8992 |
| tt_denoiser_rank_4.onnx      | LSTM TT R4  | DNS         | PESQ=       STOI=       | PESQ=2.2942 STOI=0.8831 | PESQ=2.2866 STOI=0.8831 | PESQ=2.1710 STOI=0.8915 | PESQ=2.2079 STOI=0.8904 |
| tt_denoiser_rank_2.onnx      | LSTM TT R2  | DNS         | PESQ=       STOI=       | PESQ=2.3431 STOI=0.8822 | PESQ=2.3414 STOI=0.8796 | PESQ=2.2362 STOI=0.8874 | PESQ=2.2737 STOI=0.8864 |

## Results on small Valentini Dataset:

These results have been obtained with `bash_scripts/run_nntool_test.sh` for nntool and `bash_scripts/run_gvsoc_test.sh` for gvsoc. In gvsoc the application has been run in DenoiseWav mode.

**denoiser_GRU_dns.onnx BFP16**
|      filename |  NNTool PESQ  | GVSOC PESQ | NNTool STOI | GVSOC STOI |
|---------------|:-------------:|:----------:|:-----------:|:----------:|
|  p257_403.wav |     2.121222  |   2.048298 |    0.913772 |   0.930076 |
|  p232_170.wav |     1.872853  |   1.837506 |    0.780692 |   0.765858 |
|  p257_114.wav |     1.149774  |   1.174865 |    0.832479 |   0.833701 |
|  p257_058.wav |     2.203916  |   2.138830 |    0.955514 |   0.952348 |
|  p257_259.wav |     2.065216  |   2.035814 |    0.862383 |   0.843457 |
|  p232_116.wav |     2.233573  |   2.245732 |    0.929711 |   0.925160 |
|  p257_062.wav |     2.074991  |   1.844434 |    0.919069 |   0.901747 |
|  p232_334.wav |     2.862745  |   2.719331 |    0.981879 |   0.982893 |
|  p232_050.wav |     2.043524  |   2.057978 |    0.922732 |   0.919254 |
|  p232_254.wav |     1.815302  |   1.970535 |    0.923826 |   0.923854 |
|  **OVERALL**  |   **2.04**    |  **2.01**  |  **0.9022** | **0.8974** |

**denoiser_GRU_dns.onnx FP16**
|      filename |  NNTool PESQ  | GVSOC PESQ | NNTool STOI | GVSOC STOI |
|---------------|:-------------:|:----------:|:-----------:|:----------:|
|  p257_403.wav |     2.173632  |   2.076354 |    0.915731 |   0.935242 |
|  p232_170.wav |     1.945011  |   1.952889 |    0.779314 |   0.770128 |
|  p257_114.wav |     1.153799  |   1.175280 |    0.830670 |   0.831067 |
|  p257_058.wav |     2.214221  |   2.202593 |    0.956876 |   0.953315 |
|  p257_259.wav |     2.098329  |   2.067579 |    0.855297 |   0.835670 |
|  p232_116.wav |     2.256996  |   2.220318 |    0.930446 |   0.925938 |
|  p257_062.wav |     2.098467  |   2.025881 |    0.918559 |   0.913390 |
|  p232_334.wav |     2.874148  |   2.697017 |    0.981922 |   0.980854 |
|  p232_050.wav |     2.050680  |   2.016107 |    0.923126 |   0.925443 |
|  p232_254.wav |     1.888132  |   2.077618 |    0.924739 |   0.924963 |
|  **OVERALL**  |   **2.07**    |  **2.05**  |  **0.9017** | **0.8996** |

## Citing

For more insights and on how the models were trained, refer to:

```BibTex
@misc{rusci2022accelerating,
      title={Accelerating RNN-based Speech Enhancement on a Multi-Core MCU with Mixed FP16-INT8 Post-Training Quantization}, 
      author={Manuele Rusci and Marco Fariselli and Martin Croome and Francesco Paci and Eric Flamand},
      year={2022},
      eprint={2210.07692},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

[dns]: https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2020/
[valentini]: https://datashare.ed.ac.uk/handle/10283/2791
