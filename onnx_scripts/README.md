

# passing a folder containing noisy files:

```
python ./test_onnx_vs_nntool_tt_models.py --samples_dir ~/denoiser/samples/dataset/noisy --rank 2 --models_dir ./tt_onnx_models --output_dir ./test_samples
```

# passing a single noisy file:

```
python ./test_onnx_vs_nntool_tt_models.py --noisy_file_path ./noisy_signal.wav --rank 2 --models_dir ./tt_onnx_models --output_dir ./test_samples
```