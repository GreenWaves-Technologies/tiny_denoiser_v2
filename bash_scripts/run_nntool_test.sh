#!/bin/sh

model=denoiser_GRU_dns
quant_mode=bfp16
python nntool_scripts/test_nntool_model.py \
    --mode test_dataset \
    --trained_model model/$model.onnx \
    --quant_type $quant_mode \
    --noisy_dataset dataset/test/noisy/ \
    --clean_dataset dataset/test/clean/ \
    --output_dataset nntool_outputs/$model/$quant_mode/ \
    --verbose


# QTYPE   ovrl_mos  dovrl_mos  sig_mos  dsig_mos   bak_mos  dbak_mos  p808_mos  dp808_mos      pesq     dpesq      stoi    dstoi  meanerr  dmeanerr
# FP16    2.719404   0.103277  3.07319 -0.268312  3.752034  0.787279  3.084125   0.247209  2.075576  0.689891  0.901699  0.01218  0.96289 -3.087368
# BF16    2.726884   0.110757  3.08651 -0.254999  3.744666  0.779911  3.081411   0.244495  2.044378  0.658693  0.902197  0.01268  0.96590 -3.084353
