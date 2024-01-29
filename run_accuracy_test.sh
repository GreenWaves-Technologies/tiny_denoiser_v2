#!/bin/bash

nn_name=( "denoiser_LSTM_Valetini" "tt_denoiser_rank_80" "tt_denoiser_rank_48" "tt_denoiser_rank_16" "tt_denoiser_rank_8" "tt_denoiser_rank_4" "tt_denoiser_rank_2" "denoiser_GRU_dns")
quant_type=( "fp32" "fp16" "mixedfp16" "mixedne16fp16" )

for model_name in "${nn_name[@]}"
do
	echo "$model_name"
    for quant in "${quant_type[@]}"
    do
        echo "$quant"
        fp_32_arg=""
        quant_type_arg=""
        if [ $quant = "fp32" ]; then
            fp_32_arg="--float_exec_test"
            quant_type_arg="fp16"
        else
            quant_type_arg=$quant
        fi
        tt_arg=""
        model_arg=""
        if [[ $model_name = tt* ]]; then
            tt_arg="--tensor_train"
            model_arg="model/tensor_train/$model_name.onnx"
        else
            model_arg="model/$model_name.onnx"
        fi

        output=$(python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model ${model_arg} --quant_type ${quant_type_arg} --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ ${fp_32_arg} ${tt_arg} 2>&1)
        echo $output
    done
done
