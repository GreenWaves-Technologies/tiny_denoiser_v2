#!/bin/bash

#nn_name=( "denoiser_LSTM_Valetini" "tt_denoiser_rank_2" )
#quant_type=( "fp32" "fp16" )
#quant_type=( "fp32" "fp16" "mixedne16fp16")
#quant_type=( "mixedne16fp16" )
#quant_type=( "fp32" "fp16" "mixedfp16" "mixedne16fp16" "8x8_ne16" "16x8_ne16" "8x8_sq8")
#quant_type=( "fp32" "fp16" "16x8_ne16" "8x8_sq8" )


nn_name=( "denoiser_LSTM_Valetini" "tt_denoiser_rank_80" "tt_denoiser_rank_48" "tt_denoiser_rank_16" "tt_denoiser_rank_8" "tt_denoiser_rank_4" "tt_denoiser_rank_2" "denoiser_GRU_dns")
quant_type=( "fp32" "fp16" "mixedfp16" "mixedne16fp16")


##################################################
# Choose dataset to test
##################################################

## FULL DATASET
# noisy_dataset="dataset/test/noisy/"
# clean_dataset="dataset/test/clean/"

## Subset ~30 samples
# noisy_dataset="dataset/test_Libri2Mix/subset_all_snr/mix_single/"
# clean_dataset="dataset/test_Libri2Mix/subset_all_snr/s1/"

## DIVIDED By SNR

# noisy_dataset="dataset/test_Libri2Mix/-8_-6db/mix_single/"
# clean_dataset="dataset/test_Libri2Mix/-8_-6db/s1/"


# noisy_dataset="dataset/test_Libri2Mix/-5_-1db/mix_single/"
# clean_dataset="dataset/test_Libri2Mix/-5_-1db/s1/"

# noisy_dataset="dataset/test_Libri2Mix/0_4db/mix_single/"
# clean_dataset="dataset/test_Libri2Mix/0_4db/s1/"

# noisy_dataset="dataset/test_Libri2Mix/5_9db/mix_single/"
# clean_dataset="dataset/test_Libri2Mix/5_9db/s1/"

# noisy_dataset="dataset/test_Libri2Mix/10_14db/mix_single/"
# clean_dataset="dataset/test_Libri2Mix/10_14db/s1/"

noisy_dataset="dataset/test_Libri2Mix/15_19db/mix_single/"
clean_dataset="dataset/test_Libri2Mix/15_19db/s1/"


##################################################
#if test_sample you can choose a sample to run
##################################################

# clean_test_sample="dataset/test_Libri2Mix/test/s1/61-70968-0004_1188-133604-0014.wav"
# noise_test_sample="dataset/test_Libri2Mix/test/mix_single/61-70968-0004_1188-133604-0014.wav"

clean_test_sample="dataset/test/clean/p232_050.wav"
noise_test_sample="dataset/test/noisy/p232_050.wav"

file="dataset_results_new.csv"

if [ -f "$file" ] ; then
    rm "$file"
    touch $file
else
    touch $file
fi

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

        #python3 nntool_scripts/test_nntool_model.py --mode test_sample  --clean_test_sample ${clean_test_sample} --trained_model ${model_arg} --quant_type ${quant_type_arg} --test_sample ${noise_test_sample} --out_wav output_nntool_${model_name}_${quant}.wav ${fp_32_arg} ${tt_arg} 
        #python3 nntool_scripts/test_nntool_model.py --csv_file=${file} --output_dataset="reports/dataset_out" --mode test_dataset --trained_model ${model_arg} --quant_type ${quant_type_arg} --noisy_dataset ${noisy_dataset} --clean_dataset ${clean_dataset} ${fp_32_arg} ${tt_arg}
        python3 nntool_scripts/test_nntool_model.py --csv_file=${file}  --mode test_dataset --trained_model ${model_arg} --quant_type ${quant_type_arg} --noisy_dataset ${noisy_dataset} --clean_dataset ${clean_dataset} ${fp_32_arg} ${tt_arg}
    done
done
