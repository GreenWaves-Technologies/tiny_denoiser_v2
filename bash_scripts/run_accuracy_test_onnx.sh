#!/bin/sh

# nn_name=( "FullRankLSTM_Stateful" "tt_denoiser_rank_80" "tt_denoiser_rank_48" "tt_denoiser_rank_16" "tt_denoiser_rank_8" "tt_denoiser_rank_4" "tt_denoiser_rank_2" )
nn_name=( "tt_denoiser_rank_48" "tt_denoiser_rank_16" "tt_denoiser_rank_8" "tt_denoiser_rank_4" "tt_denoiser_rank_2" )

##################################################
# Choose dataset to test
##################################################
# VALENTINI (Subset 10 samples)
# noisy_dataset="dataset/test/noisy/"
# clean_dataset="dataset/test/clean/"
# csv_file="results_on_dataset/onnx_valentini_10samples"

# LIBRIMIX (Subset 31 samples)
# noisy_dataset="/home/marco/Dataset/test_Libri2Mix/subset_all_snr/mix_single"
# clean_dataset="/home/marco/Dataset/test_Libri2Mix/subset_all_snr/s1"
# csv_file=results_on_dataset/nntool_librimix_subset_all_snr

# LIBRIMIX (FULL)
# noisy_dataset="/home/marco/Dataset/test_Libri2Mix/full/mix_single"
# clean_dataset="/home/marco/Dataset/test_Libri2Mix/full/s1"
# csv_file=results_on_dataset/nntool_librimix_full

# DNS (FULL)
noisy_dataset="../DNS-Challenge/V5_BlindTestSet_mono/Track1_Headset/noisy/"
clean_dataset="XX"
csv_file=results_on_dataset/nntool_dns5_blind

for model_name in "${nn_name[@]}"
do
	echo "$model_name"

    tt_arg=""
    model_arg=""
    if [[ $model_name = tt* ]]; then
        model_arg="model/tensor_train/$model_name.onnx"
    else
        model_arg="model/$model_name.onnx"
    fi

    python3 nntool_scripts/test_onnx.py --output_dataset dataset/${model_name}_onnx/ --csv_file=${csv_file}.csv --csv_file_allfiles ${csv_file}_allfiles.csv --mode test_dataset --trained_model ${model_arg} --noisy_dataset ${noisy_dataset} --clean_dataset ${clean_dataset}
done
