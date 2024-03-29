#!/bin/sh

noisy_dir=dataset/test/noisy
clean_dir=dataset/test/clean

output_dir=dataset/test/gvsoc_outputs/bfp16
mkdir -p $output_dir

for file_path in "$noisy_dir"/*
do
    file_name=$(basename ${file_path})
    clean_file=$clean_dir/$file_name
    echo $file_path $file_name
    cmake -B build -DCONFIG_DENOISE_WAV=y \
        -DCONFIG_WAV_FILE_TO_DENOISE=$file_name \
        -DCONFIG_BFP16=y
    cmake --build build --target run -j
    cp build/denoised_output.wav $output_dir/$file_name 
    python nntool_scripts/eval_pesq.py --noisy_file build/denoised_output.wav --clean_file $clean_file
done
