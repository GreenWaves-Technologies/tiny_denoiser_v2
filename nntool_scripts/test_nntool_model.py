import argparse
import os

import argcomplete
import numpy as np
import soundfile as sf
from denoiser_nntool_utils import single_audio_inference
from denoiser_utils import open_wav, pesq_stoi, postprocessing, preprocessing
from nn_nntool_script import build_nntool_graph, create_model_parser
from tqdm import tqdm

def extend_parser_for_test(main_parser: argparse.ArgumentParser):
    main_parser.add_argument('--mode', default="test_dataset", choices=["test_sample", "test_dataset"],
                             help="Script mode")
    # mode = test_sample
    main_parser.add_argument('--test_sample', default="dataset/test/noisy/p232_050.wav",
                             help=".wav file to use for testing")
    main_parser.add_argument('--clean_test_sample', default=None,
                             help="path to the clean version of the test_sample to check pesq/stoi metrics (if not provided it will not be used)")
    main_parser.add_argument('--float_exec_test', action="store_true",
                             help="run the model with floating point backend")
    main_parser.add_argument('--out_wav', default="output_nn.wav",
                             help="path to the cleaned wav file")
    # mode = test_dataset
    main_parser.add_argument('--noisy_dataset', default=None,
                             help="path to folder of noisy samples")
    main_parser.add_argument('--clean_dataset', default=None,
                             help="path to folder of clean samples")
    main_parser.add_argument('--output_dataset', default=None,
                             help="path to folder of output samples")
    main_parser.add_argument('--dns_dataset', action="store_true",
                             help="slect correct names for dns dataset")
    main_parser.add_argument('--verbose', action="store_true",
                             help="run the inference and print pesq/stoi for every file")
    return main_parser


def calculate_pesq_stoi(estimate, clean_data):
    sz0 = clean_data.shape[0]
    sz1 = estimate.shape[0]
    if sz0 > sz1:
        estimate = np.pad(estimate, (0,sz0-sz1))
    else:
        estimate = estimate[:sz0]
    pesq_i, stoi_i = pesq_stoi(clean_data, estimate, 16000)
    return pesq_i, stoi_i

if __name__ == '__main__':
    model_parser = create_model_parser()
    parser = extend_parser_for_test(model_parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    G = build_nntool_graph(
        args.trained_model,
        args.quant_type,
        quant_dataset=args.quant_dataset,
        stats_file=args.stats_pickle,
        requantize=args.requantize
    )

    if args.mode == "test_sample":
        stft = preprocessing(args.test_sample).T
        stft_out = single_audio_inference(G, stft, quant_exec=not args.float_exec_test)
        estimate = postprocessing(stft_out.T)

        if args.clean_test_sample:
            clean_data = open_wav(args.clean_test_sample)
            pesq_i, stoi_i = calculate_pesq_stoi(estimate, clean_data)
            print(f"pesq=\t{pesq_i}\tand stoi=\t{stoi_i}")

        sf.write(args.out_wav, estimate, 16000)

    elif args.mode == "test_dataset":
        print(f"Testing on dataset: {args.noisy_dataset}")
        files = os.listdir(args.noisy_dataset)
        pesq = 0
        stoi = 0
        for c, filename in enumerate(tqdm(files)):
            noisy_file = os.path.join(args.noisy_dataset, filename)
            stft = preprocessing(noisy_file)

            stft_frame_i_T = np.transpose(stft) # swap the axis to select the tmestamp
            stft_frame_o_T = single_audio_inference(G, stft_frame_i_T, quant_exec=not args.float_exec_test, stats_collector=None, disable_tqdm=True)

            estimate = postprocessing(stft_frame_o_T.T)

            # compute the metrics
            if args.dns_dataset:
                clean_filename = "clean_fileid_" + filename.split("_")[-1]
            else:
                clean_filename = filename

            clean_file = os.path.join(args.clean_dataset, clean_filename)
            clean_data = open_wav(clean_file)
            pesq_i, stoi_i = calculate_pesq_stoi(estimate, clean_data)
            if args.verbose:
                print(f"Sample ({c}/{len(files)})\t{filename}\twith pesq=\t{pesq_i}\tand stoi=\t{stoi_i}")

            pesq += pesq_i
            stoi += stoi_i
            if ((c+1) % 10) == 0:
                print(f"After {c+1} files: PESQ={pesq/(c+1):.4f} STOI={stoi/(c+1):.4f}")

            if args.output_dataset:
                output_file = os.path.join(args.output_dataset, filename)
                # Write out audio as 24bit PCM WAV
                sf.write(output_file, estimate, 16000)

        print(f"Result of accuracy on the ({len(files)} samples):")
        print(f"PESQ: {pesq / len(files)}")
        print(f"STOI: {stoi / len(files)}")
