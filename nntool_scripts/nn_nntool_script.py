import argparse
import os
import pickle
from glob import glob
from tqdm import tqdm

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from denoiser_nntool_utils import get_astats, single_audio_inference
from denoiser_utils import open_wav, pesq_stoi, postprocessing, preprocessing
from nntool.api import NNGraph
from nntool.api.utils import model_settings, quantization_options
from nntool.graph.types import LSTMNode, RNNNodeBase

def calculate_pesq_stoi(estimate, clean_data, verbose=False):
    sz0 = clean_data.shape[0]
    sz1 = estimate.shape[0]
    if sz0 > sz1:
        estimate = np.pad(estimate, (0,sz0-sz1))
    else:
        estimate = estimate[:sz0]
    pesq_i, stoi_i = pesq_stoi(clean_data, estimate, 16000)
    if verbose:
        print(f"Sample ({c}/{len(files)})\t{filename}\twith pesq=\t{pesq_i}\tand stoi=\t{stoi_i}")
    return pesq_i, stoi_i


def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='nntool_script')

    parser.add_argument('--mode', default="generate_at_model", choices=["test_sample", "test_dataset", "performance", "generate_at_model"],
                        help="Script mode")
    # Model options
    parser.add_argument('--trained_model', required=True,
                        help="Path to the trained tflite/onnx")
    parser.add_argument('--quant_dataset', default="dataset/quant/*",
                        help="path to .wav files to use to quantize the network")
    parser.add_argument('--stats_pickle', default="/tmp/denoiser_stats.pickle",
                        help="pickle file where to store the statistics or get statistics if already saved")
    parser.add_argument('--requantize', action="store_true",
                        help="even if the stats pickle file exists, requantize the NN anyway")
    parser.add_argument('--quant_type', default="mixedfp16", choices=["mixedfp16", "mixedne16fp16", "fp16", "8x8_sq8", "8x8_ne16", "16x8_ne16"],
                        help="Quantization options")
    # mode = test_sample
    parser.add_argument('--test_sample', default="dataset/test/noisy/p232_050.wav",
                        help=".wav file to use for testing")
    parser.add_argument('--clean_test_sample', default=None,
                        help="path to the clean version of the test_sample to check pesq/stoi metrics (if not provided it will not be used)")
    parser.add_argument('--float_exec_test', action="store_true",
                        help="run the model with floating point backend")
    parser.add_argument('--out_wav', default="output_nn.wav",
                        help="path to the cleaned wav file")
    # mode = test_dataset
    parser.add_argument('--noisy_dataset', default=None,
                        help="path to folder of noisy samples")
    parser.add_argument('--clean_dataset', default=None,
                        help="path to folder of clean samples")
    parser.add_argument('--output_dataset', default=None,
                        help="path to folder of output samples")
    parser.add_argument('--dns_dataset', action="store_true",
                        help="slect correct names for dns dataset")
    parser.add_argument('--verbose', action="store_true",
                        help="run the inference and print pesq/stoi for every file")

    # At options (mode = performance or generate_at_model)
    parser.add_argument('--tensors_dir', default="tensors",
                        help="Where nntool stores the weights/bias tensors dir (only used in generate and performance mode)")
    parser.add_argument('--at_model_path', default="nn_model.c",
                        help="Path to the C autotiler model file to generate (only used in generate mode)")
    parser.add_argument('--flash_type', default="flash", choices=["flash", "mram"],
                        help="Flash type")
    return parser

def build_nntool_graph(trained_model, quant_type, quant_dataset=None, stats_file=None, requantize=False) -> NNGraph:
    print(f"Building model with {quant_type} Quantization options")

    G = NNGraph.load_graph(trained_model, old_dsp_lib=False)
    G.name = "tinydenoiser"
    G.adjust_order()
    G.fusions('scaled_match_group')

    if quant_type == "fp16":
        G.quantize(
            graph_options=quantization_options(
                scheme="FLOAT", float_type="float16"
            )
        )
    else:
        quant_files = glob(quant_dataset)[:5]
        if len(quant_files) < 1:
            raise ValueError("Provide quant_dataset")
        if stats_file and os.path.exists(stats_file) and not requantize:
            print(f"Loading stats dictionary from {stats_file}")
            with open(stats_file, 'rb') as fp:
                stats = pickle.load(fp)
        else:
            stats = get_astats(G, quant_files)
            if stats_file:
                print(f"Saving stats dictionary to {stats_file}")
                with open(stats_file, 'wb') as fp:
                    pickle.dump(stats, fp, protocol=pickle.HIGHEST_PROTOCOL)

        node_opts = None

        if quant_type in ["mixedfp16", "mixedne16fp16"]:
            quant_opts = quantization_options(clip_type="none", allow_asymmetric_out=True, force_rnn_1_minus_1_out=True, use_ne16=quant_type == "mixedne16fp16")
            node_opts = {
                nname: quantization_options(scheme="FLOAT", float_type="float16")
                for nname in [
                    "input_1",
                    "Conv_0_reshape_in",
                    "Conv_0_fusion",
                    "Conv_147_fusion",
                    "Conv_150_fusion",
                    "Conv_150_reshape_out",
                    "Conv_139_fusion",
                    "Conv_142_fusion",
                    "Conv_142_reshape_out",
                    "Sigmoid_151",
                    "output_1"
                ]
            }
        elif quant_type == "8x8_sq8":
            quant_opts = quantization_options(clip_type="none", allow_asymmetric_out=True, force_rnn_1_minus_1_out=True)

        elif quant_type == "8x8_ne16":
            quant_opts = quantization_options(clip_type="none", allow_asymmetric_out=True, force_rnn_1_minus_1_out=True, use_ne16=True)

        elif quant_type == "16x8_ne16":
            quant_opts = quantization_options(clip_type="none", allow_asymmetric_out=True, force_rnn_1_minus_1_out=True, use_ne16=True, force_input_size=16, force_external_size=16, force_output_size=16)

        G.quantize(
            statistics=stats,
            graph_options=quant_opts,
            node_options=node_opts
        )

    return G


if __name__ == '__main__':
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if args.mode == "test_dataset":
        if not args.clean_dataset or not args.noisy_dataset:
            raise ValueError("In test_dataset mode you need to provide clean_dataset and noisy_dataset")

    G = build_nntool_graph(
        args.trained_model,
        args.quant_type,
        quant_dataset=args.quant_dataset,
        stats_file=args.stats_pickle,
        requantize=args.requantize
    )
    for rnn_node in G.nodes(RNNNodeBase):
        rnn_node.set_states_as_inputs(G)

    G.draw(view=False, filepath="graph")
    G.draw(view=False, filepath="graph_q", quant_labels=True)

    if args.mode == "test_sample":
        stft = preprocessing(args.test_sample).T
        stft_out = single_audio_inference(G, stft, quant_exec=not args.float_exec_test)
        estimate = postprocessing(stft_out.T)

        if args.clean_test_sample:
            clean_data = open_wav(args.clean_test_sample)
            pesq_i, stoi_i = calculate_pesq_stoi(estimate, clean_data, True)

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
            pesq_i, stoi_i = calculate_pesq_stoi(estimate, clean_data, args.verbose)

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

    else:
        model_build_dir = os.path.split(args.at_model_path)[0]
        at_model_settings = model_settings(
            tensor_directory=args.tensors_dir,
            model_directory=model_build_dir,
            model_file=os.path.split(args.at_model_path)[1],
            l3_ram_device="AT_MEM_L3_DEFAULTRAM",
            l3_flash_device="AT_MEM_L3_DEFAULTFLASH" if args.flash_type == "flash" else "AT_MEM_L3_MRAMFLASH",
            basic_kernel_header_file="NN_Expression_Kernels.h",
            basic_kernel_source_file="NN_Expression_Kernels.c",
            l2_size=1200000,
            l1_size=128000,
            graph_l1_promotion=2,
            graph_warm_construct=1,
            graph_size_opt=2,
            graph_const_exec_from_flash=True,
            graph_group_weights=True,
            graph_async_fork=True,
            graph_monitor_cycles=False,
            graph_produce_node_names=False,
            graph_produce_operinfos=False,
            # privileged_l3_flash_device="AT_MEM_L3_MRAMFLASH",
            # privileged_l3_flash_size=int(1.8*1024*1024)
        )

        if args.mode == "performance":
            stft = preprocessing(args.test_sample)
            stft_frame_i_T = np.transpose(stft) # swap the axis to select the tmestamp

            #init lstm to zeros
            rnn_nodes = [node for node in G.nodes(node_classes=RNNNodeBase, sort=True)]
            rnn_states = []
            for rnn_node in rnn_nodes:
                rnn_states.append(np.zeros(rnn_node.out_dims[0].size()))
                if isinstance(rnn_node, LSTMNode):
                    rnn_states.append(np.zeros(rnn_node.out_dims[0].size()))

            # take a frame in the middle
            stft_clip = stft_frame_i_T[10]
            stft_clip_mag = np.abs(stft_clip)

            data = [stft_clip_mag, *rnn_states]
            qout = G.execute(data, dequantize=False, quantize=True)

            res = G.execute_on_target(
                input_tensors=[qout[in_node.step_idx][0] for in_node in G.input_nodes()],
                check_on_target=True,
                tolerance=0.04 if "fp16" in args.quant_type else 0,
                directory="test_run",
                print_output=True,
                at_loglevel=2,
                settings=at_model_settings
            )
            assert res.returncode == 0, "Something went wrong"
            res.plot_memory_boxes()
            plt.show()

        elif args.mode == "generate_at_model":

            G.gen_at_model(
                write_constants=True,
                directory=os.path.split(args.at_model_path)[0],
                settings=at_model_settings
            )
