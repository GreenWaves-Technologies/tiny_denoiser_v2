import argparse
import os

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
from denoiser_utils import preprocessing
from nn_nntool_script import build_nntool_graph, create_model_parser
from nntool.api.types import LSTMNode, RNNNodeBase
from nntool.api.utils import model_settings

def extend_parser_for_perf(main_parser: argparse.ArgumentParser):
    main_parser.add_argument('--test_sample', default="dataset/test/noisy/p232_050.wav",
                             help=".wav file to use for testing")
    return main_parser

if __name__ == '__main__':
    parser = create_model_parser()
    parser = extend_parser_for_perf(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    G, states_idx = build_nntool_graph(
        args.trained_model,
        args.quant_type,
        quant_dataset=args.quant_dataset,
        stats_file=args.stats_pickle,
        requantize=args.requantize,
        tensor_train=args.tensor_train
    )

    G.draw(view=False, filepath="graph")
    G.draw(view=False, filepath="graph_q", quant_labels=True)

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
        settings=model_settings(
            tensor_directory=args.tensors_dir,
            model_directory=os.path.split(args.at_model_path)[0],
            model_file=os.path.split(args.at_model_path)[1],
            l3_ram_device="AT_MEM_L3_DEFAULTRAM",
            l3_flash_device="AT_MEM_L3_DEFAULTFLASH" if args.flash_type == "flash" else "AT_MEM_L3_MRAMFLASH",
            basic_kernel_header_file="NN_Expression_Kernels.h",
            basic_kernel_source_file="NN_Expression_Kernels.c",
            l2_size=1000000,
            l1_size=128000,
            # graph_l1_promotion=2,
            graph_warm_construct=1,
            graph_size_opt=2,
            graph_const_exec_from_flash=True,
            # graph_group_weights=True,
            graph_async_fork=True,
            graph_monitor_cycles=False,
            graph_produce_node_names=False,
            graph_produce_operinfos=False,
            # privileged_l3_flash_device="AT_MEM_L3_MRAMFLASH",
            # privileged_l3_flash_size=int(1.8*1024*1024)
        )
    )
    assert res.returncode == 0, "Something went wrong"
    res.plot_memory_boxes()
    plt.show()
