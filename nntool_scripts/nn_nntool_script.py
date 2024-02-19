import argparse
import os
import pickle
from glob import glob
import argcomplete
from denoiser_nntool_utils import get_astats
from nntool.api import NNGraph
from nntool.api.types import RNNNodeBase, LSTMNode
from nntool.api.utils import model_settings, quantization_options


def create_model_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='nntool_script')

    # Model options
    parser.add_argument('--trained_model', required=True,
                        help="Path to the trained tflite/onnx")
    parser.add_argument('--quant_dataset', default="dataset/quant/*",
                        help="path to .wav files to use to quantize the network")
    parser.add_argument('--stats_pickle', default=None,
                        help="pickle file where to store the statistics or get statistics if already saved")
    parser.add_argument('--requantize', action="store_true",
                        help="even if the stats pickle file exists, requantize the NN anyway")
    parser.add_argument('--quant_type', default="mixedfp16", choices=["mixedfp16", "mixedne16fp16", "fp16", "8x8_sq8", "8x8_ne16", "16x8_ne16"],
                        help="Quantization options")
    parser.add_argument('--tensor_train', action="store_true",
                        help="If the model is tensor train")
    # At options (mode = performance or generate_at_model)
    parser.add_argument('--tensors_dir', default="tensors",
                        help="Where nntool stores the weights/bias tensors dir (only used in generate and performance mode)")
    parser.add_argument('--at_model_path', default="nn_model.c",
                        help="Path to the C autotiler model file to generate (only used in generate mode)")
    parser.add_argument('--flash_type', default="flash", choices=["flash", "mram"],
                        help="Flash type")
    return parser

def get_states_idxs(G: NNGraph, tensor_train):
    if tensor_train:
        # states_idxs = [(out_node.step_idx, 0) for out_node in G.output_nodes() if out_node.name != "output_1"].sort(key=lambda x: x[0])
        states_idxs = [(G[f"output_{h}"].step_idx, 0) for h in range(2, 6)]
    else:
        rnn_nodes = [node for node in G.nodes(node_classes=RNNNodeBase, sort=True)]
        states_idxs = []
        for rnn_node in rnn_nodes:
            states_idxs.append((rnn_node.step_idx, 0))
            if isinstance(rnn_node, LSTMNode):
                states_idxs.append((rnn_node.step_idx, -1))
    return states_idxs

def build_nntool_graph(trained_model, quant_type, quant_dataset=None, stats_file=None, requantize=False, tensor_train=False) -> NNGraph:
    print(f"Building model with {quant_type} Quantization options")
    
    print(trained_model)
    
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
        quant_files = glob(quant_dataset)[:1]
        if len(quant_files) < 1:
            raise ValueError("Provide quant_dataset")
        if stats_file is None:
            stats_file = f"/tmp/{os.path.splitext(os.path.basename(trained_model))[0]}.pickle"
        if stats_file and os.path.exists(stats_file) and not requantize:
            print(f"Loading stats dictionary from {stats_file}")
            with open(stats_file, 'rb') as fp:
                stats = pickle.load(fp)
        else:
            stats = get_astats(G, quant_files, get_states_idxs(G, tensor_train))
            if stats_file:
                print(f"Saving stats dictionary to {stats_file}")
                with open(stats_file, 'wb') as fp:
                    pickle.dump(stats, fp, protocol=pickle.HIGHEST_PROTOCOL)

        node_opts = None

        if quant_type in ["mixedfp16", "mixedne16fp16"]:

            quant_opts = quantization_options(clip_type="none", allow_asymmetric_out=True, force_rnn_1_minus_1_out=True, use_ne16=quant_type == "mixedne16fp16")
            if tensor_train:
                node_opts = {
                    nname: quantization_options(scheme="FLOAT", float_type="float16")
                    for nname in [
                        "input_1",
                        # "input_2",
                        # "input_3",
                        # "input_4",
                        # "input_5",
                        # "expr_0",
                        # "expr_2", 
                        "_fc0_Conv_fusion",
                        "_fc0_Conv_reshape_in",
                        "_fc0_Conv_reshape_out",
                        "_fc1_Conv_fusion",
                        "_fc2_Conv_fusion",
                        "_fc2_Conv_reshape_out",
                        "output_1"
                        # "output_2",
                        # "output_3",
                        # "output_4",
                        # "output_5"
                    ]
                }
            else:
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
                        "output_1",
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

    if not tensor_train:
        for rnn_node in G.nodes(RNNNodeBase):
            rnn_node.set_states_as_inputs(G)
        

    states_idxs = get_states_idxs(G, tensor_train)
    # print("After quantization and fusion states output identified as:")
    # for state_idx in states_idxs:
    #    print(f"Node {G[state_idx[0]].name} ({state_idx[0]}) output {state_idx[1]}")
    return G, states_idxs

if __name__ == '__main__':
    parser = create_model_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    G, _ = build_nntool_graph(
        args.trained_model,
        args.quant_type,
        quant_dataset=args.quant_dataset,
        stats_file=args.stats_pickle,
        requantize=args.requantize,
        tensor_train=args.tensor_train
    )

    #G.draw(view=False, filepath="graph")
    #G.draw(view=False, filepath="graph_q", quant_labels=True)

    G.gen_at_model(
        write_constants=True,
        directory=os.path.split(args.at_model_path)[0],
        settings=model_settings(
            tensor_directory=args.tensors_dir,
            model_directory=os.path.split(args.at_model_path)[0],
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
    )
