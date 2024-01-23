import os
from nntool.api import NNGraph
from nntool.api.utils import model_settings
from nntool.graph.types import RFFT2DPreprocessingNode, IRFFT2DPreprocessingNode
from nntool.graph.dim import Dim
from nntool.graph.types.base import NNEdge
import argparse
import argcomplete

def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='fft_at_generators')

    parser.add_argument('--float_type', default="float32",
                        help="Float data type")
    parser.add_argument('--n_fft', default=512, type=int,
                        help="number of fft points")
    parser.add_argument('--frame_size', default=400, type=int,
                        help="number of fft points")
    parser.add_argument('--frame_step', default=100, type=int,
                        help="number of fft points")
    parser.add_argument('--window_type', default="hanning",
                        help="windowing function")
    parser.add_argument('--forward_at_model_path', default=None,
                        help="Path to the C autotiler model file to generate")
    parser.add_argument('--inverse_at_model_path', default=None,
                        help="Path to the C autotiler model file to generate")
    parser.add_argument('--forward_tensors_dir', default=None,
                        help="Path to the autotiler model constant files to generate")
    parser.add_argument('--inverse_tensors_dir', default=None,
                        help="Path to the autotiler model constant files to generate")
    parser.add_argument('--flash_type', default="flash", choices=["flash", "mram"],
                        help="Flash type")
    return parser

if __name__ == '__main__':
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    # NOTE: in librosa the frame length is always n_fft. The window_size can be less and will be padded centered to n_fft before multiplying to the frame values
    frame_size = args.frame_size
    n_fft = args.n_fft
    frame_step = args.frame_step
    window_type = args.window_type
    graph_opts={"scheme": "FLOAT", "float_type": args.float_type}

    ## Forward Graph
    G_fft_forward = NNGraph(name='fft_forward', old_dsp_lib=False)
    inp = G_fft_forward.add_input(Dim.unnamed([frame_size]))
    power = 0
    rfft_params = RFFT2DPreprocessingNode.librosa_stft(
        "rfft",
        win_length=frame_size,
        n_fft=n_fft,
        hop_length=frame_step,
        window=window_type,
        center=False
    )(inp)
    out = G_fft_forward.add_output()(rfft_params)

    G_fft_forward.add_dimensions()
    G_fft_forward.adjust_order()

    # Don't need real stats since we quantize to Float
    G_fft_forward.quantize(graph_options=graph_opts)
    res = G_fft_forward.gen_at_model(
        write_constants=True,
        settings=model_settings(
            l1_size=128000,
            l2_size=1300000,
            tensor_directory=args.forward_tensors_dir,
            model_directory=os.path.split(args.forward_at_model_path)[0],
            model_file=os.path.split(args.forward_at_model_path)[1],
            graph_l1_promotion=1,
            l3_flash_device="AT_MEM_L3_DEFAULTFLASH" if args.flash_type == "flash" else "AT_MEM_L3_MRAMFLASH",
            graph_monitor_cvar_name="FFT_Monitor",
            graph_produce_operinfos_cvar_name="FFT_Op",
            graph_produce_node_cvar_name="FFT_Nodes",
            basic_kernel_header_file="FFT_Expression_Kernels.h",
            basic_kernel_source_file="FFT_Expression_Kernels.c",
            graph_warm_construct=1,
        )
    )

    ## Inverse Graph
    G_fft_inverse = NNGraph(name='fft_inverse', old_dsp_lib=False)
    inp = G_fft_inverse.add_input(Dim.unnamed([1, 2*(n_fft//2 + 1)]))
    power = 0
    rfft_params = IRFFT2DPreprocessingNode(
        "test_irfft",
        n_fft=n_fft,
        n_frames=1,
        window=None # Reconstructing Hanning with overlapp and add - no need to invert the window
    )(inp)
    out = G_fft_inverse.add_output()(rfft_params)
    G_fft_inverse.add_dimensions()
    G_fft_inverse.adjust_order()

    # Don't need real stats since we quantize to Float
    G_fft_inverse.quantize(graph_options=graph_opts)
    G_fft_inverse.gen_at_model(
        write_constants=True,
        settings=model_settings(
            l1_size=128000,
            l2_size=1300000,
            tensor_directory=args.inverse_tensors_dir,
            model_directory=os.path.split(args.inverse_at_model_path)[0],
            model_file=os.path.split(args.inverse_at_model_path)[1],
            graph_l1_promotion=1,
            l3_flash_device="AT_MEM_L3_DEFAULTFLASH" if args.flash_type == "flash" else "AT_MEM_L3_MRAMFLASH",
            graph_monitor_cvar_name="IFFT_Monitor",
            graph_produce_operinfos_cvar_name="IFFT_Op",
            graph_produce_node_cvar_name="IFFT_Nodes",
            basic_kernel_header_file="IFFT_Expression_Kernels.h",
            basic_kernel_source_file="IFFT_Expression_Kernels.c",
            graph_warm_construct=1,
        )
    )
