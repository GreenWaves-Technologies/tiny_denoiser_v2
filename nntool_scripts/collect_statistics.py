import argparse
import argcomplete
import pickle
from glob import glob
from nntool.api import NNGraph
from nntool_scripts.denoiser_nntool_utils import get_astats

def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='nntool_script')

    # Model options
    parser.add_argument('--trained_model', default="model/denoiser_GRU_dns.onnx",
                        help="Path to the trained tflite/onnx")
    parser.add_argument('--quant_dataset', default="dataset/quant/*",
                        help="path to .wav files to use to quantize the network")
    parser.add_argument('--stats_pickle', default=None, required=True,
                        help="pickle file where to store the statistics or get statistics if already saved")
    return parser

if __name__ == '__main__':
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    G = NNGraph.load_graph(args.trained_model)
    G.name = "tinydenoiser"
    G.adjust_order()
    G.fusions('scaled_match_group')

    quant_files = glob(args.quant_dataset)[:20]
    if len(quant_files) < 1:
        raise ValueError("Provide quant_dataset")
    stats = get_astats(G, quant_files)

    print(f"Saving stats dictionary to {args.stats_file}")
    with open(args.stats_file, 'wb') as fp:
        pickle.dump(stats, fp, protocol=pickle.HIGHEST_PROTOCOL)
