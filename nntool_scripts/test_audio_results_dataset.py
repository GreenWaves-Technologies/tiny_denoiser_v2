import argparse
import os
import argcomplete
import pandas as pd
from denoiser_utils import TRACK_METRICS, open_wav
from nntool_python_utils.audio_utils import compare_audio
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def parser_for_audio_test():
    parser = argparse.ArgumentParser(prog='nntool_script')

    parser.add_argument('--noisy_dataset', default=None,
                        help="path to folder of noisy samples")
    parser.add_argument('--clean_dataset', default=None,
                        help="path to folder of clean samples")
    return parser


if __name__ == '__main__':
    parser = parser_for_audio_test()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    row_list = []
    filenames = os.listdir(args.noisy_dataset)
    for filename in tqdm(filenames):
        noisy_file = os.path.join(args.noisy_dataset, filename)
        noisy_data = open_wav(noisy_file)

        if args.clean_dataset:
            clean_file = os.path.join(args.clean_dataset, filename)
            clean_data = open_wav(clean_file) if clean_file is not None else None
        else:
            clean_data = None

        res = compare_audio(noisy_data, clean_data, samplerate=16000)
        res.update({"filename": filename})
        row_list.append(res)

    df = pd.DataFrame(row_list, columns=["filename"] + TRACK_METRICS)
    print(df)
    df_mean = df[TRACK_METRICS].mean().to_frame().transpose()
    print(df_mean)
