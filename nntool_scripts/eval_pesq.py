#!/usr/bin/env python3

import argparse
import argcomplete
from denoiser_utils import open_wav

def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='nntool_script')

    # Model options
    parser.add_argument('--noisy_file', required=True,
                        help="Path to the noisy file to analyze")
    parser.add_argument('--clean_file', default=None,
                        help="Path to the corresponding clean file (if not provided some metrics will not be displayed)")
    parser.add_argument('--pesq_thr', default=None, type=float,
                        help='Used to check results in CI')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    noisy_data = open_wav(args.noisy_file)
    clean_data = open_wav(args.clean_file) if args.clean_file is not None else None
    if args.pesq_thr is None:
        from nntool_python_utils.audio_utils import compare_audio
        res = compare_audio(noisy_data, clean_data, samplerate=16000)
        for k, v in res.items():
            print(f"{k:>30}: {v:.3f}")
    else:
        from pesq import pesq
        res = pesq(16000, clean_data, noisy_data, 'wb')
        assert res > args.pesq_thr, f"Pesq lower than expected {res:.3f} <= {args.pesq_thr}"
        print(f"Test passed ({res:.3f} > {args.pesq_thr})")
