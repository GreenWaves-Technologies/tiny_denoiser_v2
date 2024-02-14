import argparse
import os
import argcomplete
import numpy as np
import pandas as pd
import soundfile as sf
from denoiser_utils import gather_results, open_wav, postprocessing, preprocessing
from nntool_python_utils.audio_utils import compare_audio
import onnxruntime
from tqdm import tqdm

def create_model_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='nntool_script')

    # Model options
    parser.add_argument('--trained_model', required=True,
                        help="Path to the trained tflite/onnx")
    parser.add_argument('--mode', default="test_dataset", choices=["test_sample", "test_dataset"],
                        help="Script mode")
    # mode = test_sample
    parser.add_argument('--test_sample', default="dataset/test/noisy/p232_050.wav",
                        help=".wav file to use for testing")
    parser.add_argument('--clean_test_sample', default=None,
                        help="path to the clean version of the test_sample to check pesq/stoi metrics (if not provided it will not be used)")
    parser.add_argument('--out_wav', default="output_nn.wav",
                        help="path to the cleaned wav file")
    # mode = test_dataset
    parser.add_argument('--csv_file', default=None,
                        help="path to csv_file to save tests")
    parser.add_argument('--csv_file_allfiles', default=None,
                        help="path to csv_file to save tests")
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
    return parser

def inference(model_path, in_features_frames, ordered_rnn_states_shapes, output_tensor_names, disable_tqdm=False):
    # Load the ONNX model
    onnx_sess = onnxruntime.InferenceSession(model_path)

    rnn_states = [np.zeros(shape, dtype=np.float32) for shape in ordered_rnn_states_shapes]
    # Loop over all mags (timesteps) to creat denoising masks
    masked_features = np.empty_like(in_features_frames)
    for i, in_features in enumerate(tqdm(in_features_frames, disable=disable_tqdm)):
        in_features_mag = np.abs(in_features)

        inputs = {
            'input': in_features_mag.reshape((1, -1, 1)),
        }
        inputs.update({
            'rnn1_h_state_in': rnn_states[0],
            'rnn1_c_state_in': rnn_states[1],
            'rnn2_h_state_in': rnn_states[2],
            'rnn2_c_state_in': rnn_states[3]
        })
        outputs = onnx_sess.run(output_tensor_names, inputs)
        feat_mask = outputs[0]
        rnn_states = outputs[1:]

        # print(feat_mask.shape)
        in_features_out = in_features * feat_mask[0, :, 0]
        masked_features[i] = in_features_out
    return masked_features

if __name__ == '__main__':
    parser = create_model_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Load the ONNX model
    onnx_session = onnxruntime.InferenceSession(args.trained_model)
    # Get the names of input tensors
    ordered_rnn_states_shapes = []
    for input_node in onnx_session.get_inputs() + onnx_session.get_outputs():
        print (f"{input_node.name}: shape {input_node.shape}, type {input_node.type}")
        if "state_in" in input_node.name:
            ordered_rnn_states_shapes.append(input_node.shape)


    input_tensor_names = [input.name for input in onnx_session.get_inputs()]
    # Get the names of output tensors
    output_tensor_names = [output.name for output in onnx_session.get_outputs()]

    if args.mode == "test_sample":
        stft = preprocessing(args.test_sample).T
        stft_out = inference(
            args.trained_model,
            stft,
            ordered_rnn_states_shapes,
            output_tensor_names,
            disable_tqdm=not args.verbose
        )
        estimate = postprocessing(stft_out.T)

        if args.clean_test_sample:
            clean_data = open_wav(args.clean_test_sample)
            res = compare_audio(estimate, clean_data, samplerate=16000)
            print(f"{res}\n")

        sf.write(args.out_wav, estimate, 16000)

    elif args.mode == "test_dataset":
        print(f"Testing on dataset: {args.noisy_dataset}")
        files = os.listdir(args.noisy_dataset)
        row_list = []
        noisy_row_list = []
        for c, filename in enumerate(tqdm(files)):
            noisy_file = os.path.join(args.noisy_dataset, filename)
            noisy_data = open_wav(noisy_file)
            stft = preprocessing(noisy_data).T
            stft_out = inference(
                args.trained_model,
                stft,
                ordered_rnn_states_shapes,
                output_tensor_names,
                disable_tqdm=True
            )
            estimate = postprocessing(stft_out.T)

            # compute the metrics
            if args.dns_dataset:
                clean_filename = "clean_fileid_" + filename.split("_")[-1]
            else:
                clean_filename = filename

            clean_file = os.path.join(args.clean_dataset, clean_filename)
            clean_data = open_wav(clean_file)

            res = compare_audio(estimate, clean_data, samplerate=16000)
            res.update({"filename": noisy_file})
            row_list.append(res)
            noisy_res = compare_audio(noisy_data, clean_data, samplerate=16000)
            noisy_res.update({"filename": noisy_file})
            noisy_row_list.append(noisy_res)

            if args.verbose:
                print(f"Sample ({c}/{len(files)})\t{filename}\t{res}")

            if args.output_dataset:
                if not os.path.exists(args.output_dataset):
                    os.mkdir(args.output_dataset)
                filename = os.path.splitext(os.path.basename(filename))[0]
                model_name = os.path.splitext(os.path.basename(args.trained_model))[0]
                output_file = os.path.join(args.output_dataset, f"{filename}_{model_name}_onnx.wav")
                # Write out audio as 24bit PCM WAV
                sf.write(output_file, estimate, 16000)

        df_mean = gather_results(
            row_list,
            noisy_row_list,
            csv_file=args.csv_file,
            csv_file_allfiles=args.csv_file_allfiles,
            model_name=args.trained_model
        )
        print(df_mean)
