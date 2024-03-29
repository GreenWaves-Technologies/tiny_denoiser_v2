import argparse
import os
from csv import writer
import argcomplete
import numpy as np
import soundfile as sf
from denoiser_nntool_utils import single_audio_inference
from denoiser_utils import gather_results, open_wav, postprocessing, preprocessing
from nntool_python_utils.audio_utils import compare_audio
from nn_nntool_script import build_nntool_graph, create_model_parser
from nntool.api.utils import model_settings
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def extend_parser_for_test(main_parser: argparse.ArgumentParser):
    main_parser.add_argument('--mode', default="test_dataset", choices=["test_sample", "test_dataset", "test_on_target"],
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
    main_parser.add_argument('--csv_file', default=None,
                             help="path to csv_file to save tests")
    main_parser.add_argument('--csv_file_allfiles', default=None,
                             help="path to csv_file to save tests")
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
    main_parser.add_argument('--draw', action="store_true",
                             help="Draw the prepared graph")
    return main_parser

if __name__ == '__main__':
    model_parser = create_model_parser()
    parser = extend_parser_for_test(model_parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    G, states_idxs = build_nntool_graph(
        args.trained_model,
        args.quant_type,
        quant_dataset=args.quant_dataset,
        stats_file=args.stats_pickle,
        requantize=args.requantize,
        tensor_train=args.tensor_train
    )

    if args.draw:
        G.draw()

    if args.mode == "test_on_target" and not args.float_exec_test:
        
        nn_states = [np.zeros(G[s].out_dims[so].shape) for s, so in states_idxs]
        mag = np.ones((1, 257, 1))
        model_input_data = [mag,*nn_states]
        
        model_report_dir = "reports/"+args.trained_model+args.quant_type+"/"
        os.makedirs(model_report_dir, exist_ok=True)
        
        res = G.execute_on_target(
            directory=os.path.join(model_report_dir,"build"),
            input_tensors=model_input_data,
            check_on_target=True,
            output_tensors=False,
            print_output=True,
            at_loglevel=2,
            platform='board',
            settings=model_settings(
                #graph_trace_exec=True,
                l1_size=125000,
                l2_size=1300000,
                graph_const_exec_from_flash=True,
                graph_group_weights=True,
                #graph_l1_promotion=2,
                privileged_l3_flash_device='AT_MEM_L3_MRAMFLASH'
            ),
            tolerance=0.02
        )
        with open(os.path.join(model_report_dir, 'table.txt'), 'w') as file:
            print(res.pretty_performance(), file=file)

        if args.csv_file !=  None:
            # List that we want to add as a new row
            # name, cycles, model size, ops, ops/cycles
            List = [args.trained_model, args.quant_type, res.performance[-1][1],G.total_memory_usage[1],res.performance[-1][2],res.performance[-1][3]]
 
            with open(args.csv_file, 'a') as f_object:
    
                # Pass this file object to csv.writer()
                # and get a writer object
                writer_object = writer(f_object)
    
                # Pass the list as an argument into
                # the writerow()
                writer_object.writerow(List)
 
                # Close the file object
                f_object.close()

    elif args.mode == "test_sample":
        stft = preprocessing(args.test_sample).T
        stft_out = single_audio_inference(
            G,
            stft,
            states_idxs=states_idxs,
            quant_exec=not args.float_exec_test,
            disable_tqdm=True
        )
        estimate = postprocessing(stft_out.T)

        if args.clean_test_sample:
            clean_data = open_wav(args.clean_test_sample)
            res = compare_audio(estimate, clean_data, samplerate=16000)
            for k, v in res.items():
                print(f"{k:>30}: {v:.3f}")

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
            stft_out = single_audio_inference(
                G,
                stft,
                states_idxs=states_idxs,
                quant_exec=not args.float_exec_test,
                stats_collector=None,
                disable_tqdm=True
            )
            estimate = postprocessing(stft_out.T)
            
            # compute the metrics
            if args.dns_dataset:
                clean_filename = "clean_fileid_" + filename.split("_")[-1]
            else:
                clean_filename = filename

            clean_file = os.path.join(args.clean_dataset, clean_filename)
            clean_data = open_wav(clean_file) if os.path.exists(clean_file) else None

            res = compare_audio(estimate, clean_data, samplerate=16000)
            if args.verbose:
                print(f"Sample ({c}/{len(files)})\t{filename}")
                for k, v in res.items():
                    print(f"{k:>30}: {v:.3f}")

            res.update({"filename": noisy_file})
            row_list.append(res)
            noisy_data = np.clip(noisy_data, -1.0, 1.0)
            noisy_res = compare_audio(noisy_data, clean_data, samplerate=16000)
            noisy_res.update({"filename": noisy_file})
            noisy_row_list.append(noisy_res)

            if args.output_dataset:
                os.makedirs(args.output_dataset, exist_ok=True)
                # model_name = os.path.splitext(os.path.basename(args.trained_model))[0]
                # filename = os.path.splitext(os.path.basename(filename))[0]
                # output_file = os.path.join(args.output_dataset, f"{filename}_{model_name}_{'fp32' if args.float_exec_test else args.quant_type}.wav")
                output_file = os.path.join(args.output_dataset, filename)
                # Write out audio as 24bit PCM WAV
                sf.write(output_file, estimate, 16000)

        df_mean, df_noisy_mean = gather_results(
            row_list,
            noisy_row_list,
            csv_file=args.csv_file,
            csv_file_allfiles=args.csv_file_allfiles,
            model_name=args.trained_model
        )
        print(df_mean)
