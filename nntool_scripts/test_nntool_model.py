import argparse
import os
from csv import writer
import argcomplete
import numpy as np
import soundfile as sf
from denoiser_nntool_utils import single_audio_inference
from denoiser_utils import open_wav, pesq_stoi, postprocessing, preprocessing
from nn_nntool_script import build_nntool_graph, create_model_parser
from nntool.api.utils import model_settings, quantization_options
from tqdm import tqdm

import texttable

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


def calculate_pesq_stoi(estimate, clean_data):
    # sz0 = clean_data.shape[0]
    # sz1 = estimate.shape[0]
    # print(sz0)
    # print(sz1)
    # if sz0 > sz1:
    #     #estimate = np.pad(estimate, (0, sz0-sz1))
    #     clean_data = clean_data[:sz1]
    # else:
    #     estimate = estimate[:sz0]
    # print(clean_data.shape)
    # print(estimate.shape)
    
    pesq_i, stoi_i,dnsmos_i,mean_error = pesq_stoi(clean_data, estimate, 16000)
    return pesq_i, stoi_i,dnsmos_i, mean_error

class TexttableEx(texttable.Texttable):
    @classmethod
    def _fmt_int(cls, x, **kw):
        return f'{x:,}'
    
def pretty_performance(res):
    """
        Return a nice to print table for performance, usage: print(res.pretty_performance())
    """
    table = TexttableEx()
    table.header(['Layer', 'Cycles', 'Ops',
                    'Ops/Cycle', '% ops', '% cycles'])
    table.set_header_align(['l', 'c', 'c', 'c', 'c', 'c'])
    table.set_cols_dtype(['t', 'i', 'i', 'f', 'f', 'f'])
    table.set_cols_align(['l', 'r', 'r', 'r', 'r', 'r'])
    table.set_max_width(0)
    for row in res.performance:
        table.add_row(row)
    return table.draw()

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
        if not os.path.exists(model_report_dir):
            os.makedirs(model_report_dir)
        
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
        table = pretty_performance(res)
        with open(os.path.join(model_report_dir, 'table.txt'), 'w') as file:
            print(table, file=file)

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
            pesq_i, stoi_i, dnsmos_i, mean_error_i= calculate_pesq_stoi(estimate, clean_data)
            print(f"pesq=\t{pesq_i}\tand stoi=\t{stoi_i} and dnsmos=\t{dnsmos_i} and mean error=\t{ mean_error_i}\n")

        sf.write(args.out_wav, estimate, 16000)

    elif args.mode == "test_dataset":
        print(f"Testing on dataset: {args.noisy_dataset}")
        files = os.listdir(args.noisy_dataset)
        pesq = 0
        stoi = 0
        mos = 0 
        mean_error=0
        for c, filename in enumerate(tqdm(files)):
            noisy_file = os.path.join(args.noisy_dataset, filename)
            stft = preprocessing(noisy_file).T
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
            clean_data = open_wav(clean_file)
            pesq_i, stoi_i,dnsmos_i, mean_error_i = calculate_pesq_stoi(estimate, clean_data)
            if args.verbose:
                print(f"Sample ({c}/{len(files)})\t{filename}\twith pesq=\t{pesq_i}\tand stoi=\t{stoi_i} and mean error=\t{mean_error_i}")

            pesq += pesq_i
            stoi += stoi_i
            mos  += dnsmos_i
            mean_error += mean_error_i
            #if ((c+1) % 10) == 0:
            #    print(f"After {c+1} files: PESQ={pesq/(c+1):.4f} STOI={stoi/(c+1):.4f}")

            if args.output_dataset:
                output_file = os.path.join(args.output_dataset, filename+os.path.basename(args.trained_model)+("fp32" if args.float_exec_test else args.quant_type)+".wav")
                # Write out audio as 24bit PCM WAV
                sf.write(output_file, estimate, 16000)

        if args.csv_file !=  None:
 
            # List that we want to add as a new row
            List = [args.trained_model, "fp32" if args.float_exec_test else args.quant_type, pesq/len(files), stoi/len(files), mos/len(files), mean_error/len(files)]
 
            with open(args.csv_file, 'a') as f_object:
    
                # Pass this file object to csv.writer()
                # and get a writer object
                writer_object = writer(f_object)
    
                # Pass the list as an argument into
                # the writerow()
                writer_object.writerow(List)
 
                # Close the file object
                f_object.close()
        print(f"Result of accuracy on the ({len(files)} samples):")
        print(f"PESQ:       {np.round(pesq / len(files),4)}")
        print(f"STOI:       {np.round(stoi / len(files),4)}")
        print(f"DNSMOS:     {np.round(mos / len(files),4)}")
        print(f"Mean Error: {np.round(mean_error / len(files),4)}")
