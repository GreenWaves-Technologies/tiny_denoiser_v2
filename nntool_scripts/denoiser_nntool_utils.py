import os

import numpy as np
from nntool.api import NNGraph
from nntool.graph.types import LSTMNode, RNNNodeBase
from nntool.stats.activation_ranges_collector import ActivationRangesCollector
from denoiser_utils import preprocessing
from tqdm import tqdm


def single_audio_inference(G: NNGraph, stft_frame_i_T, stats_collector: ActivationRangesCollector = None, quant_exec=False, disable_tqdm=False):
    stft_frame_o_T = np.empty_like(stft_frame_i_T)
    rnn_nodes = [node for node in G.nodes(node_classes=RNNNodeBase, sort=True)]
    rnn_states = []
    for rnn_node in rnn_nodes:
        rnn_states.append(np.zeros(rnn_node.out_dims[0].size()))
        if isinstance(rnn_node, LSTMNode):
            rnn_states.append(np.zeros(rnn_node.out_dims[0].size()))

    len_seq = stft_frame_i_T.shape[0]

    #init lstm to zeros
    stft_mask = np.zeros(257)
    for i in tqdm(range(len_seq), disable=disable_tqdm):
        stft_clip = stft_frame_i_T[i]
        stft_clip_mag = np.abs(stft_clip)

        data = [stft_clip_mag, *rnn_states]
        outputs = G.execute(data, dequantize=quant_exec)

        cnt = 0
        for node in rnn_nodes:
            rnn_states[cnt] = outputs[node.step_idx][0]
            cnt += 1
            if isinstance(node, LSTMNode):
                rnn_states[cnt] = outputs[node.step_idx][-1]
                cnt += 1

        if stats_collector:
            stats_collector.collect_stats(G, data)

        new_stft_mask = outputs[G['output_1'].step_idx][0].squeeze()
        # See how the mask changes over time
        # EUCLIDEAN_DISTANCES.append(np.linalg.norm(new_stft_mask - stft_mask))
        # masks.append( new_stft_mask)
        stft_mask = new_stft_mask

        stft_clip = stft_clip * stft_mask
        stft_frame_o_T[i] = stft_clip

    # fig, ax = plt.subplots()
    # ax.plot(EUCLIDEAN_DISTANCES)
    # ax.imshow(np.array(masks))
    # plt.show()
    return stft_frame_o_T

def get_astats(G: NNGraph, files):
    stats_collector = ActivationRangesCollector(use_ema=False)
    for c, filename in tqdm(enumerate(files)):
        print(f"Collecting Stats from file {c+1}/{len(files)}")
        stft = preprocessing(filename)

        stft_frame_i_T = np.transpose(stft) # swap the axis to select the tmestamp
        _ = single_audio_inference(G, stft_frame_i_T, stats_collector=stats_collector, quant_exec=False)
    return stats_collector.stats
