import os

import numpy as np
from nntool.api import NNGraph
from nntool.graph.types import LSTMNode, RNNNodeBase
from nntool.stats.activation_ranges_collector import ActivationRangesCollector
from denoiser_utils import preprocessing
from tqdm import tqdm


def single_audio_inference(
        G: NNGraph,
        in_features_frames,
        stats_collector: ActivationRangesCollector = None,
        quant_exec=False,
        disable_tqdm=False,
        states_idxs=None,
        output_idx="output_1"
    ):

    if isinstance(output_idx, str):
        out_idx = G[output_idx].step_idx
    elif isinstance(output_idx, int):
        out_idx = output_idx
    else:
        raise ValueError("output_idx must be integer or string or None")

    if states_idxs is None:
        raise ValueError("You need to provide states indexes")
    nn_states = [np.zeros(G[s].out_dims[so].shape) for s, so in states_idxs]

    masked_features = np.empty_like(in_features_frames)
    feat_mask = np.zeros(G[out_idx].out_dims[0].shape)
    for i, in_features in enumerate(tqdm(in_features_frames, disable=disable_tqdm)):
        in_features_mag = np.abs(in_features)

        data = [in_features_mag, *nn_states]
        outputs = G.execute(data, dequantize=quant_exec)
        if stats_collector:
            stats_collector.collect_stats(G, data)

        nn_states = [outputs[s][so] for s, so in states_idxs]
        new_feat_mask = outputs[out_idx][0].squeeze()
        # See how the mask changes over time
        # EUCLIDEAN_DISTANCES.append(np.linalg.norm(new_feat_mask - feat_mask))
        # masks.append( new_feat_mask)
        feat_mask = new_feat_mask

        in_features = in_features * feat_mask
        masked_features[i] = in_features

    # fig, ax = plt.subplots()
    # ax.plot(EUCLIDEAN_DISTANCES)
    # ax.imshow(np.array(masks))
    # plt.show()
    return masked_features

def get_astats(G: NNGraph, files, states_idxs):
    stats_collector = ActivationRangesCollector(use_ema=False)
    for c, filename in tqdm(enumerate(files)):
        print(f"Collecting Stats from file {c+1}/{len(files)}")
        stft = preprocessing(filename)

        stft_frame_i_T = np.transpose(stft) # swap the axis to select the tmestamp
        _ = single_audio_inference(G, stft_frame_i_T, states_idxs=states_idxs, stats_collector=stats_collector, quant_exec=False)
    return stats_collector.stats
