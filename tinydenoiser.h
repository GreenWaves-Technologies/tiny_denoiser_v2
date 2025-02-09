#ifndef __tinydenoiser_H__
#define __tinydenoiser_H__

#define __PREFIX(x) tinydenoiser ## x
// Include basic GAP builtins defined in the Autotiler
#include "at_api.h"
#include "fft_forwardKernels.h"
#include "fft_inverseKernels.h"
#include "tinydenoiserKernels.h"
#include "tinydenoiserModelInfos.h"

#define STFT_SIZE      257
#define NN_IN_SIZE     257
#define RNN_STATE_SIZE 256

#define QUANTIZE(a, inv_scale, zp) (((int) (((float) a) * inv_scale) + 0.5) + zp)
#define DEQUANTIZE(a, scale, zp)   (scale * (float)(((int) a) - zp))

extern AT_DEFAULTFLASH_EXT_ADDR_TYPE tinydenoiser_L3_Flash;
extern AT_DEFAULTFLASH_EXT_ADDR_TYPE tinydenoiser_L3_PrivilegedFlash;

typedef struct {
    STFT_TYPE *InFrame;
    STFT_TYPE *StftOut;
    STFT_TYPE *DenoisedFrame;
    NN_TYPE   *InputNN;
    NN_TYPE   *OutputNN;
    RNN_TYPE_H   *RNN1HState;
    RNN_TYPE_C   *RNN1CState;
    RNN_TYPE_H   *RNN2HState;
    RNN_TYPE_C   *RNN2CState;
    int *PerfCounter;
} DenoiserArg_T;

void RunDenoiser(DenoiserArg_T *Arg);

#endif
