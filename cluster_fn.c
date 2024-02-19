#include "tinydenoiser.h"

// #define DUMP_FILES

#ifdef DUMP_FILES
#include "gaplib/fs_switch.h"
static void DumpToFile(char *PrefixName, int FrameId, void *Arr, int Size) {
    char FileName[50];
    sprintf(FileName, "%s_%d.bin", PrefixName, FrameId);

    switch_fs_t fs;
    __FS_INIT(fs);
    printf("Writing to file %s\n", FileName);
    void *File = __OPEN_WRITE(fs, FileName);
    int ret_Input_1 = __WRITE(File, Arr, Size);
    __CLOSE(File);
    __FS_DEINIT(fs);
}
#endif

float alpha = 1;
static int FrameIdx = 0;

void RunDenoiser(DenoiserArg_T *Arg)
{
    STFT_TYPE *InFrame       = Arg->InFrame;
    STFT_TYPE *StftOut       = Arg->StftOut;
    STFT_TYPE *DenoisedFrame = Arg->DenoisedFrame;
    NN_TYPE   *InputNN       = Arg->InputNN;
    NN_TYPE   *OutputNN      = Arg->OutputNN;
    RNN_TYPE_H   *RNN1HState    = Arg->RNN1HState;
    RNN_TYPE_H   *RNN2HState    = Arg->RNN2HState;
    #ifndef GRU
    RNN_TYPE_C   *RNN1CState    = Arg->RNN1CState;
    RNN_TYPE_C   *RNN2CState    = Arg->RNN2CState;
    #endif
    int *PerfCounter = Arg->PerfCounter;

    gap_cl_starttimer();
    gap_cl_resethwtimer();

    /***********
     *  FFT Forward
     **********/
    #ifdef DUMP_FILES
    DumpToFile("frame", FrameIdx, InFrame, FRAME_SIZE*sizeof(STFT_TYPE));
    #endif
    // float min=1000000;float max=-1000000;
    // for(int i=0;i<400;i++){
    //     if(InFrame[i]>max) max=InFrame[i];
    //     if(InFrame[i]<min) min=InFrame[i];
    // }
    // for(int i=0;i<400;i++){
    //     InFrame[i] = (InFrame[i]-1.f );
    // }
    //printf("max: %f min: %f\n",max,min);
    int start = gap_cl_readhwtimer();
    fft_forwardCNN_ConstructCluster();
    fft_forwardCNN(InFrame, StftOut);
    PerfCounter[0] += gap_cl_readhwtimer() - start;

    #ifdef DUMP_FILES
    DumpToFile("stft", FrameIdx, StftOut, 2*STFT_SIZE*sizeof(STFT_TYPE));
    #endif

    // StftOut[0]=0;
    // StftOut[1]=0;
    // StftOut[2]=0;
    // StftOut[3]=0;
    // StftOut[4]=0;
    // StftOut[5]=0;
    // StftOut[6]=0;
    // StftOut[7]=0;
    /***********
     *  Preprocessing (MagSquared)
     **********/
    start = gap_cl_readhwtimer();
    for (unsigned int i=0; i<STFT_SIZE; i++) {
        InputNN[i] = (NN_TYPE) Sqrtf32(StftOut[2*i] * StftOut[2*i] + StftOut[2*i+1] * StftOut[2*i+1]);
    }
    PerfCounter[1] += gap_cl_readhwtimer() - start;
    #ifdef DUMP_FILES
    DumpToFile("stft_mag", FrameIdx, InputNN, STFT_SIZE*sizeof(NN_TYPE));
    #endif

    /***********
     *  NN
     **********/
    start = gap_cl_readhwtimer();
    // In state and out state can point to the same mem area
#ifndef TT
    tinydenoiserCNN(
        #ifndef GRU
        RNN1CState,
        RNN2CState,
        #endif
        RNN1HState,
        RNN2HState,
        InputNN,
        0,
        0,
        OutputNN
    );
#else
    tinydenoiserCNN(
        InputNN,
        RNN1HState,
        RNN1CState,
        RNN2HState,
        RNN2CState,
        OutputNN
        #ifndef TT_FULL_RANK
        ,
        RNN1HState,
        RNN1CState,
        RNN2HState,
        RNN2CState
        #endif
    );
#endif
    PerfCounter[2] += gap_cl_readhwtimer() - start;

    #ifdef DUMP_FILES
    DumpToFile("nnout_mask", FrameIdx, OutputNN, STFT_SIZE*sizeof(STFT_TYPE));
    #endif

    /***********
     *  Postprocessing (Masking)
     **********/
    start = gap_cl_readhwtimer();
    for (int i = 0; i< STFT_SIZE; i++ ){
        #if 0
        StftOut[2*i]    = (STFT_TYPE) (StftOut[2*i]  );
        StftOut[2*i+1]  = (STFT_TYPE) (StftOut[2*i+1]);
        #else
        StftOut[2*i]    = (STFT_TYPE) ( ((float) StftOut[2*i]  ) * ((((float) OutputNN[i]) * alpha ) + (1-alpha)) );
        StftOut[2*i+1]  = (STFT_TYPE) ( ((float) StftOut[2*i+1]) * ((((float) OutputNN[i]) * alpha ) + (1-alpha)) );
        #endif
    }
    PerfCounter[3] += gap_cl_readhwtimer() - start;

    #ifdef DUMP_FILES
    DumpToFile("stft_masked", FrameIdx, StftOut, STFT_SIZE*sizeof(STFT_TYPE));
    #endif

    /***********
     *  FFT Inverse
     **********/
    start = gap_cl_readhwtimer();
    fft_inverseCNN_ConstructCluster();
    fft_inverseCNN(StftOut, DenoisedFrame);
    PerfCounter[4] += gap_cl_readhwtimer() - start;

    FrameIdx++;
}
