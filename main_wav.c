
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

/* Autotiler includes. */
#include "tinydenoiser.h"
#include "gaplib/wavIO.h"
#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s

#define MAX_N_SAMPLES 40000
#define GAP9_FREQ 370      // 370MHz

struct pi_device DefaultRam; 
struct pi_device* ram = &DefaultRam;
//Setting a big buffer to load files from PC to L2 and then store in ram
#define TEMP_L2_SIZE 1200000
#define AUDIO_BUFFER_SIZE (TEMP_L2_SIZE>>1)

AT_DEFAULTFLASH_EXT_ADDR_TYPE fft_forward_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE fft_inverse_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE tinydenoiser_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE tinydenoiser_L3_PrivilegedFlash = 0;

int main(int argc, char *argv[])
{
    printf("\n\n\t *** TinyDenoiser (WAV) ***\n\n");

    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;
    cl_conf.scratch_size = 8*SLAVE_STACK_SIZE;

    cl_conf.id = 0; /* Set cluster ID. */
                    // Enable the special icache for the master core
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |
                    // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                    PI_CLUSTER_ICACHE_PREFETCH_ENABLE |
                    // Enable the icache for all the cores
                    PI_CLUSTER_ICACHE_ENABLE;

    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        return -4;
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, GAP9_FREQ*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, GAP9_FREQ*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, GAP9_FREQ*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        return -4;
    }
	printf("FC Frequency = %d Hz CL Frequency = %d Hz PERIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

    /****
        Configure And Open the External Ram. 
    ****/
    struct pi_default_ram_conf ram_conf;
    pi_default_ram_conf_init(&ram_conf);
    ram_conf.baudrate = GAP9_FREQ*1000*1000;
    pi_open_from_conf(&DefaultRam, &ram_conf);
    if (pi_ram_open(&DefaultRam))
    {
        printf("Error ram open !\n");
        return -3;
    }
    printf("RAM Opened\n");

    /****
        Load Audio Wav from file 

    ****/
    // Read Audio Data from file using temp_L2_memory as temporary buffer
    // Data are prepared in L3 external memory
    char* temp_L2_memory = pi_l2_malloc(TEMP_L2_SIZE);
    if (temp_L2_memory == 0) {
        printf("Error when allocating L2 buffer\n");
        return -5;
    }

    uint32_t inSig, outSig;
    // Allocate L3 buffers for audio IN/OUT
    if (pi_ram_alloc(&DefaultRam, &inSig, (uint32_t) AUDIO_BUFFER_SIZE*sizeof(short)))
    {
        printf("inSig Ram malloc failed !\n");
        return -4;
    }
    if (pi_ram_alloc(&DefaultRam, &outSig, (uint32_t) AUDIO_BUFFER_SIZE*sizeof(short)))
    {
        printf("outSig Ram malloc failed !\n");
        return -5;
    }

    header_struct header_info;
    if (ReadWavFromFileL3(__XSTR(WAV_FILE), inSig, AUDIO_BUFFER_SIZE*sizeof(short), &header_info, &DefaultRam)){
        printf("\nError reading wav file %s\n", __XSTR(WAV_FILE));
        return -1;
    }
    int samplerate = header_info.SampleRate;
    int num_samples = header_info.DataSize * 8 / (header_info.NumChannels * header_info.BitsPerSample);
    num_samples = num_samples > MAX_N_SAMPLES ? MAX_N_SAMPLES : num_samples;
    printf("Num Samples: %d with BitsPerSample: %d SR: %dkHz\n", num_samples, header_info.BitsPerSample, samplerate);

    // Reset Output Buffer and copy to L3
    short * out_temp_buffer = (short *) temp_L2_memory;
    for(int i=0; i < num_samples; i++){
        out_temp_buffer[i] = 0;
    }
    pi_ram_write(&DefaultRam, outSig, temp_L2_memory, num_samples * sizeof(short));

    // free the temporary input memory
    pi_l2_free(temp_L2_memory, TEMP_L2_SIZE);

    /****
        Autotiler functions constructors
    ****/
    gap_fc_starttimer();
    gap_fc_resethwtimer();
    unsigned long int start=0, elapsed=0;

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorFFTErr = fft_forwardCNN_Construct(0);
    if (ConstructorFFTErr) {
        printf("Graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorFFTErr));
        return -6;
    }
    fft_forwardCNN_Destruct(1);
    int ConstructoriFFTErr = fft_inverseCNN_Construct(0);
    if (ConstructoriFFTErr) {
        printf("Graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructoriFFTErr));
        return -6;
    }
    fft_inverseCNN_Destruct(1);
    int ConstructorNNErr = tinydenoiserCNN_Construct(0);
    fft_forward_L1_Memory = tinydenoiser_L1_Memory;
    fft_inverse_L1_Memory = tinydenoiser_L1_Memory;
    if (ConstructorNNErr) {
        printf("Graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorNNErr));
        return -6;
    }

    /****
     * Allocate cluster arguments
    ****/
    STFT_TYPE *InFrame          = (STFT_TYPE *) pi_l2_malloc(FRAME_SIZE * sizeof(STFT_TYPE));
    STFT_TYPE *StftOut          = (STFT_TYPE *) pi_l2_malloc(2 * STFT_SIZE * sizeof(STFT_TYPE));
    STFT_TYPE *DenoisedFrame    = (STFT_TYPE *) pi_l2_malloc(FRAME_SIZE * sizeof(STFT_TYPE));
    short int *DenoisedFrameTmp = (short int *) pi_l2_malloc(FRAME_SIZE * sizeof(short int));
    NN_TYPE   *InputNN          = (NN_TYPE *)   pi_l2_malloc(STFT_SIZE * sizeof(NN_TYPE));
    NN_TYPE   *OutputNN         = (NN_TYPE *)   pi_l2_malloc(STFT_SIZE * sizeof(NN_TYPE));
    NN_TYPE   *RNN1HState       = (NN_TYPE *)   pi_l2_malloc(RNN_STATE_SIZE * sizeof(NN_TYPE));
    NN_TYPE   *RNN2HState       = (NN_TYPE *)   pi_l2_malloc(RNN_STATE_SIZE * sizeof(NN_TYPE));
    if (InFrame==NULL || StftOut==NULL || DenoisedFrame==NULL || DenoisedFrameTmp==NULL || InputNN==NULL || OutputNN==NULL || RNN1HState==NULL || RNN2HState==NULL) {
        printf("Error allocating input/output buffers\n");
        return -1;
    }
    #ifndef GRU
    NN_TYPE   *RNN1CState       = (NN_TYPE *)   pi_l2_malloc(RNN_STATE_SIZE * sizeof(NN_TYPE));
    NN_TYPE   *RNN2CState       = (NN_TYPE *)   pi_l2_malloc(RNN_STATE_SIZE * sizeof(NN_TYPE));
    if (RNN1CState==NULL || RNN2CState==NULL) {
        printf("Error allocating input/output buffers\n");
        return -1;
    }
    #endif
    
    int perf_counter[5] = {0, 0, 0, 0, 0};

    DenoiserArg_T Arg = {
        .InFrame=InFrame,
        .StftOut=StftOut,
        .DenoisedFrame=DenoisedFrame,
        .InputNN=InputNN,
        .OutputNN=OutputNN,
        .RNN1HState=RNN1HState,
        .RNN2HState=RNN2HState,
        #ifndef GRU
        .RNN1CState=RNN1CState,
        .RNN2CState=RNN2CState,
        #endif
        .PerfCounter=perf_counter
    };

    /****
     * Main Loop
    ****/
    int tot_frames = (int) (((float) (num_samples - FRAME_SIZE) / FRAME_STEP));
    printf("Number of frames to be processed: %d\n", tot_frames);

    struct pi_cluster_task task_cluster;
    pi_cluster_task(&task_cluster, (void (*)(void *))RunDenoiser, (void *) &Arg);
    void *stacks = pi_cl_l1_scratch_alloc(&cluster_dev, &task_cluster, 8*SLAVE_STACK_SIZE);
    pi_cluster_task_stacks(&task_cluster, stacks, SLAVE_STACK_SIZE);

    for (int i=0; i<256; i++) {
        RNN1HState[i] = ZERO;
        RNN2HState[i] = ZERO;
        #ifndef GRU
        RNN1CState[i] = ZERO;
        RNN2CState[i] = ZERO;
        #endif
    }
    for (int frame_id=0; frame_id < tot_frames; frame_id++)
    {
        printf("Frame [%3d/%3d]\n", frame_id+1, tot_frames);

        /******
         * Copy Data from L3 to L2
        ******/
        short * in_temp_buffer = (short *) InFrame;
        pi_ram_read(
            &DefaultRam, 
            inSig + frame_id * FRAME_STEP * sizeof(short), 
            in_temp_buffer, 
            (uint32_t) FRAME_SIZE*sizeof(short)
        );

        /******
         * cast data from Q16.15 to f16 (may be float16)
        ******/
        for (int i=(FRAME_SIZE-1) ; i>=0; i--){
            InFrame[i] = ((STFT_TYPE) in_temp_buffer[i])/(1<<15);
        }
        STFT_TYPE min=1000000;float max=-1000000;
        for(int i=300;i<400;i++){
            if(InFrame[i]>max) max=InFrame[i];
            if(InFrame[i]<min) min=InFrame[i];
        }
        printf("max: %f min: %f\n",max,min);
        /******
         * Compute Preprocessing + NN + PostProcessing
        ******/
        pi_cluster_send_task_to_cl(&cluster_dev, &task_cluster);

        /******
         * Hanning window requires divide by 1 when overlapp and add 50%
        ******/
        for (int i= 0 ; i<FRAME_SIZE; i++){
            DenoisedFrame[i] = DenoisedFrame[i] / 2;
        }

        /******
         * Read the outsignal
        ******/
        pi_ram_read(&DefaultRam, (uint32_t) ((short *) outSig + (frame_id*FRAME_STEP)), 
            DenoisedFrameTmp, FRAME_SIZE * sizeof(short));

        /******
         * Overlap And ADD
        ******/
        for (int i= 0 ; i<FRAME_SIZE; i++){
            DenoisedFrameTmp[i] += (short int)(DenoisedFrame[i] * (1<<15));
        }

        /******
         * Write to RAM
        ******/
        pi_ram_write(&DefaultRam, (uint32_t)( (short *) outSig + (frame_id*FRAME_STEP)),
            DenoisedFrameTmp, FRAME_SIZE * sizeof(short));

    }

    unsigned long int elapsed_tot = perf_counter[0] + perf_counter[1] + perf_counter[2] + perf_counter[3] + perf_counter[4];
    float elapsed_frame_us = ((float) elapsed_tot) / tot_frames / GAP9_FREQ;
    float real_time_constraint_us = ((float) FRAME_STEP) / 16 * 1000;
    printf("===================================================\n");
    printf("| Func | Total Cycles | Cycles/Frame | Percentage |\n");
    printf("===================================================\n");
    printf("| FFT  | %12d | %12d |     %5.2f%% |\n", perf_counter[0], perf_counter[0]/tot_frames, ((float) perf_counter[0]) / elapsed_tot * 100);
    printf("| MAG  | %12d | %12d |     %5.2f%% |\n", perf_counter[1], perf_counter[1]/tot_frames, ((float) perf_counter[1]) / elapsed_tot * 100);
    printf("| NN   | %12d | %12d |     %5.2f%% |\n", perf_counter[2], perf_counter[2]/tot_frames, ((float) perf_counter[2]) / elapsed_tot * 100);
    printf("| MASK | %12d | %12d |     %5.2f%% |\n", perf_counter[3], perf_counter[3]/tot_frames, ((float) perf_counter[3]) / elapsed_tot * 100);
    printf("| iFFT | %12d | %12d |     %5.2f%% |\n", perf_counter[4], perf_counter[4]/tot_frames, ((float) perf_counter[4]) / elapsed_tot * 100);
    printf("---------------------------------------------------\n");
    printf("| Frame: %3.0fus to process %4.0fus (%5.2f%% DC)      |\n", elapsed_frame_us, real_time_constraint_us, elapsed_frame_us / real_time_constraint_us * 100);
    printf("===================================================\n");

    tinydenoiserCNN_Destruct(0);
    fft_forwardCNN_Destruct(0);
    fft_inverseCNN_Destruct(0);

    /*
        Exit the real-time mode (only for testing)
        and write clean speech audio to file
    */
    printf("Writing wav file to %s completed successfully\n", __XSTR(OUT_FILE));
    WriteWavFromL3ToFile(__XSTR(OUT_FILE), 16, samplerate, 1, (uint32_t *) outSig, num_samples* sizeof(short), &DefaultRam);

    /*
        Deallocate everything and Close the cluster
    */
    pi_cluster_close(&cluster_dev);

    printf("Ended\n");
    return 0;
}
