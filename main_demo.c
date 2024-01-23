/*
 * Copyright (C) 2022 GreenWaves Technologies
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "tinydenoiser.h"
#include "bsp/bsp.h"
#include "SFUGraph_L2_Descr.h"
#include "sfu_pmsis_runtime.h"

#define Q_BIT_IN 26
#define Q_BIT_OUT 26

#define SAI_SCK(itf)         (48+(itf*4)+0)
#define SAI_WS(itf)          (48+(itf*4)+1)
#define SAI_SDI(itf)         (48+(itf*4)+2)
#define SAI_SDO(itf)         (48+(itf*4)+3)

#define MAX_PERFORMANCE
#ifdef MAX_PERFORMANCE
    // Max performance settings
    #define CL_FREQ  370000000      // 370MHz
    #define SOC_FREQ 370000000      // 370MHz
    #define VOLTAGE  800
#else
    // Max energy efficiency settings
    #define CL_FREQ  240000000      // 240MHz
    #define SOC_FREQ 240000000      // 240MHz
    #define VOLTAGE  650
#endif

#define PAD_GPIO_LED2    (PI_PAD_086)

/***************************************************************************************************
 * Global data (stored in L2)
 **************************************************************************************************/
AT_DEFAULTFLASH_EXT_ADDR_TYPE fft_forward_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE fft_inverse_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE tinydenoiser_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE tinydenoiser_L3_PrivilegedFlash = 0;

L2_MEM STFT_TYPE ReconstructedFrameTmp[N_FFT];

extern float alpha;

// Application's main settings *************************************************

#define SSM6515_VOLUME 0        // Output Volume [dB]

// Channels definition
#define CHANNEL_LEFT    0
#define CHANNEL_RIGHT   1
#define CHANNEL_NUMBER  2

#define CHUNK_SIZE      (FRAME_STEP*sizeof(int))
#define NB_BUFF_OUT     2
#define NB_BUFF_IN      2

#define MICRO_PDM_DIRECTION     DT_MICRO_AL_PDM_DIRECTION
#define MICRO_PDM_RIGHT_CHANNEL DT_MICRO_AL_PDM_CHANNEL
#define MICRO_PDM_LEFT_CHANNEL  DT_MICRO_AL_PDM_CHANNEL
#define MICRO_PDM_SAI           DT_MICRO_AL_SAI_ITF

// SAI Common configuration ****************************************************

// Both SAI are shared between microphone and DAC
#define SAMPLING_FREQUENCY      48000 
#define SFU_UPSAMPLING_FACTOR   (64)
#define PDM_FREQUENCY           (SAMPLING_FREQUENCY * SFU_UPSAMPLING_FACTOR)

// Application's devices *******************************************************

static pi_device_e  ssm6515_enum[CHANNEL_NUMBER] = {PI_DAC_SSM6515_LEFT, PI_DAC_SSM6515_RIGHT};
static pi_device_t* ssm6515_device[CHANNEL_NUMBER]; // Pointers to SSM6515 device instances

// In this example, two SAI are involved to input and output audio
// CHANNEL_LEFT  refers to SAI2. It is also connected to Microphone A
// CHANNEL_RIGHT refers to SAI1. It is also connected to Microphone B
static pi_device_t   i2s_device[CHANNEL_NUMBER];

// SFU *************************************************************************

static pi_sfu_graph_t* sfu_graph; // SFU graph instance.
static int             sfu_ssm6515_desc_id[CHANNEL_NUMBER] = {SFU_Name(SFUGraph, PdmOut1), SFU_Name(SFUGraph, PdmOut2)};

static pi_sfu_mem_port_t *sfu_memout_port; // Memory port context
static pi_sfu_mem_port_t *sfu_memin_port;

// List of buffer lists (a list for each output port)
static pi_sfu_buffer_t sfu_pdmin_buff[NB_BUFF_IN];
static pi_sfu_buffer_t sfu_pdmout_buff[NB_BUFF_OUT];

static uint32_t sfu_pdmin_buff_idx = 0;
static uint32_t sfu_pdmout_buff_idx = 0;

// OS task to trigger main thread
static pi_evt_t proc_task_pdmin;
static pi_evt_t proc_task_pdmout;

// OS task to register end of chunk transfer callback
static pi_evt_t sfu_mem_out_task;
static pi_evt_t sfu_mem_in_task;

// SSM6515 Configuration *******************************************************

/**
 * @brief SSM6515 Registers configuration
 */
static const pi_ssm6515_confreg_t ssm6515_confreg[CHANNEL_NUMBER] = 
{ // CHANNEL_LEFT
    {
        { // Power
            SSM6515_AUTO_PWDN_DISABLED, // Set the Automatic power down feature
            SSM6515_LIMITER_DISABLED    // Disable Limiter
        },

        { // Clock
            SSM6515_BCLK_RATE_AUTO // Automatic BCLK rate detection.
        },

        { // PDM
            SSM6515_MODE_PDM,                // PDM used for input.
            SSM6515_PDM_FS_2_8224_TO_3_072,  // 2.8224 MHz to 3.072 MHz clock in PDM mode.
            SSM6515_PDM_RISING_EDGE_CHANNEL, // Channel selection
            SSM6515_PDM_FILTER_LOW,          // Lowest filtering. Lowest latency.
            SSM6515_PDM_PHASE_FALL           // Fall-rise channel pair is in phase.
        },

        { // DAC
            SSM6515_DAC_FS_44_1_TO_48KHZ, // Sample rate
            SSM6515_DAC_INVERT_DISABLE,   // No phase inversion in DAC.
            SSM6515_HF_CLIP_ENCODE(255),  // DAC High Frequency Clip Value.
            { // DAC Volume
                SSM6515_VOLUME_ENCODE(SSM6515_VOLUME), // DAC Volume
                SSM6515_DAC_VOL_CTRL_ENABLED,          // Volume control enabled.
                SSM6515_DAC_RAMPING_ENABLE,            // Soft volume ramping.
                SSM6515_DAC_UNMUTE,                    // DAC is not muted
                SSM6515_DAC_ZEROCROSSING_ENABLED       // Volume change occurs only at zero crossing.
            },
            { // Filter
                SSM6515_DAC_FILTER_DISABLED, // Interpolation filter is disabled
                { // Highpass
                    SSM6515_HP_FILTER_OFF,         // DAC high-pass filter off.
                    SSM6515_HP_FILTER_CUTOFF_1_HZ, // Unused if high pass filter is off
                },
            },
            { // Mode
                SSM6515_AMP_LPM_OFF,         // Low Power Mode disabled
                SSM6515_PWR_MODE_NO_SAVINGS, // No power savings.
                SSM6515_DAC_IBIAS_NORMAL,    // Bias set to normal operation
            },
        },

        { // SAI
            SSM6515_SAI_MODE_STEREO,     // I2S mode (2 channels)
            SSM6515_DATA_FORMAT_I2S,     // I2S format
            SSM6515_TDM_32BCLK_PER_SLOT, // Slot width
            SSM6515_BCLK_POL_RISING,     // Capture on rising edge.
            SSM6515_LRCLK_POL_NORMAL,    // LRCLK Polarity.
            SSM6515_SLOT_1_LEFT          // Channel to output
        },
        { // Amplifier
            { // Amplifier mode
                SSM6515_EMI_NORMAL,  // EMI Mode
                SSM6515_AMP_LPM_OFF, // Amplifier Low Power Mode disabled.
            },
            SSM6515_AMP_RLOAD_32,            // Resistive Load set to 32 ohms
            SSM6515_AMP_OVC_PROTECT_DISABLE, // Overcurrent Protection disabled
        },
        { // Output
            SSM6515_LIM_ATR_30,          // Audio Limiter Attack Rate. [µs/dB]
            SSM6515_LIM_RRT_1200,        // Audio Limiter Release Rate. [ms/dB]
            SSM6515_LIM_THRES_MINUS_2dB, // Limiter Threshold. [dB]
        },
        { // Fault
            SSM6515_MRCV_NORMAL,         // Manual Fault Recover set to normal
            SSM6515_FAULT_RECOVERY_AUTO, // Automatically revover from undervoltage fault 
            SSM6515_FAULT_RECOVERY_AUTO, // Automatically revover from overtemperature fault 
            SSM6515_FAULT_RECOVERY_AUTO  // Automatically revover from overcurrent fault 
        },
    },
    { // CHANNEL_RIGHT
        { // Power
            SSM6515_AUTO_PWDN_DISABLED, // Set the Automatic power down feature
            SSM6515_LIMITER_DISABLED    // Disable Limiter
        },

        { // Clock
            SSM6515_BCLK_RATE_AUTO // Automatic BCLK rate detection.
        },

        { // PDM
            SSM6515_MODE_PDM,                // PDM used for input.
            SSM6515_PDM_FS_2_8224_TO_3_072,  // 2.8224 MHz to 3.072 MHz clock in PDM mode.
            SSM6515_PDM_RISING_EDGE_CHANNEL, // Channel selection
            SSM6515_PDM_FILTER_LOW,          // Lowest filtering. Lowest latency.
            SSM6515_PDM_PHASE_FALL           // Fall-rise channel pair is in phase.
        },

        { // DAC
            SSM6515_DAC_FS_44_1_TO_48KHZ, // Sample rate
            SSM6515_DAC_INVERT_DISABLE,   // No phase inversion in DAC.
            SSM6515_HF_CLIP_ENCODE(255),  // DAC High Frequency Clip Value.
            { // DAC Volume
                SSM6515_VOLUME_ENCODE(SSM6515_VOLUME), // DAC Volume
                SSM6515_DAC_VOL_CTRL_ENABLED,          // Volume control enabled.
                SSM6515_DAC_RAMPING_ENABLE,            // Soft volume ramping.
                SSM6515_DAC_UNMUTE,                    // DAC is not muted
                SSM6515_DAC_ZEROCROSSING_ENABLED       // Volume change occurs only at zero crossing.
            },
            { // Filter
                SSM6515_DAC_FILTER_DISABLED, // Interpolation filter is disabled
                { // Highpass
                    SSM6515_HP_FILTER_OFF,         // DAC high-pass filter off.
                    SSM6515_HP_FILTER_CUTOFF_1_HZ, // Unused if high pass filter is off
                },
            },
            { // Mode
                SSM6515_AMP_LPM_OFF,         // Low Power Mode disabled
                SSM6515_PWR_MODE_NO_SAVINGS, // No power savings.
                SSM6515_DAC_IBIAS_NORMAL,    // Bias set to normal operation
            },
        },

        { // SAI
            SSM6515_SAI_MODE_STEREO,     // I2S mode (2 channels)
            SSM6515_DATA_FORMAT_I2S,     // I2S format
            SSM6515_TDM_32BCLK_PER_SLOT, // Slot width
            SSM6515_BCLK_POL_RISING,     // Capture on rising edge.
            SSM6515_LRCLK_POL_NORMAL,    // LRCLK Polarity.
            SSM6515_SLOT_2_RIGHT         // Channel to output
        },
        { // Amplifier
            { // Amplifier mode
                SSM6515_EMI_NORMAL,  // EMI Mode
                SSM6515_AMP_LPM_OFF, // Amplifier Low Power Mode disabled.
            },
            SSM6515_AMP_RLOAD_32,            // Resistive Load set to 32 ohms
            SSM6515_AMP_OVC_PROTECT_DISABLE, // Overcurrent Protection disabled
        },
        { // Output
            SSM6515_LIM_ATR_30,          // Audio Limiter Attack Rate. [µs/dB]
            SSM6515_LIM_RRT_1200,        // Audio Limiter Release Rate. [ms/dB]
            SSM6515_LIM_THRES_MINUS_2dB, // Limiter Threshold. [dB]
        },
        { // Fault
            SSM6515_MRCV_NORMAL,         // Manual Fault Recover set to normal
            SSM6515_FAULT_RECOVERY_AUTO, // Automatically revover from undervoltage fault 
            SSM6515_FAULT_RECOVERY_AUTO, // Automatically revover from overtemperature fault 
            SSM6515_FAULT_RECOVERY_AUTO  // Automatically revover from overcurrent fault 
        },
    }
};

/*
 * Called when a chunk has been transferred from MEM_OUT to L2.
 */
static void handle_mem_in_end(void * arg)
{
    /* Enqueue next buffer to receive from MEM_OUT */
    pi_sfu_enqueue(
        sfu_graph, 
        sfu_memin_port, 
        &sfu_pdmout_buff[sfu_pdmout_buff_idx]);

    sfu_pdmout_buff_idx ^= 1;
    pi_evt_push(&proc_task_pdmout);
}

static void handle_input_transfer_end(void *arg)
{
    // Enqueue next buffer
    pi_sfu_enqueue(
        sfu_graph,
        sfu_memout_port,
        &sfu_pdmin_buff[sfu_pdmin_buff_idx]);
    sfu_pdmin_buff_idx ^= 1;
    pi_evt_push(&proc_task_pdmin);
}

/**
 * @brief Handle the SFU's opening process
 * 
 * @return int32_t Error management
 */
static pi_err_t open_sfu()
{
    pi_sfu_conf_t sfu_conf = { .sfu_frequency=SOC_FREQ };

    if(PI_OK != pi_sfu_open(&sfu_conf)) {
        printf("Failed to open SFU\n");
        return PI_FAIL;
    }

    sfu_graph = pi_sfu_graph_open(&SFU_RTD(SFUGraph));
    if(NULL == sfu_graph) {
        printf("Failed to open SFU Graph\n");
        return PI_FAIL;
    }

    /***********************************************************************************************
     * Configure Input / Output (MEM_OUT,MEM_IN uDMA/1D)
     **********************************************************************************************/

    //Allocated Output buffers
    sfu_memin_port = pi_sfu_mem_port_get(sfu_graph, SFU_Name(SFUGraph, MemIn));
    if (sfu_memin_port == NULL) {
        printf("Failed to get MEM_OUT port references\n");
        return PI_FAIL;
    }
    pi_evt_callback_irq_init(&sfu_mem_in_task, handle_mem_in_end, 0);

    for (int i = 0; i < NB_BUFF_OUT; i++) {
        void *data = pi_l2_malloc(CHUNK_SIZE);
        if (data == NULL) {
            printf("error allocating buffer out\n");
            return PI_FAIL;
        }
        for (int p=0; p<FRAME_STEP; p++) ((int *) data)[p] = 0;
        pi_sfu_buffer_init(&sfu_pdmout_buff[i], data, FRAME_STEP, sizeof(int));

        sfu_pdmout_buff[i].task = &sfu_mem_in_task;
        //pi_sfu_enqueue(sfu_graph, sfu_memin_port, &sfu_pdmout_buff[i]);
    }
    
    // Allocate Input buffers
    sfu_memout_port = pi_sfu_mem_port_get(sfu_graph, SFU_Name(SFUGraph, MemOut));
    if (sfu_memout_port == NULL) {
        printf("Failed to get MEM_IN port references\n");
        return PI_FAIL;
    }
    pi_evt_callback_irq_init(&sfu_mem_out_task, handle_input_transfer_end, 0);

    for (int i = 0; i < NB_BUFF_IN; i++)
    {
        void *data = pi_l2_malloc(CHUNK_SIZE);
        if (data == NULL) {
            printf("error allocating buffer in\n");
            return PI_FAIL;
        }
        for (int p=0; p<FRAME_STEP; p++) ((int *) data)[p] = 0;
        pi_sfu_buffer_init(&sfu_pdmin_buff[i], data, FRAME_STEP, sizeof(int));

        sfu_pdmin_buff[i].task = &sfu_mem_out_task;
        pi_sfu_enqueue(sfu_graph, sfu_memout_port, &sfu_pdmin_buff[i]);
    }

    printf("Buffers MEM_OUT/MEM_IN for uDMA transfer initialized\n");

    return PI_OK;
}

static pi_err_t open_i2s(pi_device_t * i2s_dev, int i2s_itf)
{
    pi_i2s_conf_t i2s_conf;
    pi_i2s_conf_init(&i2s_conf);

    i2s_conf.options        = PI_I2S_OPT_INT_CLK | PI_I2S_OPT_REF_CLK_FAST | PI_I2S_OPT_WS_POLARITY_FALLING_EDGE;
    i2s_conf.frame_clk_freq = PDM_FREQUENCY; // 3.072MHz
    i2s_conf.itf            = i2s_itf;
    i2s_conf.mode           = PI_I2S_MODE_PDM;
    i2s_conf.pdm_direction = MICRO_PDM_DIRECTION;
    i2s_conf.pdm_diff = 0; // Differential mode is not used.

    pi_open_from_conf(i2s_dev, &i2s_conf);
    if (PI_OK != pi_i2s_open(i2s_dev)) {
        printf("Failed to open I2S device\n");
        return PI_FAIL;
    }
    pi_pad_function_set(SAI_SCK(i2s_itf),PI_PAD_FUNC0);
    pi_pad_function_set(SAI_SDI(i2s_itf),PI_PAD_FUNC0);
    pi_pad_function_set(SAI_SDO(i2s_itf),PI_PAD_FUNC0);
    pi_pad_function_set(SAI_WS(i2s_itf),PI_PAD_FUNC0);


    return PI_OK;
}

/**
 * @brief Example entry point.
 * 
 * @return int32_t Error management
 *          @arg PI_OK The test succeed
 *          @arg PI_FAIL One step of the test failed
 */
int main(void)
{
    printf("\n\t *** TinyDenoiser (DEMO) ***\n\n");

    /***********************************************************************************************
     * Configure And open cluster.
     **********************************************************************************************/
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

    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev))  {
        printf("Cluster open failed !\n\r");
        return PI_FAIL;
    }
    pi_freq_set(PI_FREQ_DOMAIN_CL,     CL_FREQ);
    pi_freq_set(PI_FREQ_DOMAIN_FC,     SOC_FREQ);
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH, SOC_FREQ);
    pi_freq_set(PI_FREQ_DOMAIN_SFU,    SOC_FREQ);
    #ifndef __PLATFORM_GVSOC__
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
    #endif

    pi_pad_function_set(PAD_GPIO_LED2, PI_PAD_FUNC1);
    pi_gpio_pin_configure(PAD_GPIO_LED2, PI_GPIO_OUTPUT);


    /***********************************************************************************************
     * Enable slider
     **********************************************************************************************/
    #ifndef __PLATFORM_GVSOC__
    pi_device_t*    ads1014;        // Pointer to ads10114 instance
    float th_low  = 1600.0;         // Low  Threshold In mV
    float th_high = 1800.0;         // High Threshold In mV
    if(PI_OK != pi_open(PI_ADC_ADS1014,  &ads1014)) {
        printf("Error Can not open the ADS10114 Driver\n");
        return -1;
    }
  
    if(PI_OK != pi_ads1014_write_threshold(ads1014,th_low,th_high)) {
        printf("Error Can not write Threshold registers\n");
        return -1;
    }
    float fpot;
    /* Slider values: 1475 .. 2047 */
    pi_ads1014_read_value(ads1014,&fpot);
    alpha = ((fpot - 1470) / 577);
    #else
    alpha = 1;
    #endif

    /***********************************************************************************************
     * Configure and open SFU Graph
     **********************************************************************************************/
    if(PI_OK != open_sfu()) return PI_FAIL;
    printf("SFU Opened\n");

    /***********************************************************************************************
     * Setup SAI interfaces
     **********************************************************************************************/
#ifndef __PLATFORM_GVSOC__
    if((PI_OK != pi_open(ssm6515_enum[CHANNEL_LEFT],  &ssm6515_device[CHANNEL_LEFT])) ||
       (PI_OK != pi_open(ssm6515_enum[CHANNEL_RIGHT], &ssm6515_device[CHANNEL_RIGHT]))
    ) {
        printf("Failed to open SSM6515 devices\n");
        return PI_FAIL;
    }

    pi_ssm6515_conf_t* ssm6515_conf_left  = (pi_ssm6515_conf_t*)(ssm6515_device[CHANNEL_LEFT]->config);
    pi_ssm6515_conf_t* ssm6515_conf_right = (pi_ssm6515_conf_t*)(ssm6515_device[CHANNEL_RIGHT]->config);
    int i2s_itf_left  = ssm6515_conf_left->i2s_itf;
    int pdm_ch_left   = ssm6515_conf_left->pdm_channel;
    int i2s_itf_right = ssm6515_conf_right->i2s_itf;
    int pdm_ch_right  = ssm6515_conf_right->pdm_channel;
#else
    int i2s_itf_left  = 2;
    int pdm_ch_left   = 0;
    int i2s_itf_right = 1;
    int pdm_ch_right  = 0;
#endif

    if ((PI_OK != open_i2s(&i2s_device[CHANNEL_LEFT],  i2s_itf_left)) ||
        (PI_OK != open_i2s(&i2s_device[CHANNEL_RIGHT], i2s_itf_right))
    ) {
        return PI_FAIL;
    }
    printf("SAI setup\n");

    /***********************************************************************************************
     * Binding PDM out Channel
     **********************************************************************************************/
    pi_sfu_pdm_itf_id_t pdm_out_itf_id_left  = {i2s_itf_left,  pdm_ch_left, 1};
    pi_sfu_pdm_itf_id_t pdm_out_itf_id_right = {i2s_itf_right, pdm_ch_right, 1};
    if ((PI_OK != pi_sfu_graph_pdm_bind(sfu_graph, SFU_Name(SFUGraph, PdmOut1),  &pdm_out_itf_id_left)) ||
        (PI_OK != pi_sfu_graph_pdm_bind(sfu_graph, SFU_Name(SFUGraph, PdmOut2),  &pdm_out_itf_id_right))
    ) {
        printf("Failed to open I2S device\n");
        return PI_FAIL;
    }
    printf("Setup sfu out bindings to %d %d - %d %d\n", i2s_itf_left,  pdm_ch_left, i2s_itf_right, pdm_ch_right);

#ifndef __PLATFORM_GVSOC__
    // Writing SSM6515 configuration
    if( PI_OK != pi_dac_configure(ssm6515_device[CHANNEL_LEFT],  (void*)&ssm6515_confreg[CHANNEL_LEFT]) ||
        PI_OK != pi_dac_configure(ssm6515_device[CHANNEL_RIGHT], (void*)&ssm6515_confreg[CHANNEL_RIGHT])) {
        printf("Failed to configure SSM6515 devices\n");
        return PI_FAIL;
    }

    pi_dac_start(ssm6515_device[CHANNEL_LEFT]);
    pi_dac_start(ssm6515_device[CHANNEL_RIGHT]);
#endif

    /***********************************************************************************************
     * Binding this SAI's microphone if it has been enabled by the configuration file
     **********************************************************************************************/
    //Sai, Channel, isOutput
    pi_sfu_pdm_itf_id_t pdm_in_itf_id_left  = { 2, 2, 1};
    if (PI_OK != pi_sfu_graph_pdm_bind(sfu_graph, SFU_Name(SFUGraph, PdmIn),  &pdm_in_itf_id_left)) {
        printf("Failed to open I2S device\n");
        return PI_FAIL;
    }
    printf("Setup sfu in bindings to %d %d\n", i2s_itf_left,  pdm_ch_left);

    /***********************************************************************************************
     * Construct the Autotiler Graphs
     **********************************************************************************************/
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
    tinydenoiserCNN_Destruct(1);
    pi_cluster_close(&cluster_dev);

    /***********************************************************************************************
     * Start I2S Transmission
     **********************************************************************************************/
    pi_i2s_ioctl(&i2s_device[CHANNEL_LEFT],  PI_I2S_IOCTL_START, NULL);
    pi_i2s_ioctl(&i2s_device[CHANNEL_RIGHT], PI_I2S_IOCTL_START, NULL);

    printf("I2S Started\n");


    /***********************************************************************************************
     * Configure OS task that will be used to trigger processing of chunk received from MEM_OUT.
     **********************************************************************************************/

    STFT_TYPE *InFrame          = (STFT_TYPE *)  pi_l2_malloc(N_FFT * sizeof(STFT_TYPE));
    STFT_TYPE *StftOut          = (STFT_TYPE *)  pi_l2_malloc(2 * STFT_SIZE * sizeof(STFT_TYPE));
    STFT_TYPE *DenoisedFrame    = (STFT_TYPE *)  pi_l2_malloc(N_FFT * sizeof(STFT_TYPE));
    NN_TYPE   *InputNN          = (NN_TYPE *)    pi_l2_malloc(STFT_SIZE * sizeof(NN_TYPE));
    NN_TYPE   *OutputNN         = (NN_TYPE *)    pi_l2_malloc(STFT_SIZE * sizeof(NN_TYPE));
    RNN_TYPE  *RNN1HState       = (RNN_TYPE *)   pi_l2_malloc(RNN_STATE_SIZE * sizeof(RNN_TYPE));
    RNN_TYPE  *RNN2HState       = (RNN_TYPE *)   pi_l2_malloc(RNN_STATE_SIZE * sizeof(RNN_TYPE));
    if (InFrame==NULL || StftOut==NULL || DenoisedFrame==NULL || InputNN==NULL || OutputNN==NULL || RNN1HState==NULL || RNN2HState==NULL) {
        printf("Error allocating input/output buffers\n");
        return -1;
    }
    #ifndef GRU
    RNN_TYPE *RNN1CState       = (RNN_TYPE *)   pi_l2_malloc(RNN_STATE_SIZE * sizeof(RNN_TYPE));
    RNN_TYPE *RNN2CState       = (RNN_TYPE *)   pi_l2_malloc(RNN_STATE_SIZE * sizeof(RNN_TYPE));
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

    for (int i=0; i<256; i++) {
        RNN1HState[i] = 0;
        RNN2HState[i] = 0;
        #ifndef GRU
        RNN1CState[i] = 0;
        RNN2CState[i] = 0;
        #endif
    }

    struct pi_cluster_task task_cluster;
    pi_cluster_task(&task_cluster, (void (*)(void *))RunDenoiser, (void *) &Arg);
    void *stacks = pi_cl_l1_scratch_alloc(&cluster_dev, &task_cluster, 8*SLAVE_STACK_SIZE);
    pi_cluster_task_stacks(&task_cluster, stacks, SLAVE_STACK_SIZE);

    gap_fc_starttimer();
    gap_fc_resethwtimer();

    pi_evt_sig_init(&proc_task_pdmout);
    pi_evt_sig_init(&proc_task_pdmin);
    printf("Start running\n");

    for (int i=0; i<N_FFT; i++) {InFrame[i] = 0;}
    for (int i=0; i<N_FFT; i++) {ReconstructedFrameTmp[i]=0;}

    int counter = 0, gpio_val = 0;
    pi_sfu_graph_load(sfu_graph);
    while(1) {
        #ifndef __PLATFORM_GVSOC__
        pi_ads1014_read_value(ads1014,&fpot);
        alpha = ((fpot - 1470) / 577);
        #endif

        /* Wait Events for MEMIN before filling the buffers */
        pi_evt_wait(&proc_task_pdmin);

        /* Data arrived, wake up the cluster and set high the frequency of the SoC so that you have a better L2->L1 BW */
        pi_fll_ioctl(PI_FREQ_DOMAIN_FC, PI_FLL_IOCTL_DIV_SET, (void *) 1);

        // int start = gap_fc_readhwtimer();

        /* Rotate buffer */
        for (int i=0; i<(N_FFT - FRAME_STEP); i++) {
            InFrame[i] = InFrame[i + FRAME_STEP];
        }
        /* Read buffer in (MemOut) */
        for (int i=0; i<FRAME_STEP; i++) {
            InFrame[i + (N_FFT - FRAME_STEP)] = (((float) ((int32_t *) sfu_pdmin_buff[sfu_pdmin_buff_idx ^ 1].data)[i]) / (1<<Q_BIT_IN));
        }

        /* Run processing on the cluster */
        pi_open_from_conf(&cluster_dev, &cl_conf);
        if (pi_cluster_open(&cluster_dev))  {
            printf("Cluster open failed !\n\r");
            return PI_FAIL;
        }

        tinydenoiserCNN_Construct(1);
        fft_forward_L1_Memory = tinydenoiser_L1_Memory;
        fft_inverse_L1_Memory = tinydenoiser_L1_Memory;
        /******
         * Compute Preprocessing + NN + PostProcessing
        ******/
        pi_cluster_send_task_to_cl(&cluster_dev, &task_cluster);
        tinydenoiserCNN_Destruct(1);

        pi_cluster_close(&cluster_dev);

        /* Hanning window requires divide by X when overlapp and add */
        for (int i=0; i<(N_FFT-FRAME_STEP); i++) {
            ReconstructedFrameTmp[i] = ReconstructedFrameTmp[i+FRAME_STEP] + ((STFT_TYPE) (DenoisedFrame[i] / 4));
        }
        for (int i=(N_FFT-FRAME_STEP); i<N_FFT; i++) {
            ReconstructedFrameTmp[i] = (STFT_TYPE)(DenoisedFrame[i]/4);
        }

        //pi_evt_wait(&proc_task_pdmout);

        //First Copy previous loop processed frame to output
        for(int i=0;i<FRAME_STEP;i++) {
            ((int32_t*)sfu_pdmout_buff[sfu_pdmout_buff_idx ^ 1].data)[i]= (int32_t)(((float)ReconstructedFrameTmp[i])*(1<<Q_BIT_OUT));
            //((int32_t*)sfu_pdmout_buff[sfu_pdmout_buff_idx ^ 1].data)[i]= (int32_t)((float)(InFrame[i]) * (1<<Q_BIT_OUT));
        }
        
        if(counter==3){
            pi_time_wait_us(500);
            pi_sfu_enqueue(sfu_graph, sfu_memin_port, &sfu_pdmout_buff[0]);
            pi_sfu_enqueue(sfu_graph, sfu_memin_port, &sfu_pdmout_buff[1]);
            //SFU_GraphResetInputs(sfu_graph);
            pi_sfu_reset();
        }

        /* Toggle GPIO (e.g. LED or gpio for measurements)*/
        if ((counter++ % 30) == 0) {
            gpio_val ^= 1;
            pi_gpio_pin_write(PAD_GPIO_LED2, gpio_val);
        }

        /* Set the SoC frequency back to a minimal that allow you to keep fetching data from mics */
        pi_fll_ioctl(PI_FREQ_DOMAIN_FC, PI_FLL_IOCTL_DIV_SET, (void *) 15);

        // int elapsed = gap_fc_readhwtimer() - start;
        // printf("%d (Elapsed: %.2fms - Realtime: %.2fms)\n", elapsed, ((float) elapsed) / (SOC_FREQ / 1000),  ((float) FRAME_STEP) / 16);

        /* Init Events for MEMIN */
        //pi_evt_sig_init(&proc_task_pdmout);
        pi_evt_sig_init(&proc_task_pdmin);
    }

    tinydenoiserCNN_Destruct(0);
    fft_forwardCNN_Destruct(0);
    fft_inverseCNN_Destruct(0);

    pi_dac_stop(ssm6515_device[CHANNEL_LEFT]);
    pi_dac_stop(ssm6515_device[CHANNEL_RIGHT]);

    // Stop I2S transmission
    pi_i2s_ioctl(&i2s_device[CHANNEL_LEFT],  PI_I2S_IOCTL_STOP, NULL);
    pi_i2s_ioctl(&i2s_device[CHANNEL_RIGHT], PI_I2S_IOCTL_STOP, NULL);

    pi_i2s_close(&i2s_device[CHANNEL_LEFT]);
    pi_i2s_close(&i2s_device[CHANNEL_RIGHT]);

    // Closing devices
    pi_close(PI_DAC_SSM6515_LEFT);
    pi_close(PI_DAC_SSM6515_RIGHT);
    
    // End SFU
    pi_sfu_graph_unload(sfu_graph);

    pi_sfu_graph_close(sfu_graph);
    pi_sfu_close();

    return PI_OK;
}
