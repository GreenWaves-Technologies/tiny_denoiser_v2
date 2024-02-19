# Copyright (c) 2022 GreenWaves Technologies SAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of GreenWaves Technologies SAS nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.16)

###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

###############################################################################
# Application related definitions
###############################################################################

set(TARGET_NAME         "denoiser")
set(APP_MSG_PREFIX "[${BoldRed}${TARGET_NAME}${ColourReset}]")

if (${CONFIG_DEMO})
  set(MAIN              ${CMAKE_CURRENT_SOURCE_DIR}/main_demo.c)
elseif(${CONFIG_DENOISE_WAV})
  set(MAIN              ${CMAKE_CURRENT_SOURCE_DIR}/main_wav.c)
else()
  message(ERROR "${APP_MSG_PREFIX} No demo mode chosen")
endif()

set(WAV_FILE  ${CMAKE_CURRENT_SOURCE_DIR}/dataset/test/noisy/p232_050.wav)

set(FFT_NNTOOL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/nntool_scripts/fft_nntool_script.py)
set(NN_NNTOOL_SCRIPT  ${CMAKE_CURRENT_SOURCE_DIR}/nntool_scripts/nn_nntool_script.py)
set(QUANT_DATASET     ${CMAKE_CURRENT_SOURCE_DIR}/dataset/quant/*)

if (${CONFIG_MODEL_FLASH_TYPE_DEFAULT})
  set(FLASH_TYPE flash)
else()
  set(FLASH_TYPE mram)
endif()


if (${CONFIG_GRU_DNS})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/denoiser_GRU_dns.onnx)
  set(GRU_FLAG -DGRU)
elseif(${CONFIG_GRU_VALENTINI})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/denoiser_GRU_Valentini.onnx)
  set(GRU_FLAG -DGRU)
elseif(${CONFIG_LSTM_VALENTINI})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/denoiser_LSTM_Valentini.onnx)
elseif(${CONFIG_NEW_LSTM_DNS})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/FullRankLSTM.onnx)
  elseif(${CONFIG_NEW_GRU_DNS})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/FullRankGRU.onnx)
  set(GRU_FLAG -DGRU)
elseif(${CONFIG_NEW_LSTM_DNS_TT_RANK_2})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/tensor_train/tt_denoiser_rank_2.onnx)
  set(TT_FLAG -DTT)
elseif(${CONFIG_NEW_LSTM_DNS_TT_RANK_4})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/tensor_train/tt_denoiser_rank_4.onnx)
  set(TT_FLAG -DTT)
elseif(${CONFIG_NEW_LSTM_DNS_TT_RANK_8})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/tensor_train/tt_denoiser_rank_8.onnx)
  set(TT_FLAG -DTT)
elseif(${CONFIG_NEW_LSTM_DNS_TT_RANK_16})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/tensor_train/tt_denoiser_rank_16.onnx)
  set(TT_FLAG -DTT)
elseif(${CONFIG_NEW_LSTM_DNS_TT_RANK_48})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/tensor_train/tt_denoiser_rank_48.onnx)
  set(TT_FLAG -DTT)
elseif(${CONFIG_NEW_LSTM_DNS_TT_RANK_80})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/tensor_train/tt_denoiser_rank_80.onnx)
  set(TT_FLAG -DTT)
elseif(${CONFIG_NEW_LSTM_DNS_TT_FULL_RANK})
  set(TRAINED_ONNX      ${CMAKE_CURRENT_SOURCE_DIR}/model/tensor_train/tt_TinyDenoiser_FullRankLSTM.onnx)
  set(TT_FLAG -DTT)
  set(TT_FULL_FLAG -DTT_FULL_RANK)
  set(NNTOOL_TT_FLAG --tensor_train)
else()
  message(ERROR "${APP_MSG_PREFIX} Model selection went wrong!")
endif()

message(STATUS "${APP_MSG_PREFIX} Using ${TRAINED_ONNX} model")

###############################################################################
# Application's kconfig options
###############################################################################

if (CONFIG_FP16MIXED)
  set(QUANT_TYPE "mixedfp16")
  list(APPEND QUANT_FLAG -DNN_TYPE=f16)
  list(APPEND QUANT_FLAG -DRNN_TYPE_H=int8_t)
  list(APPEND QUANT_FLAG -DRNN_TYPE_C=int8_t)
  list(APPEND QUANT_FLAG -DZERO=0)
elseif(CONFIG_FP16)
  set(QUANT_TYPE "fp16")
  list(APPEND QUANT_FLAG -DNN_TYPE=f16)
  list(APPEND QUANT_FLAG -DRNN_TYPE_H=f16)
  list(APPEND QUANT_FLAG -DRNN_TYPE_C=f16)
  list(APPEND QUANT_FLAG -DZERO=0)
elseif(CONFIG_FP16NE16MIXED)
  set(QUANT_TYPE "mixedne16fp16")
  list(APPEND QUANT_FLAG -DNN_TYPE=f16)
  list(APPEND QUANT_FLAG -DRNN_TYPE_H=uint8_t)
  list(APPEND QUANT_FLAG -DRNN_TYPE_C=int16_t)
  list(APPEND QUANT_FLAG -DZERO=128)
else()
  message(ERROR "${APP_MSG_PREFIX} You need to provide a valid quant type")
endif()
list(APPEND QUANT_FLAG -DSTFT_TYPE=float)

###############################################################################
# FFT Pre/post processing code gen
###############################################################################

set(H_STATE_LEN          256)
set(FRAME_SIZE           400)
set(FRAME_STEP           100)
set(FRAME_NFFT           512)

###############################################################################
# Autotiler Variables
###############################################################################

set(FFT_FORWARD_DIR  ${CMAKE_BINARY_DIR}/BUILD_MODEL_FFT)
set(FFT_FORWARD_AT_MODEL_PATH      ${FFT_FORWARD_DIR}/fft_forward_model.c)
set(FFT_FORWARD_AT_TENSORS_DIR     ${FFT_FORWARD_DIR}/twiddles)
autotiler_add_model(
  TARGET_NAME "fft_forward"
  MODEL_PATH ${FFT_FORWARD_AT_MODEL_PATH}
  MODEL_TENSORS_DIR ${FFT_FORWARD_AT_TENSORS_DIR}
  MODEL_BUILDDIR ${FFT_FORWARD_DIR}
  PRI1_USED TRUE
  PRI1_FLASH ${FLASH_TYPE}
  PRI1_FILENAME "L3_Flash_Const.dat"
  L1_MEMORY ${CONFIG_MODEL_L1_MEMORY}
  L2_MEMORY ${CONFIG_MODEL_L2_MEMORY}
  L3_MEMORY ${CONFIG_MODEL_L3_MEMORY}
  L2STATIC_MEMORY ${CONFIG_MODEL_L2STATIC_MEMORY}
  L3STATIC_MEMORY ${CONFIG_MODEL_L3STATIC_MEMORY}
)

set(FFT_INVERSE_DIR  ${CMAKE_BINARY_DIR}/BUILD_MODEL_IFFT)
set(FFT_INVERSE_AT_MODEL_PATH      ${FFT_INVERSE_DIR}/fft_inverse_model.c)
set(FFT_INVERSE_AT_TENSORS_DIR     ${FFT_INVERSE_DIR}/twiddles)
autotiler_add_model(
  TARGET_NAME "fft_inverse"
  MODEL_PATH ${FFT_INVERSE_AT_MODEL_PATH}
  MODEL_TENSORS_DIR ${FFT_INVERSE_AT_TENSORS_DIR}
  MODEL_BUILDDIR ${FFT_INVERSE_DIR}
  PRI1_USED TRUE
  PRI1_FLASH ${FLASH_TYPE}
  PRI1_FILENAME "L3_Flash_Const.dat"
  L1_MEMORY ${CONFIG_MODEL_L1_MEMORY}
  L2_MEMORY ${CONFIG_MODEL_L2_MEMORY}
  L3_MEMORY ${CONFIG_MODEL_L3_MEMORY}
  L2STATIC_MEMORY ${CONFIG_MODEL_L2STATIC_MEMORY}
  L3STATIC_MEMORY ${CONFIG_MODEL_L3STATIC_MEMORY}
)

set(NN_DIR  ${CMAKE_BINARY_DIR}/BUILD_MODEL_NN)
set(NN_AT_MODEL_PATH      ${NN_DIR}/tinydenoiserModel.c)
set(NN_AT_TENSORS_DIR     ${NN_DIR}/tensors)
set(NN_EXPRESSION_KERNELS ${NN_DIR}/NN_Expression_Kernels.c)
autotiler_add_model(
  TARGET_NAME "tinydenoiser"
  MODEL_PATH ${NN_AT_MODEL_PATH}
  MODEL_TENSORS_DIR ${NN_AT_TENSORS_DIR}
  MODEL_BUILDDIR ${NN_DIR}
  PRI1_USED TRUE
  PRI1_FLASH ${FLASH_TYPE}
  PRI1_FILENAME "L3_Flash_Const.dat"
  L1_MEMORY ${CONFIG_MODEL_L1_MEMORY}
  L2_MEMORY ${CONFIG_MODEL_L2_MEMORY}
  L3_MEMORY ${CONFIG_MODEL_L3_MEMORY}
  L2STATIC_MEMORY ${CONFIG_MODEL_L2STATIC_MEMORY}
  L3STATIC_MEMORY ${CONFIG_MODEL_L3STATIC_MEMORY}
)

###############################################################################
# NNTool scripts
###############################################################################

add_custom_command(
  OUTPUT ${FFT_FORWARD_AT_MODEL_PATH} ${FFT_FORWARD_AT_TENSORS_DIR} ${FFT_INVERSE_AT_MODEL_PATH} ${FFT_INVERSE_AT_TENSORS_DIR}
  COMMENT "Generate the FFT autotiler model via ${FFT_NNTOOL_SCRIPT}"
  DEPENDS ${FFT_NNTOOL_SCRIPT}
  COMMAND python3 ${FFT_NNTOOL_SCRIPT}
              --n_fft=${FRAME_NFFT}
              --frame_size=${FRAME_SIZE}
              --frame_step=${FRAME_STEP}
              --window_type="hann"
              --forward_at_model_path=${FFT_FORWARD_AT_MODEL_PATH}
              --forward_tensors_dir=${FFT_FORWARD_AT_TENSORS_DIR}
              --inverse_at_model_path=${FFT_INVERSE_AT_MODEL_PATH}
              --inverse_tensors_dir=${FFT_INVERSE_AT_TENSORS_DIR}
              --flash_type=${FLASH_TYPE}
)
add_custom_target(clean_fft_model
  COMMENT "clean_fft_model: removing ${COMMON_MODEL_BUILDDIR}/${MODEL_GEN}"
  COMMAND rm -rf ${FFT_FORWARD_DIR} ${FFT_INVERSE_DIR}
)

add_custom_command(
  OUTPUT ${NN_AT_MODEL_PATH} ${NN_AT_TENSORS_DIR} ${NN_EXPRESSION_KERNELS}
  COMMENT "Generate the autotiler model: ${NN_AT_MODEL_PATH}"
  DEPENDS ${NN_NNTOOL_SCRIPT}
  COMMAND python3 ${NN_NNTOOL_SCRIPT} --trained_model=${TRAINED_ONNX}
                                      --quant_dataset=${QUANT_DATASET}
                                      --quant_type=${QUANT_TYPE}
                                      --at_model_path=${NN_AT_MODEL_PATH}
                                      --tensors_dir=${NN_AT_TENSORS_DIR}
                                      --flash_type=${FLASH_TYPE}
                                      ${NNTOOL_TT_FLAG}
)
add_custom_target(clean_nn_model
  COMMENT "clean_nn_model: removing ${COMMON_MODEL_BUILDDIR}/${MODEL_GEN}"
  COMMAND rm -rf ${NN_AT_MODEL_PATH}
)

###############################################################################
# Building SFU Graph
###############################################################################
set(GRAPH_FILE   ${CMAKE_CURRENT_SOURCE_DIR}/SFUGraph.src)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/SFUGraph_L2_Descr.c ${CMAKE_BINARY_DIR}/SFUGraph_L2_Descr.h
  COMMENT "Generate SFU Graph"
  DEPENDS ${GRAPH_FILE}
  COMMAND SFU -i ${GRAPH_FILE} -C
)
add_custom_target(gen_sfu DEPENDS ${CMAKE_BINARY_DIR}/SFUGraph_L2_Descr.c ${CMAKE_BINARY_DIR}/SFUGraph_L2_Descr.h)

###############################################################################
# Application dependencies
###############################################################################
list(APPEND TARGET_SRCS ${MAIN}
                        ${CMAKE_SOURCE_DIR}/cluster_fn.c
                        ${CNN_LIB}
                        ${KERNEL_C_PATHS}
                        ${SFU_KERNEL_C}
                        $ENV{SFU_RUNTIME}/SFU_RT.c
                        ${CMAKE_BINARY_DIR}/SFUGraph_L2_Descr.c
                        ${NN_EXPRESSION_KERNELS}
)

list(APPEND TARGET_INCS -I$ENV{GAP_H_INC}
                        -I${TILER_INC}
                        -I${TILER_EMU_INC}
                        -I${SFU_BUILDDIR}
                        -I$ENV{SFU_RUNTIME}/include
                        -I${GAP_SDK_HOME}/libs/gap_lib/include
                        ${CNN_LIB_INCLUDE}
                        -I${COMMON_MODEL_BUILDDIR}
                        -I${FFT_FORWARD_DIR}
                        -I${FFT_INVERSE_DIR}
                        -I${NN_DIR}
                        -I${COMMON_MODEL_BUILDDIR}
                        -I${FFT_BUILD_DIR}
                        -I${CMAKE_CURRENT_SOURCE_DIR}
                        -I${CMAKE_BINARY_DIR}
                        -I${GAP_SDK_HOME}/utils/power_meas_utils
                        -I${CMAKE_CURRENT_SOURCE_DIR}/samples
)

list(APPEND TARGET_CFLAGS   -gdwarf-2
                            -gstrict-dwarf
                            -O3
                            -mno-memcpy
                            -fno-tree-loop-distribute-patterns
                            -Wno-incompatible-pointer-types)

math(EXPR FREQ_CL_MHZ "${CONFIG_FREQUENCY_CLUSTER}/1000/1000")
math(EXPR FREQ_FC_MHZ "${CONFIG_FREQUENCY_FC}/1000/1000")
math(EXPR FREQ_SFU_MHZ "${CONFIG_FREQUENCY_SFU}/1000/1000")
list(APPEND TARGET_PREPROCESSOR -DPERF
                                -DSTACK_SIZE=${CONFIG_CL_MASTER_CORE_STACK_SIZE}
                                -DSLAVE_STACK_SIZE=${CONFIG_CL_SLAVE_CORE_STACK_SIZE}
                                -DFREQ_CL=${FREQ_CL_MHZ}
                                -DFREQ_FC=${FREQ_FC_MHZ}
                                -DFREQ_SFU=${FREQ_SFU_MHZ}
                                -DWAV_FILE=${WAV_FILE}
                                -DOUT_FILE=${CMAKE_BINARY_DIR}/denoised_output.wav
                                -DFRAME_SIZE=${FRAME_NFFT}
                                -DFRAME_STEP=${FRAME_STEP}
                                -DN_FFT=${FRAME_NFFT}
                                ${QUANT_FLAG}
                                ${GRU_FLAG}
                                ${TT_FLAG}
                                ${TT_FULL_FLAG}
)

project(${TARGET_NAME} C ASM)
add_executable(${TARGET_NAME} ${TARGET_SRCS})

target_compile_options(${TARGET_NAME} PUBLIC    ${TARGET_CFLAGS} 
                                                ${TARGET_INCS}
                                                ${TARGET_PREPROCESSOR})
message(STATUS ${TARGET_PREPROCESSOR})

###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})
