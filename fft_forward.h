#ifndef __fft_forward_H__
#define __fft_forward_H__

// Include basic GAP builtins defined in the Autotiler
#include "Gap.h"

#ifdef __EMUL__
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/param.h>
#include <string.h>
#endif

extern AT_DEFAULTFLASH_EXT_ADDR_TYPE fft_forward_L3_Flash;
#endif