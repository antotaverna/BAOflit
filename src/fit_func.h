/*******************************************************************************
* fit_func.h: this file is part of the BAOflit program.

* BAOflit: Baryon Acoustic Oscillation Fitter for muLtI-Tracers.

* Github repository:
        https://github.com/cheng-zhao/BAOflit

* Copyright (c) 2021 Cheng Zhao <zhaocheng03@gmail.com>
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

*******************************************************************************/

#ifndef __FIT_FUNC_H__
#define __FIT_FUNC_H__

#include "load_conf.h"
#include "fit_args.h"

/*============================================================================*\
                      Functions for likelihood evaluation
\*============================================================================*/

/******************************************************************************
Function `run_multinest`:
  Perform the MultiNest fit.
Arguments:
  * `conf`:     the structure for storing configurations;
  * `fit`:      the structure for storing information for the fit.
******************************************************************************/
void run_multinest(const CONF *conf, ARGS *args);

#endif
