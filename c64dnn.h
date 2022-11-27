/**
 * @file c64dnn.h
 * @author dexmac
 * @brief 
 * @version 0.1
 * @date 2022-11-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef C64DNN_H
#define C64DNN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include "float.h"
#include <time.h>

#ifdef __CC65__
#include <peekpoke.h>
#include <joystick.h>
#include <conio.h>
#endif

#define SUCCESS_INIT 0
#define ERR_INIT 1

#define SUCCESS_DINIT 0
#define ERR_DINIT 1

#define SUCCESS_INIT_WEIGHTS 0
#define ERR_INIT_WEIGHTS 1

#define SUCCESS_UPDATE_WEIGHTS 0

#define SUCCESS_CREATE_ARCHITECTURE 0
#define ERR_CREATE_ARCHITECTURE 1


int init(void);
int dinit(void);

int create_architecture(void);
int initialize_weights(void);
void feed_input(int i);
void train_neural_net(void);
void forward_prop(void);
void compute_cost(int i);
void back_prop(int p);
void update_weights(void);
void get_inputs(void);
void get_desired_outputs(void);
void test_nn(void);

int initialize_dummy_weights(void);


#endif



