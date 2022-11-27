/**
 * @file main.c
 * @author dexmac
 * @brief 
 * @version 0.1
 * @date 2022-11-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "c64dnn.h"

#define NEPERO "2.718281828459"
#define ITERATIONS 32

int num_layers;
int *num_neurons;
char temporary_alpha[32];
float alpha;
float *cost;
float full_cost=0;
float **input;
float **desired_outputs;
int num_training_ex;
int n = 1;

// Layers of Neurons
float *actv;
float **out_weights;
float *bias;
float *z;
float *dactv;
float **dw;
float *dbias;
float *dz;
float nepero;

char strbuf[32];

int iterations = ITERATIONS;

/**
 * @brief Learn logical block
 * 
 * @param outputs 
 * @return int 
 */
int logical(int *outputs)
{
    int i;

    //srand(time(0));

    num_layers = 4;

    num_neurons = (int *)malloc(num_layers * sizeof(int));

    memset(num_neurons, 0, num_layers * sizeof(int));

    num_neurons[0] = 2;
    num_neurons[1] = 2;
    num_neurons[2] = 2;
    num_neurons[3] = 1;

    // Initialize the neural network module
    if (init() != SUCCESS_INIT)
    {
        printf("Error in Initialization...\n");
        exit(0);
    }

    alpha = atof("0.15");

    printf("learning rate: %s\n", _ftostr(strbuf, alpha));

    num_training_ex = 4;

    input = (float **)malloc(num_training_ex * sizeof(float *));
    for (i = 0; i < num_training_ex; i++)
    {
        input[i] = (float *)malloc(num_neurons[0] * sizeof(float));
    }

    input[0][0] = itof(0);
    input[0][1] = itof(0);
    input[1][0] = itof(0);
    input[1][1] = itof(1);
    input[2][0] = itof(1);
    input[2][1] = itof(0);
    input[3][0] = itof(1);
    input[3][1] = itof(1);

    desired_outputs = (float **)malloc(num_training_ex * sizeof(float *));
    for (i = 0; i < num_training_ex; i++)
    {
        desired_outputs[i] = (float *)malloc(num_neurons[num_layers - 1] * sizeof(float));
    }

    desired_outputs[0][0] = itof(outputs[0]);
    desired_outputs[1][0] = itof(outputs[1]);
    desired_outputs[2][0] = itof(outputs[2]);
    desired_outputs[3][0] = itof(outputs[3]);

    cost = (float *)malloc(num_neurons[num_layers - 1] * sizeof(float));

    memset(cost, 0, num_neurons[num_layers - 1] * sizeof(float));

    printf("learning rate: %s\n", _ftostr(strbuf, alpha));

    nepero = atof(NEPERO);

    printf("nepero:%s\n", _ftostr(strbuf, nepero));

    getchar();

    train_neural_net();

    printf("\n");

    printf("***************\n");
    printf("*  Inference  *\n");
    printf("***************\n\n");

    printf("Input: 0, 0\n");

    actv[0] = itof(0);
    actv[1] = itof(0);

    forward_prop();

    printf("\n");

    printf("Input: 0, 1\n");

    actv[0] = itof(1);
    actv[1] = itof(0);

    forward_prop();

    printf("\n");

    if (dinit() != SUCCESS_DINIT)
    {
        printf("Error in Dinitialization...\n");
    }

    return 0;
}

/**
 * @brief Interactive with the user
 * 
 * @return int 
 */
int interactive()
{
    int i;

    srand(time(0));

    printf("Enter the number of Iterations:\n");
    scanf("%d", &iterations);

    printf("Enter the number of Layers in Neural Network:\n");
    scanf("%d", &num_layers);

    num_neurons = (int *)malloc(num_layers * sizeof(int));
    memset(num_neurons, 0, num_layers * sizeof(int));

    // Get number of neurons per layer
    for (i = 0; i < num_layers; i++)
    {
        printf("Enter number of neurons in layer[%d]: \n", i + 1);
        scanf("%d", &num_neurons[i]);
    }

    printf("\n");

    // Initialize the neural network module
    if (init() != SUCCESS_INIT)
    {
        printf("Error in Initialization...\n");
        exit(0);
    }

    printf("Enter the learning rate (Usually 0.15): \n");
    scanf("%s", &temporary_alpha);

    alpha = atof(temporary_alpha);

    printf("Learning rate: %s\n", _ftostr(strbuf, alpha));

    printf("Enter the number of training examples: \n");
    scanf("%d", &num_training_ex);
    printf("\n");

    input = (float **)malloc(num_training_ex * sizeof(float *));
    for (i = 0; i < num_training_ex; i++)
    {
        input[i] = (float *)malloc(num_neurons[0] * sizeof(float));
    }

    desired_outputs = (float **)malloc(num_training_ex * sizeof(float *));
    for (i = 0; i < num_training_ex; i++)
    {
        desired_outputs[i] = (float *)malloc(num_neurons[num_layers - 1] * sizeof(float));
    }

    cost = (float *)malloc(num_neurons[num_layers - 1] * sizeof(float));
    memset(cost, 0, num_neurons[num_layers - 1] * sizeof(float));

    nepero = atof(NEPERO);
    printf("nepero:%s\n", _ftostr(strbuf, nepero));

    // Get Training Examples
    get_inputs();

    // Get Output Labels
    get_desired_outputs();

    train_neural_net();
    test_nn();

    if (dinit() != SUCCESS_DINIT)
    {
        printf("Error in Dinitialization...\n");
    }

    return 0;
}

/**
 * @brief Initialize dnn
 * 
 * @return int 
 */
int init()
{
    if (create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
    {
        printf("Error in creating architecture...\n");
        return ERR_INIT;
    }

    printf("Neural Network Created Successfully...\n\n");
    return SUCCESS_INIT;
}

/**
 * @brief Get the inputs from the user
 * 
 */
void get_inputs()
{
    int i, j, temporary_input;

    for (i = 0; i < num_training_ex; i++)
    {
        printf("Enter the Inputs for training example[%d]:\n", i);

        for (j = 0; j < num_neurons[0]; j++)
        {
            scanf("%d", &temporary_input);

            input[i][j] = itof(temporary_input);
        }
        printf("\n");
    }
}

/**
 * @brief Get the desired outputs from the user
 * 
 */
void get_desired_outputs()
{
    int i, j, temporary_outputs;

    for (i = 0; i < num_training_ex; i++)
    {
        for (j = 0; j < num_neurons[num_layers - 1]; j++)
        {
            printf("Enter the Desired Outputs (Labels) for training example[%d]: \n", i);
            scanf("%d", &temporary_outputs);
            desired_outputs[i][j] = itof(temporary_outputs);
            printf("\n");
        }
    }
}

/**
 * @brief Feeds input to input layer
 * 
 * @param i 
 */
void feed_input(int i)
{
    int j;

    for (j = 0; j < num_neurons[0]; j++)
    {
        actv[0 + j] = input[i][j];
        printf("Input: %s\n", _ftostr(strbuf, actv[0 + j]));
    }
}

/**
 * @brief Create neural network architecture
 * 
 * @return int 
 */
int create_architecture()
{
    int i = 0, j = 0;

    actv = (float *)malloc(num_layers * sizeof(float));
    out_weights = (float **)malloc(num_layers * sizeof(float *));
    bias = (float *)malloc(num_layers * sizeof(float));
    z = (float *)malloc(num_layers * sizeof(float));
    dactv = (float *)malloc(num_layers * sizeof(float));
    dw = (float **)malloc(num_layers * sizeof(float *));
    dbias = (float *)malloc(num_layers * sizeof(float));
    dz = (float *)malloc(num_layers * sizeof(float));

    for (i = 0; i < num_layers; i++)
    {
        printf("Created Layer: %d\n", i + 1);
        printf("Number of Neurons in Layer %d: %d\n", i + 1, num_neurons[i]);

        for (j = 0; j < num_neurons[i]; j++)
        {
            // neuron neu;
            if (i < (num_layers - 1))
            {
                // lay[i].neu[j] = create_neuron(num_neurons[i+1]);
                // neu = create_neuron(num_neurons[i+1]);

                actv[j + i] = itof(0);
                out_weights[j + i] = (float *)malloc(num_neurons[i + 1] * sizeof(float));
                bias[j + i] = itof(0);
                z[j + i] = itof(0);

                dactv[j + i] = itof(0);
                dw[j + i] = (float *)malloc(num_neurons[i + 1] * sizeof(float));
                dbias[j + i] = itof(0);
                dz[j + i] = itof(0);
            }

            printf("Neuron %d in Layer %d created\n", j + 1, i + 1);
        }

        printf("\n");
    }

    printf("\n");

    // Initialize the weights
    if (initialize_weights() != SUCCESS_INIT_WEIGHTS)
    {
        printf("Error Initilizing weights...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

/**
 * @brief Initialize weights
 * 
 * @return int 
 */
int initialize_weights(void)
{
    int i, j, k, p;

    printf("Initializing weights...\n");

    for (i = 0; i < num_layers - 1; i++)
    {
        for (j = 0; j < num_neurons[i]; j++)
        {
            for (k = 0; k < num_neurons[i + 1]; k++)
            {
                // Initialize Output Weights for each neuron
                p = i + j;
                out_weights[p][k] = fdiv(itof(rand()), itof(RAND_MAX));

                printf("%d:w[%d][%d]: %s\n", k, i, j, _ftostr(strbuf, out_weights[p][k]));

                dw[p][k] = itof(0);
            }

            if (i > 0)
            {
                p = i + j;
                bias[p] = fdiv(itof(rand()), itof(RAND_MAX));
                printf("%d:bias[%d]: %s\n", p, _ftostr(strbuf, bias[p]));
            }
        }
    }

    printf("\n");

    for (j = 0; j < num_neurons[num_layers - 1]; j++)
    {
        p = num_layers - 1 + j;
        bias[p] = fdiv(itof(rand()), itof(RAND_MAX));
        printf("%d:bias[%d]: %s\n", k, p, _ftostr(strbuf, bias[p]));
    }

    return SUCCESS_INIT_WEIGHTS;
}

/**
 * @brief Train Neural Network
 * 
 */
void train_neural_net(void)
{
    int i;
    int it = 0;

    // Gradient Descent
    for (it = 0; it < iterations; it++)
    {
        printf("Iteration: %d\n", it);
        for (i = 0; i < num_training_ex; i++)
        {
            printf("Training example: %d\n", i);

            feed_input(i);

            forward_prop();
            
            compute_cost(i);
            
            back_prop(i);
            
            update_weights();
        }
    }
}

/**
 * @brief Update weights
 * 
 */
void update_weights(void)
{
    int i, j, k, p;

    for (i = 0; i < num_layers - 1; i++)
    {
        for (j = 0; j < num_neurons[i]; j++)
        {
            p = i + j;

            for (k = 0; k < num_neurons[i + 1]; k++)
            {
                // Update Weights
                out_weights[p][k] = fsub(out_weights[p][k], fmul(alpha, dw[p][k]));
            }

            // Update Bias
            bias[p] = fsub(bias[p], fmul(alpha, dbias[p]));
        }
    }
}

/**
 * @brief Forward propagation
 * 
 */
void forward_prop(void) 
{
    int i, j, k;

    for (i = 1; i < num_layers; i++)
    {
        for (j = 0; j < num_neurons[i]; j++)
        {
            z[i + j] = bias[i + j];

            for (k = 0; k < num_neurons[i - 1]; k++)
            {
                z[i + j] = fadd(z[i + j], fmul((out_weights[i - 1 + k][j]), (actv[i - 1 + k])));
            }

            // Relu Activation Function for Hidden Layers
            if (i < num_layers - 1)
            {
                if (fcmp(z[i + j], itof(0))==1)
                {
                    actv[i + j] = itof(0);
                }
                else
                {
                    actv[i + j] = z[i + j];
                }
            }

            // Sigmoid Activation function for Output Layer
            else
            {
                float exponential = fpow(nepero, fmul(itof(-1),z[i + j]));

                actv[i + j] = fdiv(itof(1), fadd(itof(1), exponential));
                
                printf("Real Output: %s\n", _ftostr(strbuf, actv[i+j]));

                if (fcmp(actv[i + j], atof("0.5"))==255)
                {
                    printf("Output: 1\n");
                }
                else
                {
                    printf("Output: 0\n");
                }

                printf("\n");
            }
        }
    }
}

/**
 * @brief Compute total cost
 * 
 * @param i 
 */
void compute_cost(int i)
{
    int j;
    float tmpcost = 0;
    float tcost = 0;
    float partial_cost = 0;

    for (j = 0; j < num_neurons[num_layers - 1]; j++)
    {
        tmpcost = fsub(desired_outputs[i][j], actv[num_layers - 1 + j]);

        cost[j] = fdiv(fmul(tmpcost, tmpcost), itof(2));
  
        tcost = fadd(tcost, cost[j]);
    }

    partial_cost = fadd(full_cost, tcost);

    full_cost = fdiv(partial_cost, itof(n));

    n++;

    printf("Full Cost: %s\n", _ftostr(strbuf, full_cost));
}

/**
 * @brief Back propagate error
 * 
 * @param p 
 */
void back_prop(int p)
{
    int i, j, k, ps, ks;

    // Output Layer
    for (j = 0; j < num_neurons[num_layers - 1]; j++)
    {
        ps = num_layers - 1 + j;
        
        //printf("desired output %s\n",_ftostr(strbuf,desired_outputs[p][j]));

        dz[ps] = fmul(fsub(actv[ps], desired_outputs[p][j]), fmul(actv[ps], fsub(itof(1), actv[ps])));

        for (k = 0; k < num_neurons[num_layers - 2]; k++)
        {
            ks = num_layers - 2 + k;
            dw[ks][j] = fmul(dz[ps], actv[ks]);
            dactv[ks] = fmul(out_weights[ks][j], dz[ps]);
        }

        dbias[ps] = dz[ps];
    }

    // Hidden Layers
    for (i = num_layers - 2; i > 0; i--)
    {
        for (j = 0; j < num_neurons[i]; j++)
        {
            ps = i + j;

            if (fcmp(z[ps], itof(0)!=1))
            {
                dz[ps] = dactv[ps];
            }
            else
            {
                dz[ps] = 0;
            }

            for (k = 0; k < num_neurons[i - 1]; k++)
            {
                ks = i - 1 + k;

                dw[ks][j] = fmul(dz[ps], actv[ks]);

                if (i > 1)
                {
                    dactv[ks] = fmul(out_weights[ks][j], dz[ps]);
                }
            }

            dbias[ps] = dz[ps];
        }
    }
}

/**
 * @brief Test the neural network
 * 
 */
void test_nn(void)
{
    int i;
    while (1)
    {
        printf("Enter input to test:\n");

        for (i = 0; i < num_neurons[0]; i++)
        {
            scanf("%f", &actv[i]);
        }

        forward_prop();
    }
}

/**
 * @brief Deinitialize
 * 
 * @return int 
 */
int dinit(void)
{
    // TODO:
    // Free up all the structures

    return SUCCESS_DINIT;
}

/**
 * @brief Main
 * 
 * @return int 
 */
int main(){
    int desired[] = {0,1,1,0};
    int select = 0;

#ifdef __CC65__
    srand(time(NULL));
    bordercolor (0);
	bgcolor (0);
    textcolor (5);
	clrscr();   
#endif

    printf("c64dnn Deep Neural Network Simulator\nby Dexmac (2022) V0.1b\n");
    printf("Select:\n");
    printf("1)preloaded xor\n");
    printf("2)interactive\n");

    scanf("%d",&select);
    
    switch(select){
        case 1:
            logical(desired);
        break;

        case 2:
            interactive();
        break;

        default:
            break;
    }

    return 0;
}