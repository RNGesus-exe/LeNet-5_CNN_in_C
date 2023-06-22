#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
    Optimization TODO (Dr. Umar):
        1) Convert Floating points to Integer
        2) Apply Vectorization (SIMD)
        3) Apply Row Wise operations

    Dimension:
        batch_size x channels x rows x columns

    Dataset:
        Mnist [handwritten][0-9]

    LeNet-5 CNN Architecture:
        1) The input layer image is 1x1x28x28
        2) The first hidden layer is a convolution layer [p=2,s=1,f=5,a=tanh] which outputs 1x1x28x28
        3) The second layer is a subsampling [p=0,s=2,f=2,a=avg_pool] which outputs 1x6x14x14
        4) The third layer is a convolution [p=0,s=1,f=5,a=tanh] which outputs 1x6x10x10
        5) The fourth layer is a subsampling [p=0,s=2,f=2,a=avg_pool] which outputs 1x16x5x5
        6) The fifth layer is a convolution [p=0,s=1,f=5,a=tanh] which outputs 1x120x1x1
        7) We will flatten the 1x120x1x1 feature maps into 1x120
        8) The sixth layer is fully connected [a=tanh] which outputs 1x84x1x1
        9) The seventh layer is a fully connected [a=softmax] which outputs 1x10x1x1
        10) The output layer will have an argmax() to tell us which neuron from seventh layer had highest probability

    Formulae:
        Dimension of output matrix for padding, stride and, kernel_size
        1) N_out = floor( ( N_in + 2 * padding - filter_size) / stride + 1 )
        2) M_out = floor( ( M_in + 2 * padding - filter_size) / stride + 1 )

    Activation Function:
        1) tanh()
        2) softmax()

    Subsampling Function:
        1) average_pool
*/

//-------------------------------------------------------------MACROS/GLOBALS---------------------------------------------------//

#define MAX_LINE_SIZE 100000

#define COMPUTE_OUTPUT_SIZE(N_in, padding, filter_size, stride) \
    (int)floor(((N_in) + 2 * (padding) - (filter_size)) / (float)(stride) + 1)

//-------------------------------------------------------------Lenet-5 Configuration-------------------------------------------//

// Layer 1 - convolution + tanh
#define INPUT_ROWS_1 32
#define INPUT_COLS_1 32
#define KERNEL_SIZE_1 5
#define STRIDE_1 1
#define PADDING_1 0
#define NUM_FILTERS_1 6
#define OUTPUT_ROWS_1 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_1, PADDING_1, KERNEL_SIZE_1, STRIDE_1)
#define OUTPUT_COLS_1 COMPUTE_OUTPUT_SIZE(INPUT_COLS_1, PADDING_1, KERNEL_SIZE_1, STRIDE_1)

// Layer 2 - Average Pooling
#define INPUT_ROWS_2 OUPUT_ROWS_1
#define INPUT_COLS_2 OUTPUT_COLS_1
#define KERNEL_SIZE_2 2
#define STRIDE_2 2
#define PADDING_2 0
#define NUM_FILTERS_2 6
#define OUTPUT_ROWS_2 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_2, PADDING_2, KERNEL_SIZE_2, STRIDE_2)
#define OUTPUT_COLS_2 COMPUTE_OUTPUT_SIZE(INPUT_COLS_2, PADDING_2, KERNEL_SIZE_2, STRIDE_2)

// Layer 3 - convolution + tanh
#define INPUT_ROWS_3 OUPUT_ROWS_2
#define INPUT_COLS_3 OUTPUT_COLS_2
#define KERNEL_SIZE_3 5
#define STRIDE_3 1
#define PADDING_3 0
#define NUM_FILTERS_3 16
#define OUTPUT_ROWS_3 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_3, PADDING_3, KERNEL_SIZE_3, STRIDE_3)
#define OUTPUT_COLS_3 COMPUTE_OUTPUT_SIZE(INPUT_COLS_3, PADDING_3, KERNEL_SIZE_3, STRIDE_3)

// Layer 4 - Average Pooling
#define INPUT_ROWS_4 OUPUT_ROWS_3
#define INPUT_COLS_4 OUTPUT_COLS_3
#define KERNEL_SIZE_4 2
#define STRIDE_4 2
#define PADDING_4 0
#define NUM_FILTERS_4 16
#define OUTPUT_ROWS_4 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_4, PADDING_4, KERNEL_SIZE_4, STRIDE_4)
#define OUTPUT_COLS_4 COMPUTE_OUTPUT_SIZE(INPUT_COLS_4, PADDING_4, KERNEL_SIZE_4, STRIDE_4)

// Layer 5 - convolution + tanh
#define INPUT_ROWS_5 OUPUT_ROWS_4
#define INPUT_COLS_5 OUTPUT_COLS_4
#define KERNEL_SIZE_5 5
#define STRIDE_5 1
#define PADDING_5 0
#define NUM_FILTERS_5 120
#define OUTPUT_ROWS_5 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_5, PADDING_5, KERNEL_SIZE_5, STRIDE_5)
#define OUTPUT_COLS_5 COMPUTE_OUTPUT_SIZE(INPUT_COLS_5, PADDING_5, KERNEL_SIZE_5, STRIDE_5)

// NOTE: We will Flatten the layers before passing to Fully Connected Layers

// Layer 6 - Fully Connected + tanh
#define INPUT_ROWS_6 1
#define INPUT_COLS_6 120
#define WEIGHT_ROWS_6 120
#define WEIGHT_COLS_6 1
#define NUM_FILTERS_6 84

// Layer 7 - Fully Connected + softmax
#define INPUT_ROWS_7 1
#define INPUT_COLS_7 84
#define WEIGHT_ROWS_7 84
#define WEIGHT_COLS_7 1
#define NUM_FILTERS_7 10

// Define the LeNet-5 architecture
typedef struct
{
    //conv 1 (6x5x5)
    float weights1[NUM_FILTERS_1][KERNEL_SIZE_1][KERNEL_SIZE_1];
    float biases1[NUM_FILTERS_1];
    //conv 2 (16x5x5)
    float weights2[NUM_FILTERS_3][KERNEL_SIZE_3][KERNEL_SIZE_3];
    float biases2[NUM_FILTERS_3];
    //conv 3 (120x5x5)
    float weights3[NUM_FILTERS_5][KERNEL_SIZE_5][KERNEL_SIZE_5];
    float biases3[NUM_FILTERS_5];
    //fc 4 (84x120x1)
    float weights4[NUM_FILTERS_6][WEIGHT_ROWS_6];
    float biases4[NUM_FILTERS_6];
    //fc 5 (10x84x1)
    float weights5[NUM_FILTERS_7][WEIGHT_ROWS_7];
    float biases5[NUM_FILTERS_7];
} LeNet5;

// We load an image now, In the future we can load multiple images
void loadDataset(float input[INPUT_ROWS_1][INPUT_COLS_1],const char* filename){
    
    FILE* file = fopen(filename,"rb");
    
    if(file == NULL){
        printf("Failed to open the file\n");
        return;
    }

    char line[MAX_LINE_SIZE];
    if(fgets(line,sizeof(line),file)){
        
        char* token;
        token = strtok(line, ",");   //SKIP LABEL
        token = strtok(NULL,",");
        for(int i=2; i<INPUT_ROWS_1-2; i++){
            for(int j=2; j<INPUT_COLS_1-2; j++){
                input[i][j] = atof(token)/255;
                token = strtok(NULL,",");
            }
        }
    }

    fclose(file);
}

// Initialize the LeNet-5 model
void initLeNet5(LeNet5 *model)
{
    FILE *file = fopen("parameters.txt", "rb");
    if (file == NULL)
    {
        printf("Failed to open the file.\n");
        return;
    }

    char line[MAX_LINE_SIZE];

    // Read weights of layer 1
    if(fgets(line,MAX_LINE_SIZE,file)){
        
        char* token;
        token = strtok(line, " ");
     
        for (int c = 0; c < NUM_FILTERS_1; c++)
        {
            for (int i = 0; i < KERNEL_SIZE_1; i++)
            {
                for (int j = 0; j < KERNEL_SIZE_1; j++)
                {
                    model->weights1[c][i][j] = atof(token);
                    token = strtok(NULL," ");
                }
            }
        }   
    }

    // Read biases of layer 1
    if(fgets(line, MAX_LINE_SIZE,file)){
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_1; c++)
        {
            model->biases1[c] = atof(token);
            token = strtok(NULL," ");
        }
    }

    // Read weights of layer 3
    if(fgets(line,MAX_LINE_SIZE,file)){
        
        char* token;
        token = strtok(line, " ");
     
        for (int c = 0; c < NUM_FILTERS_3; c++)
        {
            for (int i = 0; i < KERNEL_SIZE_3; i++)
            {
                for (int j = 0; j < KERNEL_SIZE_3; j++)
                {
                    model->weights2[c][i][j] = atof(token);
                    token = strtok(NULL," ");
                }
            }
        }   
    }

    // Read biases of layer 3
    if(fgets(line, MAX_LINE_SIZE,file)){
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_3; c++)
        {
            model->biases2[c] = atof(token);
            token = strtok(NULL," ");
        }
    }

    // Read weights of layer 5
    if(fgets(line,MAX_LINE_SIZE,file)){
        
        char* token;
        token = strtok(line, " ");
     
        for (int c = 0; c < NUM_FILTERS_5; c++)
        {
            for (int i = 0; i < KERNEL_SIZE_5; i++)
            {
                for (int j = 0; j < KERNEL_SIZE_5; j++)
                {
                    model->weights3[c][i][j] = atof(token);
                    token = strtok(NULL," ");
                }
            }
        }   
    }

    // Read biases of layer 5
    if(fgets(line, MAX_LINE_SIZE,file)){
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_5; c++)
        {
            model->biases3[c] = atof(token);
            token = strtok(NULL," ");
        }
    }    


    // Read weights of layer 6
    if(fgets(line,MAX_LINE_SIZE,file)){
        
        char* token;
        token = strtok(line, " ");
     
        for (int c = 0; c < NUM_FILTERS_6; c++)
        {
            for (int i = 0; i < WEIGHT_ROWS_6; i++)
            {
                model->weights4[c][i] = atof(token);
                token = strtok(NULL," ");
            }
        }   
    }

    // Read biases of layer 6
    if(fgets(line, MAX_LINE_SIZE,file)){
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_6; c++)
        {
            model->biases4[c] = atof(token);
            token = strtok(NULL," ");
        }
    }

    // Read weights of layer 7
    if(fgets(line,MAX_LINE_SIZE,file)){
        
        char* token;
        token = strtok(line, " ");
     
        for (int c = 0; c < NUM_FILTERS_7; c++)
        {
            for (int i = 0; i < WEIGHT_ROWS_7; i++)
            {
                model->weights5[c][i] = atof(token);
                token = strtok(NULL," ");
            }
        }   
    }

    // Read biases of layer 7
    if(fgets(line, MAX_LINE_SIZE,file)){
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_7; c++)
        {
            model->biases5[c] = atof(token);
            token = strtok(NULL," ");
        }
    }

    fclose(file);
}

void layer_2_subsampling(double input_matrix[INPUT_ROWS_1][INPUT_COLS_1],
                 LeNet5* model, double layer1[NUM_FILTERS_1][OUTPUT_ROWS_1][OUTPUT_COLS_1])
{
    for(int filter_no = 0;filter_no < NUM_FILTERS_1;filter_no++){

    }
}

void layer_1_conv(float input_matrix[INPUT_ROWS_1][INPUT_COLS_1], LeNet5* model, float layer1[NUM_FILTERS_1][OUTPUT_ROWS_1][OUTPUT_COLS_1])
{
    for (int filter_no = 0; filter_no < NUM_FILTERS_1; filter_no++)
    {
        for (int input_row = 0; input_row < OUTPUT_ROWS_1; input_row++)
        {
            for (int input_col = 0; input_col < OUTPUT_COLS_1; input_col++)
            {
                float sum = 0.0;

                // Dot product
                for (int kernel_row = 0; kernel_row < KERNEL_SIZE_1; kernel_row++)
                {
                    for (int kernel_col = 0; kernel_col < KERNEL_SIZE_1; kernel_col++)
                    {
                        sum += input_matrix[input_row + kernel_row][input_col + kernel_col] * model->weights1[filter_no][kernel_row][kernel_col];
                    }
                }

                // Feature Map
                // layer1[filter_no][input_row][input_col] = sum;
                layer1[filter_no][input_row][input_col] = tanh((sum + model->biases1[filter_no]));
            }
        }
    }
}

#endif // CNN_H
