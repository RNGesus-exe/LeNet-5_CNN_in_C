#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <math.h>

/*
        LeNet-5 CNN Architecture
        1) The input image is 28x28x1 but we need to apply padding (p=1) with zeros and make it 32x32x1 b/c layer 1 will shrink the image back to 28x28x1
        2) The first layer is a convolution layer which first performs "y = W.X + b" for linear function then performs a "tanh()" as activation function
            2.1) The layer 1 will output 6 filter maps each having dimensions of 28x28x1
            2.2) The second part of convolution layer includes the "average pool kernel"

        Formulae:
            Dimension of output matrix for padding, stride and, kernel_size
            1) N_out = floor( ( N_in + 2 * padding - filter_size) / stride + 1 )
            2) M_out = floor( ( M_in + 2 * padding - filter_size) / stride + 1 )

        Activation Function:
            1) tanh()

        Linear Function:
            1) y = W.X + b

        Subsampling Function:
            1) average_pool

        Helper Function:
            1) Apply padding (We won't be sending matrices for padding apparently)
*/

#define COMPUTE_OUTPUT_SIZE(N_in, padding, filter_size, stride) \
    (int)floor(((N_in) + 2 * (padding) - (filter_size)) / (float)(stride) + 1)

// Layer 1 - convolution + tanh
#define INPUT_ROWS_1 32
#define INPUT_COLS_1 32
#define KERNEL_SIZE_1 5
#define STRIDE_1 1
#define PADDING_1 0
#define NUM_FILTERS_1 6 // output channel of layer 1
#define OUTPUT_ROWS_1 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_1, PADDING_1, KERNEL_SIZE_1, STRIDE_1)
#define OUTPUT_COLS_1 COMPUTE_OUTPUT_SIZE(INPUT_COLS_1, PADDING_1, KERNEL_SIZE_1, STRIDE_1)

// Layer 2 - Average Pooling
#define INPUT_ROWS_2 OUPUT_ROWS_1
#define INPUT_COLS_2 OUTPUT_COLS_1
#define KERNEL_SIZE_2 2
#define STRIDE_2 2
#define PADDING_2 0
#define NUM_FILTERS_2 6 // output channel of layer 2
#define OUTPUT_ROWS_2 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_2, PADDING_2, KERNEL_SIZE_2, STRIDE_2)
#define OUTPUT_COLS_2 COMPUTE_OUTPUT_SIZE(INPUT_COLS_2, PADDING_2, KERNEL_SIZE_2, STRIDE_2)

// Define the LeNet-5 architecture
typedef struct
{
    float weights1[NUM_FILTERS_1][KERNEL_SIZE_1][KERNEL_SIZE_1];
    float biases1[NUM_FILTERS_1];
} LeNet5;

// hyperbolic tan activation function
float tanh_activation(float x)
{
    return tanh(x);
}

// Initialize the LeNet-5 model
void initLeNet5(LeNet5 *model)
{
    FILE *file = fopen("parameters.txt", "r");
    if (file == NULL)
    {
        printf("Failed to open the file.\n");
        return;
    }

    // Read weights of layer 1
    for (int c = 0; c < NUM_FILTERS_1; c++)
    {
        for (int i = 0; i < KERNEL_SIZE_1; i++)
        {
            for (int j = 0; j < KERNEL_SIZE_1; j++)
            {
                fscanf(file, "%f", model->weights1[c][i][j]);
            }
        }
    }

    // Read biases of layer 1
    for (int c = 0; c < NUM_FILTERS_1; c++)
    {
        fscanf(file, "%f", &model->biases1[c]);
    }

    fclose(file);
}

void convolution(float **input_matrix, int input_rows, int input_cols,
                 float ***kernel, int kernel_rows, int kernel_cols,
                 float *bias, int no_of_filters,
                 float ***layer, int output_rows, int output_cols,
                 int padding, int stride, float (*activation)(float))
{
    for (int filter_no = 0; filter_no < no_of_filters; filter_no++)
    {
        for (int input_row = 0; input_row < output_rows; input_row++)
        {
            for (int input_col = 0; input_col < output_cols; input_col++)
            {
                float sum = 0.0;

                // Dot product
                for (int kernel_row = 0; kernel_row < kernel_rows; kernel_row++)
                {
                    for (int kernel_col = 0; kernel_col < kernel_cols; kernel_col++)
                    {
                        sum += input_matrix[input_row * stride + kernel_row][input_col * stride + kernel_col] * kernel[filter_no][kernel_row][kernel_col];
                    }
                }

                // Feature Map
                layer[filter_no][input_row][input_col] = (*activation)(sum + bias[filter_no]);
            }
        }
    }
}

#endif // CNN_H
