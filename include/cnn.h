#ifndef CNN_H
#define CNN_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
    Optimization TODO:
        1) Convert Floating points to Integer
        2) Apply Vectorization (SIMD)
        3) Apply Row Wise operations

    Dimension:
        batch_size x channels x rows x columns

    Dataset:
        Mnist [handwritten][0-9]

    LeNet-5 CNN Architecture:
        1) The input layer image is 1x1x28x28
        2) The first hidden layer is a convolution layer [p=2,s=1,f=5,a=tanh]
   which outputs 1x6x28x28 3) The second layer is a subsampling
   [p=0,s=2,f=2,a=avg_pool] which outputs 1x6x14x14 4) The third layer is a
   convolution [p=0,s=1,f=5,a=tanh] which outputs 1x16x10x10 5) The fourth layer
   is a subsampling [p=0,s=2,f=2,a=avg_pool] which outputs 1x16x5x5 6) The fifth
   layer is a convolution [p=0,s=1,f=5,a=tanh] which outputs 1x120x1x1 7) We
   will flatten the 1x120x1x1 feature maps into 1x120 8) The sixth layer is
   fully connected [a=tanh] which outputs 1x84x1x1 9) The seventh layer is a
   fully connected [a=softmax] which outputs 1x10x1x1 10) The output layer will
   have an argmax() to tell us which neuron from seventh layer had highest
   probability

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

#define MAX_LINE_SIZE 2000000
#define DECIMAL_PLACE_FACTOR 1000

#define COMPUTE_OUTPUT_SIZE(N_in, padding, filter_size, stride) \
    (int)floor(((N_in) + 2 * (padding) - (filter_size)) / (float)(stride) + 1)

inline int32_t relu(int32_t x) {
    return (x > 0) ? x : 0;
}

int32_t relu(int32_t x);

//-------------------------------------------------------------Lenet-5 Configuration-------------------------------------------//

// Layer 1 - convolution + tanh
#define INPUT_ROWS_1 32
#define INPUT_COLS_1 32
#define KERNEL_SIZE_1 5
#define STRIDE_1 1
#define PADDING_1 0
#define NUM_FILTERS_1 6
#define OUTPUT_ROWS_1 28
#define OUTPUT_COLS_1 28

// Layer 2 - Average Pooling
#define INPUT_ROWS_2 OUPUT_ROWS_1
#define INPUT_COLS_2 OUTPUT_COLS_1
#define KERNEL_SIZE_2 2
#define STRIDE_2 2
#define PADDING_2 0
#define NUM_FILTERS_2 6
#define OUTPUT_ROWS_2 14
#define OUTPUT_COLS_2 14

// Layer 3 - convolution + tanh
#define INPUT_ROWS_3 OUPUT_ROWS_2
#define INPUT_COLS_3 OUTPUT_COLS_2
#define KERNEL_SIZE_3 5
#define STRIDE_3 1
#define PADDING_3 0
#define NUM_FILTERS_3 16
#define OUTPUT_ROWS_3 10
#define OUTPUT_COLS_3 10

// Layer 4 - Average Pooling
#define INPUT_ROWS_4 OUPUT_ROWS_3
#define INPUT_COLS_4 OUTPUT_COLS_3
#define KERNEL_SIZE_4 2
#define STRIDE_4 2
#define PADDING_4 0
#define NUM_FILTERS_4 16
#define OUTPUT_ROWS_4 5
#define OUTPUT_COLS_4 5

// Layer 5 - convolution + tanh
#define INPUT_ROWS_5 OUPUT_ROWS_4
#define INPUT_COLS_5 OUTPUT_COLS_4
#define KERNEL_SIZE_5 5
#define STRIDE_5 1
#define PADDING_5 0
#define NUM_FILTERS_5 120
#define OUTPUT_ROWS_5 1
#define OUTPUT_COLS_5 1

// NOTE: We will Flatten the layers before passing to Fully Connected Layers

// Layer 6 - Fully Connected + tanh
#define INPUT_ROWS_6 1
#define INPUT_COLS_6 120
#define WEIGHT_ROWS_6 84
#define WEIGHT_COLS_6 120
#define NUM_FILTERS_6 84

// Layer 7 - Fully Connected + softmax
#define INPUT_ROWS_7 1
#define INPUT_COLS_7 84
#define WEIGHT_ROWS_7 10
#define WEIGHT_COLS_7 84
#define NUM_FILTERS_7 10

// Define the LeNet-5 architecture
typedef struct {
    // conv 1 (6x5x5)
    int32_t weights1[NUM_FILTERS_1][KERNEL_SIZE_1][KERNEL_SIZE_1];
    int32_t biases1[NUM_FILTERS_1];
    // conv 2 (16x5x5)
    int32_t weights2[NUM_FILTERS_3][NUM_FILTERS_1][KERNEL_SIZE_3][KERNEL_SIZE_3];
    int32_t biases2[NUM_FILTERS_3];
    // conv 3 (120x5x5)
    int32_t weights3[NUM_FILTERS_5][NUM_FILTERS_3][KERNEL_SIZE_5][KERNEL_SIZE_5];
    int32_t biases3[NUM_FILTERS_5];
    // fc 4 (84x120x1)
    int32_t weights4[WEIGHT_ROWS_6][WEIGHT_COLS_6];
    int32_t biases4[NUM_FILTERS_6];
    // fc 5 (10x84x1)
    int32_t weights5[WEIGHT_ROWS_7][WEIGHT_COLS_7];
    int32_t biases5[NUM_FILTERS_7];
} LeNet5;

void initLeNet5(LeNet5* model) {

    FILE* file = fopen("../extern/weight_bias_model_int.txt", "rb");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return;
    }
    char line[MAX_LINE_SIZE];

    // Read weights of layer 1
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int c = 0; c < NUM_FILTERS_1; c++) {
            for (int i = 0; i < KERNEL_SIZE_1; i++) {
                for (int j = 0; j < KERNEL_SIZE_1; j++) {
                    model->weights1[c][i][j] = atoi(token);
                    token = strtok(NULL, " ");
                }
            }
        }
    }

    // Read biases of layer 1
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_1; c++) {
            model->biases1[c] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 3
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");

        for (int c = 0; c < NUM_FILTERS_3; c++) {
            for (int k = 0; k < NUM_FILTERS_1; k++) {
                for (int i = 0; i < KERNEL_SIZE_3; i++) {
                    for (int j = 0; j < KERNEL_SIZE_3; j++) {
                        model->weights2[c][k][i][j] = atoi(token);
                        token = strtok(NULL, " ");
                    }
                }
            }
        }
    }

    // Read biases of layer 3
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_3; c++) {
            model->biases2[c] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 5
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");

        for (int c = 0; c < NUM_FILTERS_5; c++) {
            for (int k = 0; k < NUM_FILTERS_3; k++) {
                for (int i = 0; i < KERNEL_SIZE_5; i++) {
                    for (int j = 0; j < KERNEL_SIZE_5; j++) {
                        model->weights3[c][k][i][j] = atoi(token);
                        token = strtok(NULL, " ");
                    }
                }
            }
        }
    }

    // Read biases of layer 5
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_5; c++) {
            model->biases3[c] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 6
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int c = 0; c < WEIGHT_ROWS_6; c++) {
            for (int i = 0; i < WEIGHT_COLS_6; i++) {
                model->weights4[c][i] = atoi(token);
                token = strtok(NULL, " ");
            }
        }
    }

    // Read biases of layer 6
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < WEIGHT_ROWS_6; c++) {
            model->biases4[c] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 7
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int c = 0; c < WEIGHT_ROWS_7; c++) {
            for (int i = 0; i < WEIGHT_COLS_7; i++) {
                model->weights5[c][i] = atoi(token);
                token = strtok(NULL, " ");
            }
        }
    }

    // Read biases of layer 7
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < WEIGHT_ROWS_7; c++) {
            model->biases5[c] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    fclose(file);
}

void layer_7_fc(int32_t input_layer[NUM_FILTERS_6], LeNet5* model, int32_t output_layer[NUM_FILTERS_7]) {
    for (int n = 0; n < WEIGHT_ROWS_7; n++) {
        output_layer[n] = 0;

        for (int i = 0; i < WEIGHT_COLS_7; i++) {
            output_layer[n] += input_layer[i] * model->weights5[n][i];
        }
        output_layer[n] = output_layer[n] / DECIMAL_PLACE_FACTOR + model->biases5[n];
    }
}

void layer_6_fc(int32_t input_layer[NUM_FILTERS_5], LeNet5* model, int32_t output_layer[NUM_FILTERS_6]) {
    for (int n = 0; n < WEIGHT_ROWS_6; n++) {
        output_layer[n] = 0;

        for (int i = 0; i < WEIGHT_COLS_6; i++) {
            output_layer[n] += input_layer[i] * model->weights4[n][i];
        }

        output_layer[n] = relu(output_layer[n] / DECIMAL_PLACE_FACTOR + model->biases4[n]);
    }
}

void layer_5_conv_flat(int32_t input_layer[NUM_FILTERS_4][OUTPUT_ROWS_4][OUTPUT_COLS_4], LeNet5* model,
                       int32_t output_layer[NUM_FILTERS_5]) {
    for (int k = 0; k < NUM_FILTERS_5; k++) {
        output_layer[k] = 0;

        for (int c = 0; c < NUM_FILTERS_4; c++) {
            for (int m = 0; m < KERNEL_SIZE_5; m++) {
                for (int n = 0; n < KERNEL_SIZE_5; n++) {
                    output_layer[k] += input_layer[c][m][n] * model->weights3[k][c][m][n];
                }
            }
        }
        // output_layer[k] = sum/DECIMAL_PLACE_FACTOR + model->biases3[k];
        output_layer[k] = relu(output_layer[k] / DECIMAL_PLACE_FACTOR + model->biases3[k]);
    }
}

void layer_4_subsampling(int32_t input_layer[NUM_FILTERS_3][OUTPUT_ROWS_3][OUTPUT_COLS_3], LeNet5* model,
                         int32_t output_layer[NUM_FILTERS_4][OUTPUT_ROWS_4][OUTPUT_COLS_4]) {
    for (int filter_no = 0; filter_no < NUM_FILTERS_4; filter_no++) {
        for (int i = 0; i < OUTPUT_ROWS_4; i++) {
            for (int j = 0; j < OUTPUT_COLS_4; j++) {
                output_layer[filter_no][i][j] = 0;

                // Stride : 2 (So we divide Kernel by 2)
                for (int m = 0; m < KERNEL_SIZE_4; m++) {
                    for (int n = 0; n < KERNEL_SIZE_4; n++) {
                        output_layer[filter_no][i][j] += input_layer[filter_no][i * STRIDE_4 + m][j * STRIDE_4 + n];
                    }
                }

                // Calculate the average by dividing the sum by the pool size
                output_layer[filter_no][i][j] = output_layer[filter_no][i][j] / (KERNEL_SIZE_4 * KERNEL_SIZE_4);
            }
        }
    }
}

void layer_3_conv(int32_t input_layer[NUM_FILTERS_2][OUTPUT_ROWS_2][OUTPUT_COLS_2], LeNet5* model,
                  int32_t output_layer[NUM_FILTERS_3][OUTPUT_ROWS_3][OUTPUT_COLS_3]) {
    for (int k = 0; k < NUM_FILTERS_3; k++) {
        for (int i = 0; i < OUTPUT_ROWS_3; i++) {
            for (int j = 0; j < OUTPUT_COLS_3; j++) {
                output_layer[k][i][j] = 0;

                for (int c = 0; c < NUM_FILTERS_2; c++) {
                    for (int m = 0; m < KERNEL_SIZE_3; m++) {
                        for (int n = 0; n < KERNEL_SIZE_3; n++) {
                            output_layer[k][i][j]  += input_layer[c][i + m][j + n] * model->weights2[k][c][m][n];
                        }
                    }
                }

                output_layer[k][i][j] = relu(output_layer[k][i][j] / DECIMAL_PLACE_FACTOR + model->biases2[k]);
            }
        }
    }
}

void layer_2_subsampling(int32_t input_layer[NUM_FILTERS_1][OUTPUT_ROWS_1][OUTPUT_COLS_1], LeNet5* model,
                         int32_t output_layer[NUM_FILTERS_2][OUTPUT_ROWS_2][OUTPUT_COLS_2]) {
    for (int filter_no = 0; filter_no < NUM_FILTERS_2; filter_no++) {
        for (int i = 0; i < OUTPUT_ROWS_2; i++) {
            for (int j = 0; j < OUTPUT_COLS_2; j++) {
                output_layer[filter_no][i][j] = 0;

                for (int m = 0; m < KERNEL_SIZE_2; m++) {
                    for (int n = 0; n < KERNEL_SIZE_2; n++) {
                        output_layer[filter_no][i][j] += input_layer[filter_no][i * STRIDE_2 + m][j * STRIDE_2 + n];
                    }
                }

                output_layer[filter_no][i][j] = output_layer[filter_no][i][j] / (KERNEL_SIZE_2 * KERNEL_SIZE_2);
            }
        }
    }
}

void layer_1_conv(int32_t input_layer[INPUT_ROWS_1][INPUT_COLS_1], LeNet5* model,
                  int32_t output_layer[NUM_FILTERS_1][OUTPUT_ROWS_1][OUTPUT_COLS_1]) {

    for (int filter_no = 0; filter_no < NUM_FILTERS_1; filter_no++) {
        for (int input_row = 0; input_row < OUTPUT_ROWS_1; input_row++) {
            for (int input_col = 0; input_col < OUTPUT_COLS_1; input_col++) {
                output_layer[filter_no][input_row][input_col] = 0;

                // Dot product
                for (int kernel_row = 0; kernel_row < KERNEL_SIZE_1; kernel_row++) {
                    for (int kernel_col = 0; kernel_col < KERNEL_SIZE_1; kernel_col++) {
                        output_layer[filter_no][input_row][input_col] += (input_layer[input_row + kernel_row][input_col + kernel_col] *
                                                                          model->weights1[filter_no][kernel_row][kernel_col]);
                    }
                }
                output_layer[filter_no][input_row][input_col] =
                    relu(output_layer[filter_no][input_row][input_col] / DECIMAL_PLACE_FACTOR + model->biases1[filter_no]);
            }
        }
    }
}

#endif // CNN_H
