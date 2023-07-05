#include "../include/cnn.h"
#include <stdint.h>
#include <stdio.h>

// Forward pass through the LeNet-5 model
int8_t forwardPass(LeNet5* model, int32_t input[INPUT_ROWS_1][INPUT_COLS_1]) {
    int32_t layer1[NUM_FILTERS_1][OUTPUT_ROWS_1][OUTPUT_COLS_1]; // convolution (6*28*28)
    int32_t layer2[NUM_FILTERS_2][OUTPUT_ROWS_2][OUTPUT_COLS_2]; // max_pooling (6*14*14)
    int32_t layer3[NUM_FILTERS_3][OUTPUT_ROWS_3][OUTPUT_COLS_3]; // convolution (16*10*10)
    int32_t layer4[NUM_FILTERS_4][OUTPUT_ROWS_4][OUTPUT_COLS_4]; // max_pooling (16*5*5)
    int32_t layer5[NUM_FILTERS_5];                               // convolution + flatten (1*120)
    int32_t layer6[NUM_FILTERS_6];                               // fully connected (1*84)
    int32_t layer7[NUM_FILTERS_7];                               // fully connected (1*10)
    // int32_t output[NUM_FILTERS_7];

    layer_1_conv(input, model, layer1);

    layer_2_subsampling(layer1, model, layer2);

    layer_3_conv(layer2, model, layer3);

    layer_4_subsampling(layer3, model, layer4);

    layer_5_conv_flat(layer4, model, layer5);

    layer_6_fc(layer5, model, layer6);

    layer_7_fc(layer6, model, layer7);

    int8_t max_ind = 0;
    // printf("%d,", output[0]);
    for (int i = 1; i < NUM_FILTERS_7; i++) {
        // printf("%d,", output[i]);
        if (layer7[max_ind] < layer7[i]) {
            max_ind = i;
        }
    }
    // printf("\nThe number is = %d\n", max_ind);

    // FILE* file = fopen("layer_5_weights.txt", "wb");
    // for (int k = 0; k < NUM_FILTERS_5; k++) {
    //     for (int l = 0; l < NUM_FILTERS_4; l++) {
    //         for (int i = 0; i < KERNEL_SIZE_5; i++) {
    //             for (int m = 0; m < KERNEL_SIZE_5; m++) {
    //                 fprintf(file, "%d,", model->weights3[k][l][i][m]);
    //            }
    //             fprintf(file, "\n");
    //         }
    //         fprintf(file,"\n\n");
    //     }
    //     fprintf(file,"\n\n\n");
    // }

    // file = fopen("layer_4_output.txt", "wb");
    // for (int k = 0; k < NUM_FILTERS_4; k++) {
    //     for (int i = 0; i < OUTPUT_ROWS_4; i++) {
    //         for (int j = 0; j < OUTPUT_COLS_4; j++) {
    //             fprintf(file, "%d,", layer4[k][i][j]);
    //         }
    //    }
    // }

    // file = fopen("layer_7_output.txt", "wb");
    // for (int i = 0; i < NUM_FILTERS_7; i++) {
    //     fprintf(file, "%d,", layer7[i]);
    // }

    // fclose(file);
    //
    return max_ind;
}

int main() {
    LeNet5 model;
    initLeNet5(&model);
    printf("SUCCESS: Weights and biases have been loaded\n");

    int32_t input[INPUT_ROWS_1][INPUT_COLS_1] = {}; // Padding : 2

    FILE* file = fopen("../extern/mnist_test.csv", "rb");
    if (file == NULL) {
        printf("Failed to open the file\n");
        return -1;
    }

    char line[MAX_LINE_SIZE];
    int32_t dataset_size = 0;
    int32_t true_positive = 0;
    int32_t false_negative = 0;
    while (fgets(line, sizeof(line), file)) {

        char* input_data;
        char* truth = strtok(line, ",");
        input_data = strtok(NULL, ",");

        for (int i = 2; i < INPUT_ROWS_1 - 2; i++) {
            for (int j = 2; j < INPUT_COLS_1 - 2; j++) {
                input[i][j] = (int32_t)(atof(input_data) / 255 * DECIMAL_PLACE_FACTOR);
                input_data = strtok(NULL, ",");
            }
        }
        int pred = forwardPass(&model, input);
        if (pred == *truth - '0') {
            true_positive++;
        } else {
            false_negative++;
        }
        dataset_size++;
    }
    fclose(file);

    printf("Accuracy = %.2f \n", (float)true_positive / dataset_size);
    return 0;
}
