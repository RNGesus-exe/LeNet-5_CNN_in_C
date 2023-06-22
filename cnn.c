#include "cnn.h"

// Forward pass through the LeNet-5 model
void forwardPass(LeNet5 *model, float input[INPUT_ROWS_1][INPUT_COLS_1])
{
    float layer1[NUM_FILTERS_1][OUTPUT_ROWS_1][OUTPUT_COLS_1]; // convolution 1
    // float layer2[16][10][10];                                  // max_pooling

    layer_1_conv(input,model,layer1);

    // Max pooling layer 2
    // for (int i = 0; i < 16; i++)
    // {
    //     for (int j = 0; j < 10; j++)
    //     {
    //         for (int k = 0; k < 10; k++)
    //         {
    //             float inputPatch[2][2];
    //             for (int x = 0; x < 2; x++)
    //             {
    //                 for (int y = 0; y < 2; y++)
    //                 {
    //                     inputPatch[x][y] = layer1[i][j * 2 + x][k * 2 + y];
    //                 }
    //             }
    //             layer2[i][j][k] = maxPooling(inputPatch);
    //         }
    //     }
    // }
}

int main()
{
    LeNet5 model;
    initLeNet5(&model);

    printf("Weights and biases have been loaded\n");

    float input[INPUT_ROWS_1][INPUT_COLS_1] = {};
    loadDataset(input,"mnist_test.csv");

    forwardPass(&model, input);

    return 0;
}
