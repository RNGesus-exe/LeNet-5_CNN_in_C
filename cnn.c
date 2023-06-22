#include "cnn.h"

// Forward pass through the LeNet-5 model
void forwardPass(LeNet5 *model, double input[INPUT_ROWS_1][INPUT_COLS_1])
{
    double layer1[NUM_FILTERS_1][OUTPUT_ROWS_1][OUTPUT_COLS_1]; // convolution 1
    // float layer2[16][10][10];                                  // max_pooling

    // Convolution layer 1
    layer_1_conv(input,model,layer1);

    FILE*file =  fopen("feature.txt","w");
    for(int c =0;c<NUM_FILTERS_1;c++){
        for(int i =0;i<OUTPUT_ROWS_1;i++){
            for(int j =0;j<OUTPUT_COLS_1;j++){
                fprintf(file,"%f ",layer1[c][i][j]);        
            }   
        }   
    }

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

    double input[INPUT_ROWS_1][INPUT_COLS_1] = {};
    loadDataset(input,"mnist_test.csv");

    FILE *file = fopen("image.txt","wb");
    for(int i =2;i<INPUT_ROWS_1-2;i++){
     for(int j =2;j<INPUT_COLS_1-2;j++){
        // if(input[i][j] > 0.0f)
        fprintf(file,"%f,",input[i][j]);        
    }   
    }   

    forwardPass(&model, input);

    return 0;
}
