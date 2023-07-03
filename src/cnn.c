#include "../include/cnn.h"

void log_softmax(float input[NUM_FILTERS_7], int input_size,
                 float output[NUM_FILTERS_7]) {
  // Find the maximum value in the input array
  float max_value = input[0];
  for (int i = 1; i < input_size; i++) {
    if (input[i] > max_value) {
      max_value = input[i];
    }
  }

  // Compute the softmax values
  float sum_exp = 0.0;
  for (int i = 0; i < input_size; i++) {
    output[i] = exp(input[i] - max_value);
    sum_exp += output[i];
  }

  // Compute the log softmax values
  for (int i = 0; i < input_size; i++) {
    output[i] = log(output[i] / sum_exp);
  }
}

// Forward pass through the LeNet-5 model
void forwardPass(LeNet5 *model, float input[INPUT_ROWS_1][INPUT_COLS_1]) {
  float layer1[NUM_FILTERS_1][OUTPUT_ROWS_1]
              [OUTPUT_COLS_1]; // convolution (6*28*28)
  float layer2[NUM_FILTERS_2][OUTPUT_ROWS_2]
              [OUTPUT_COLS_2]; // max_pooling (6*14*14)
  float layer3[NUM_FILTERS_3][OUTPUT_ROWS_3]
              [OUTPUT_COLS_3]; // convolution (16*10*10)
  float layer4[NUM_FILTERS_4][OUTPUT_ROWS_4]
              [OUTPUT_COLS_4]; // max_pooling (16*5*5)
  float layer5[NUM_FILTERS_5]; // convolution + flatten (1*120)
  float layer6[NUM_FILTERS_6]; // fully connected (1*84)
  float layer7[NUM_FILTERS_7]; // fully connected (1*10)
  float output[NUM_FILTERS_7];

  layer_1_conv(input, model, layer1);

  layer_2_subsampling(layer1, model, layer2);

  layer_3_conv(layer2, model, layer3);

  layer_4_subsampling(layer3, model, layer4);

  layer_5_conv_flat(layer4, model, layer5);

  layer_6_fc(layer5, model, layer6);

  layer_7_fc(layer6, model, layer7);

  log_softmax(layer7, NUM_FILTERS_7, output);

  int max_ind = 0;
  printf("%f,", output[0]);
  for (int i = 1; i < NUM_FILTERS_7; i++) {
    printf("%f,", output[i]);
    if (output[max_ind] < output[i]) {
      max_ind = i;
    }
  }
  printf("\nThe number is = %d\n", max_ind);

  FILE *file = fopen("fc_2_weights.txt", "wb");
  for (int i = 0; i < WEIGHT_ROWS_7; i++) {
    for (int m = 0; m < WEIGHT_COLS_7; m++) {
      fprintf(file, "%f,", model->weights5[i][m]);
    }
    fprintf(file, "\n");
  }
  file = fopen("fc_2.txt", "wb");
  for (int i = 0; i < NUM_FILTERS_7; i++) {
    fprintf(file, "%f,", layer7[i]);
  }
}

int main() {
  LeNet5 model;
  initLeNet5(&model);
  printf("SUCCESS: Weights and biases have been loaded\n");

  float input[INPUT_ROWS_1][INPUT_COLS_1] = {}; // Padding : 2
  loadDataset(input, "../extern/mnist_test.csv");
  printf("SUCCESS: Dataset has been loaded into memory\n");

  forwardPass(&model, input);
  printf("SUCCESS: Forward pass complete\n");

  return 0;
}
