#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


#define SIZE(a) (sizeof(a) / sizeof(a[0]))
#define MAX_LINE_LEN 1024
#define COLS 4

static float random_float(const float min, const float max) {
    float x = (float) rand() / (float) (RAND_MAX);
    x = (x  * (max - min)) + min;
    return x;
}


typedef struct Value {
    double data;
    double grad;
    int visited;
    char operator;
    struct Value *prev_v[2];
}Value;

static Value* value(const double data, const char operator) {


    Value *v = malloc(sizeof(Value));
    v->data = data;
    v->operator = operator;
    v->grad = 0;
    v->visited = 0;
    v->prev_v[0] = NULL;
    v->prev_v[1] = NULL;
    return v;
}

static Value* Add(Value *a, Value *b) {
    Value *v = value(a->data + b->data, '+');

    v->prev_v[0] = a;
    v->prev_v[1] = b;

    return v;
}

static Value* Mul(Value *a, Value *b) {
    Value *v = value(a->data * b->data, '*');

    v->prev_v[0] = a;
    v->prev_v[1] = b;
    return v;
}

static Value* Div(Value *a, Value *b) {
    Value *v = value(a->data / b->data, '/');
    v->prev_v[0] = a;
    v->prev_v[1] = b;
    return v;
}

static Value* Neg(Value *a) {
    Value *v = value(a->data * -1, '-');
    v->prev_v[0] = a;
    return v;
}

static Value* Log(Value *a) {
    const double input_data = a->data <= 0.0 ? 1e-7 : a->data;
    Value *v = value(log(input_data),  'l');
    v->prev_v[0] = a;
    return v;
}

static Value* Exp(Value *a) {
    Value *v = value(exp(a->data),  'e');
    v->prev_v[0] = a;
    return v;
}


static Value* Tanh(Value *a) {
    Value *v = value(tanh(a->data), 't');
    v->prev_v[0] = a;
    return v;
}


static void topo_helper(Value* v, Value*** list,int* size) {
    if (v == NULL) return;
    if (v->visited == 1) return;

    v->visited = 1;

    topo_helper(v->prev_v[0], list, size);
    topo_helper(v->prev_v[1], list, size);

    (*size)++;

    *list = realloc(*list, sizeof(Value*) * *size);

    (*list)[*size - 1] = v;
}

static Value** build_topo(Value* v, int *out_size) {
    int size = 0;
    Value** topo = NULL;

    topo_helper(v, &topo, &size);
    *out_size = size;

    for (int i = 0; i < size; i++) {
        topo[i]->visited = 0;
    }

    return topo;
}

static void backward_helper(Value *a) {
    if (a->operator == '+') {
        a->prev_v[0]->grad += a->grad;
        a->prev_v[1]->grad += a->grad;
    }
    else if (a->operator == '-') {
        a->prev_v[0]->grad -= a->grad;
    }
    else if (a->operator == '*') {
        a->prev_v[0]->grad += a->grad*a->prev_v[1]->data;
        a->prev_v[1]->grad += a->grad*a->prev_v[0]->data;
    }
    else if (a->operator == 't') {
        a->prev_v[0]->grad += a->grad * (1-pow(a->data, 2));
    }
    else if (a->operator == 'e') {
        a->prev_v[0]->grad += a->grad * a->data;
    }
    else if (a->operator == 'l') {
        a->prev_v[0]->grad += a->grad * (1.0 / a->prev_v[0]->data);
    }
    else if (a->operator == '/') {
        const double u = a->prev_v[0]->data; // numerator
        const double v = a->prev_v[1]->data; // denominator

        // Gradient flowing back to the numerator: dL/du = dL/dy * (1 / v)
        a->prev_v[0]->grad += a->grad / v;

        // Gradient flowing back to the denominator: dL/dv = dL/dy * (-u / v^2)
        a->prev_v[1]->grad += a->grad * (-u / (v * v));
    }
}

static Value** backward(Value* a, int *out_size) {
    Value** list = build_topo(a, out_size);

    for (int i = *out_size - 1; i >= 0; i--) {
        backward_helper(list[i]);
    }
    return list; // Return the list straight to main
}

typedef struct Neuron_struct{
    Value **weight;
    Value *bias;
    int fan_in;
    int is_out;
}Neuron_struct;


static Neuron_struct Neuron(const int fan_in,const int is_out) {
    struct Neuron_struct n;
    n.weight = malloc(sizeof(Value*) * fan_in);
    n.fan_in = fan_in;
    n.is_out = is_out;

    for (int i = 0; i < fan_in; i++) {
        n.weight[i] = value(random_float(-1, 1), '_');
    }

    n.bias = value(random_float(-1, 1), ' ');
    return n;
}

static Value* neuron(Value* input[], const Neuron_struct n) {

    Value *output = Mul(input[0], n.weight[0]);
    const int fan_in = n.fan_in;
    for (int i = 1; i < fan_in; i++) {
        Value* wx = Mul(input[i], n.weight[i]);
        output = Add(output, wx);
    }


    output = Add(output, n.bias);

    if (n.is_out==0) {
        output = Tanh(output);
    }
    return output;
}


typedef struct Layers_struct {
    Neuron_struct* neurons;
    int fan_out;
}Layers_struct;

static Layers_struct Layers(const int fan_in, const int fan_out, const int is_out) {
    struct Layers_struct layers;
    layers.fan_out = fan_out;
    layers.neurons = malloc(sizeof(Neuron_struct)*fan_out);

    for (int i = 0; i < fan_out; i++) {
        layers.neurons[i] = Neuron(fan_in, is_out);
    }

    return layers;
}

static Value** layers(Value* input[],const Layers_struct Layers) {
    const int fan_out = Layers.fan_out;
    Value** output = malloc(sizeof(Value*) * fan_out);
    for (int i = 0; i < fan_out; i++) {
        output[i] = neuron(input, Layers.neurons[i]);
    }
    return output;
}


typedef struct MLP_struct {
    int* dimension;
    Layers_struct* layers;
    int no_layers;
}MLP_struct;



static MLP_struct MLP(const int* dimension, const int no_layers) {
    MLP_struct mlp;
    mlp.no_layers = no_layers - 1;
    mlp.dimension = malloc(sizeof(int) * no_layers);

    mlp.layers = malloc(sizeof(Layers_struct) * mlp.no_layers);

    for (int i = 0; i < mlp.no_layers; i++) {
        mlp.dimension[i] = dimension[i];
        printf("The value of is_out: %d\n", i + 1 == no_layers - 1 ? 1 : 0);
        mlp.layers[i] = Layers(dimension[i], dimension[i + 1], i + 1 == no_layers - 1 ? 1 : 0);
    }
    mlp.dimension[no_layers - 1] = dimension[no_layers - 1];

    return mlp;
}


static Value** mlp(Value* input[], const MLP_struct mlp) {
    Value** out = layers(input, mlp.layers[0]);
    for (int i = 1; i < mlp.no_layers; i++) {
        Value** temp = layers(out, mlp.layers[i]);
        free(out); // Frees the old array of pointers, not the Value nodes themselves!
        out = temp;

    }

    return out;
}


static int get_parameter(Value** input[],const MLP_struct mlp) {
    int no_param = 0;
    Value** param = NULL;
    for (int i = 0; i<mlp.no_layers; i++) {
        const Layers_struct layer = mlp.layers[i];
        for (int j = 0; j<layer.fan_out; j++) {
            const Neuron_struct neuron = layer.neurons[j];
            for (int k = 0; k<neuron.fan_in; k++) {
                no_param++;
                param = realloc(param, sizeof(Value*) * no_param);
                param[no_param-1] = neuron.weight[k];

            }
            no_param++;
            param = realloc(param, sizeof(Value*) * no_param);
            param[no_param-1] = neuron.bias;

        }
    }
    *input = param;
    return no_param;
}


static Value** Softmax(Value** input, const int out_size) {

    Value** exp_input = malloc(sizeof(Value*) * out_size);
    for (int i = 0; i < out_size; i++) {
        exp_input[i] = Exp(input[i]);
    }

    Value* sum = exp_input[0];
    for (int i = 1; i < out_size; i++) {
        sum = Add(sum, exp_input[i]);
    }

    Value** temp = malloc(sizeof(Value*) * out_size);
    for (int i = 0; i < out_size; i++) {
        temp[i] = Div(exp_input[i], sum);
    }
    free(exp_input);
    return temp;
}

static Value* neg_log_likelihood(Value* prob) {
    Value* out = Neg(Log(prob));
    return out;
}


static Value*** load_dataset(FILE* files, int *out_row_count, Value** output) {

    // User ID,Heart Rate (BPM),Blood Oxygen Level (%),Step Count,Sleep Duration (hours),Activity Level


    int max_rows = 4; // Start small, handles any size automatically
    int row_count = 0;
    Value ***data_array = malloc(max_rows * sizeof(Value **));

    // Open the text file
    FILE *file = files;
    if (file == NULL) {
        printf("Error: Could not open data.txt\n");
        return data_array;
    }

    char line[MAX_LINE_LEN];

    // Read the text file line-by-line
    while (fgets(line, sizeof(line), file) != NULL) {

        // Expand the array of arrays dynamically if we run out of space
        if (row_count >= max_rows) {
            max_rows *= 2;
            data_array = realloc(data_array, max_rows * sizeof(Value **));
        }

        // Allocate a new array (row) for the 4 double values
        data_array[row_count] = malloc(COLS * sizeof(Value*));

        //Parse the comma-separated strings into numbers
        int col_count = 0;
        char *token = strtok(line, ",");

        while (token != NULL && col_count < COLS) {
            // Convert to double to keep full decimal precision
            data_array[row_count][col_count] = value(strtod(token, NULL), '_');
            token = strtok(NULL, ",");
            col_count++;
        }
        row_count++;
    }
    *out_row_count = row_count;
    // Always close the file handle
    fclose(file);

    return data_array;

}

static void free_dataset(Value*** data_array, const int row_count) {
    if (data_array == NULL) return;

    for (int i = 0; i < row_count; i++) {
        if (data_array[i] != NULL) {
            // 1. Free each individual Value node inside the row
            for (int j = 0; j < COLS; j++) {
                if (data_array[i][j] != NULL) {
                    free(data_array[i][j]);
                }
            }
            // 2. Free the array of pointers representing this row
            free(data_array[i]);
        }
    }
    // 3. Free the root array of pointers
    free(data_array);
}



int main() {
    srand((unsigned int)time(0));
    rand();

    FILE* file = fopen("D:\\my_project\\Nural-Network-From-Scratch-with-C\\cleaned_dataset.txt", "r");
    if (file == NULL) {
        printf("Error: Could not open cleaned_dataset.txt\n");
        return 1;
    }
    int row_count = 0;

    Value*** data = load_dataset(file, &row_count);
    row_count -= 2;
    printf("The no of data in dataset: %d\n", row_count);



    const int structure[] = {6, 16, 16, 3};
    const int no_layers = SIZE(structure);
    int output_size = structure[no_layers - 1];
    const MLP_struct nn = MLP(structure, no_layers);

    Value** param = NULL;
    const int no_param = get_parameter(&param, nn);
    printf("No of parameters: %d\n", no_param);
    const double learning_rate = 0.01;


    const int epochs = 2;

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("[");
        for (int j = 0; j < row_count; j++) {
            // Forward Pass

            Value** out = mlp(data[j], nn);
            const int target = data[j][5]->data; // Index of What we want the network output data to become

            Value** prob = Softmax(out, output_size);
            // LOSS
            Value* loss = neg_log_likelihood(prob[target]);

            // Backward Pass
            loss->grad = 1.0;
            int graph_size = 0;
            Value** topo_list = backward(loss, &graph_size);



            for (int i = 0; i < no_param; i++) {
                param[i]->data -= learning_rate * param[i]->grad;
            }

            if (j % 200 == 0) {
                printf("=");
            }

            if (row_count -1 == j) {
                printf("] \nEpoch %02d | Output: %f | Loss: %.4f\n", epoch + 1, out[0]->data, loss->data);
                Value** out1 = mlp(data[row_count], nn);
                const int target1 = data[row_count][5]->data; // Index of What we want the network output data to become

                Value** prob1 = Softmax(out1, output_size);
                // LOSS
                Value* loss1 = neg_log_likelihood(prob1[target1]);
                printf("Test %02d | Output: %f | Loss: %.4f\n", 1, out1[0]->data, loss1->data);
            }

            // Clear the memory to free the space
            for (int i = 0; i < graph_size; i++) {
                Value* node = topo_list[i];
                if (node->operator == '+' || node->operator == '*' || node->operator == 't' ||
                    node->operator == '-' || node->operator == 'e' || node->operator == 'l' ||
                    node->operator == '/') {
                    free(node);
                    }
                else {
                    node->visited = 0;
                    node->grad = 0.0;
                }
            }
            for (int i = 0; i < output_size; i++) {
                if (i != target) {
                    free(prob[i]); // Frees the isolated '/' node for the non-target class
                }
            }

            free(topo_list);
            free(out);
            free(prob);
        }

    }
    for(int i = 0; i < no_param; i++) {
        free(param[i]);
    }
    free(param);
    return 0;

}