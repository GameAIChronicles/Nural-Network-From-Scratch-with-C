#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


#define SIZE(a) (sizeof(a) / sizeof(a[0]))


static float random(const float min, const float max) {
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

static Value* add(Value *a, Value *b) {
    Value *v = value(a->data + b->data, '+');

    v->prev_v[0] = a;
    v->prev_v[1] = b;

    return v;
}

static Value* mul(Value *a, Value *b) {
    Value *v = value(a->data * b->data, '*');

    v->prev_v[0] = a;
    v->prev_v[1] = b;
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

static void backward_helper(const Value *a) {
    if (a->operator == '+') {
        a->prev_v[0]->grad += a->grad;
        a->prev_v[1]->grad += a->grad;
    }
    else if (a->operator == '*') {
        a->prev_v[0]->grad += a->grad*a->prev_v[1]->data;
        a->prev_v[1]->grad += a->grad*a->prev_v[0]->data;
    }
    else if (a->operator == 't') {
        a->prev_v[0]->grad += a->grad * (1-pow(a->data, 2));
    }
}

static void backward(Value* a) {
    int output_size = 0;
    Value** list = build_topo(a, &output_size);

    for (int i = output_size -1; i >= 0; i--) {
        backward_helper(list[i]);
    }
}

typedef struct Neuron_struct{
    Value **weight;
    Value *bias;
    int fan_in;
}Neuron_struct;


static Neuron_struct Neuron(const int fan_in) {
    struct Neuron_struct n;
    n.weight = malloc(sizeof(Value*) * fan_in);
    n.fan_in = fan_in;

    for (int i = 0; i < fan_in; i++) {
        n.weight[i] = value(random(-1, 1), '_');
    }

    n.bias = value(random(-1, 1), ' ');
    return n;
}

static Value* neuron(Value* input[], const Neuron_struct n) {

    Value *output = mul(input[0], n.weight[0]);
    const int fan_in = n.fan_in;
    for (int i = 1; i < fan_in; i++) {
        Value* wx = mul(input[i], n.weight[i]);
        output = add(output, wx);
    }
    

    output = add(output, n.bias);
    output = Tanh(output);
    return output;
}


typedef struct Layers_struct {
    Neuron_struct* neurons;
    int fan_out;
}Layers_struct;

static Layers_struct Layers(const int fan_in, const int fan_out) {
    struct Layers_struct layers;
    layers.fan_out = fan_out;
    layers.neurons = malloc(sizeof(Neuron_struct)*fan_out);

    for (int i = 0; i < fan_out; i++) {
        layers.neurons[i] = Neuron(fan_in);
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



static MLP_struct MLP(const int* dimension,const int no_layers) {
    MLP_struct mlp;
    mlp.no_layers = no_layers-1;
    mlp.dimension = malloc(sizeof(int) * no_layers);
    mlp.layers = malloc(sizeof(Layers_struct));

    for (int i = 0; i<no_layers-1; i++) {
        mlp.dimension[i] = dimension[i];
        mlp.layers = realloc(mlp.layers, sizeof(Layers_struct)*(i+1));
        mlp.layers[i] = Layers(dimension[i], dimension[i+1]);
    }
    mlp.dimension[no_layers-1] = dimension[no_layers-1];

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



int main() {
    srand((unsigned int)time(0));
    rand();

    Value* x[2] = {value(2,' '), value(1,'_')};
    const int structure[] = {2, 1, 1, 1};
    const int no_layers = SIZE(structure);
    const MLP_struct nn = MLP(structure, no_layers);
    Value** out = mlp(x, nn);
    out[0]->grad = 1;

    printf("data of out:%f\n", out[0]->data);

    backward(out[0]);


    printf("%.3f %.3f \n", x[0]->grad, x[1]->grad);

    return 0;
}