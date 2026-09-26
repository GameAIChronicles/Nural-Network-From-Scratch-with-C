#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>



static float random(float min, float max) {
    float x = (float) rand() / (float) (RAND_MAX);
    x = (x  * (max - min)) + min;
    return x;
}


typedef struct Value {
    double data;
    double grad;
    char operator;
    struct Value *prev_v[2];
}Value;

static Value* value(const double data, const char operator) {


    Value *v = malloc(sizeof(Value));
    v->data = data;
    v->operator = operator;
    v->grad = 0;
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

static void backward(Value *a) {
    if (a->operator == '+') {
        a->prev_v[0]->grad += a->grad;
        a->prev_v[1]->grad += a->grad;
        backward(a->prev_v[0]);
        backward(a->prev_v[1]);
    }
    else if (a->operator == '*') {
        a->prev_v[0]->grad += a->grad*a->prev_v[1]->data;
        a->prev_v[1]->grad += a->grad*a->prev_v[0]->data;
        backward(a->prev_v[0]);
        backward(a->prev_v[1]);
    }
    else if (a->operator == 't') {
        a->prev_v[0]->grad += a->grad * (1-pow(a->data, 2));
        backward(a->prev_v[0]);
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

    Value* output = value(0.0, '_');
    const int fan_in = n.fan_in;
    for (int i = 0; i < fan_in; i++) {
        Value* wx = mul(input[i], n.weight[i]);
        output = add(output, wx);
    }

    for (int i = 0; i < fan_in; i++) {

    }

    output = add(output, n.bias);
    output = Tanh(output);
    return output;
}


typedef struct Layers_struct {
    Neuron_struct* layers;
    int fan_out;
}Layers_struct;

static Layers_struct Layers(const int fan_in, const int fan_out) {
    struct Layers_struct layers;
    layers.fan_out = fan_out;
    layers.layers = malloc(sizeof(Neuron_struct)*fan_out);

    for (int i = 0; i < fan_out; i++) {
        layers.layers[i] = Neuron(fan_in);
    }

    return layers;
}

static Value** layers(Value* input[],const Layers_struct Layers) {
    const int fan_out = Layers.fan_out;
    Value** output = malloc(sizeof(Value*) * fan_out);
    for (int i = 0; i < fan_out; i++) {
        output[i] = neuron(input, Layers.layers[i]);
    }
    return output;
}


int main() {

    // 1. Root node z
    Value* z = value(1.0, ' ');
    Value* zero = value(0.0, ' ');

    // 2. x = z + 0  (x = 1.0)
    Value* x = add(z, zero);

    // 3. Two different paths branch out from x
    Value* three = value(3.0, ' ');
    Value* four = value(4.0, ' ');

    Value* path_A = mul(x, three); // A = x * 3 = 3.0
    Value* path_B = mul(x, four);  // B = x * 4 = 4.0

    // 4. Merge them into the final output
    Value* O = add(path_A, path_B); // O = A + B = 7.0

    // 5. Run backpropagation
    O->grad = 1.0;
    backward(O);

    // 6. Print the results to see the bug!
    printf("--- THE PROOF RESULTS ---\n");
    printf("x->grad (Eventually adds up): %.3f  [Expected: 7.000]\n", x->grad);
    printf("z->grad (RUINED BY RECURSION): %.3f [Expected: 7.000]\n", z->grad);

    return 0;

    /*
    srand((unsigned int)time(0));
    rand();

    Value* x[2] = {value(2,' '), value(1,'_')};

    const int fan_in = 2;
    const int fan_out = 2;
    const Layers_struct l1 = Layers(fan_in, fan_out);
    const Layers_struct l2 = Layers(fan_in, 1);

    printf("%.3f %.3f \n", x[0]->grad, x[1]->grad);

    Value** l1_out = layers(x, l1);
    Value** out = layers(l1_out, l2);

    out[0]->grad = 1;
    backward(out[0]);


    printf("%.3f %.3f \n", x[0]->grad, x[1]->grad);

    return 0;
    */
}











