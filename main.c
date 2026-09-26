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
}Neuron_struct;


static Neuron_struct Neuron(const int fan_in) {
    struct Neuron_struct n;
    n.weight = malloc(sizeof(Value*) * fan_in);

    for (int i = 0; i < fan_in; i++) {
        n.weight[i] = value(random(-1, 1), '_');
    }

    n.bias = value(random(-1, 1), ' ');
    return n;
}

static Value* neuron(Value* input[], Neuron_struct n, const int fan_in) {

    Value* output = value(0.0, '_');

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






int main() {
    srand((unsigned int)time(0));
    rand();

    Value* x[2] = {value(2,' '), value(1,'_')};

    const int fan_in = 2;
    const Neuron_struct n1 = Neuron(fan_in);

    printf("%.3f %.3f %.3f \n", x[0]->grad, x[1]->grad, n1.weight[0]->grad);

    Value* out = neuron(x, n1, fan_in);

    out->grad = 1;
    backward(out);


    printf("%.3f %.3f %.3f %.3f \n", x[0]->grad, x[1]->grad, n1.weight[0]->grad, n1.weight[1]->grad);

    return 0;
}











