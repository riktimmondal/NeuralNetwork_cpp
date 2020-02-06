#ifndef _LAYER_
#define _LAYER_

#include <iostream>
#include <vector>
#include "Neuron.hpp"
#include "Matrix.hpp"

using namespace std;

class Layer
{
    public:
        Layer(int size);
        void setVal(int i, double v);
        Matrix *matrixifyVals();
        Matrix *matrixifyActivatedVals();
        Matrix *matrixifyDerivedVals();

        vector<Neuron *> getNeurons() { return this->neurons; };
        void setNeuron(vector<Neuron *> neurons) { this->neurons = neurons; }

    private:
        int size;

        vector<Neuron *> neurons;
};

#endif
