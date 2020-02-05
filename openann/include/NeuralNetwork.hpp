#ifndef _NEURALNETWORK_
#define _NEURALNETWORK_

#include <iostream>
#include "Matrix.hpp"
#include "Layer.hpp"
#include <vector>

using namespace std;

class NeuralNetwork {
    public:
        NeuralNetwork(vector<int> topology);
        void setCurrentInput(vector<double> input);

    private:
        int             topologySize;
        vector<int>     topology;
        vector<Layer *>  layers;
        vector<Matrix *> weightMatrices;
        vector<double>   input;
};

#endif
