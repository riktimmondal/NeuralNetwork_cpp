#ifndef _NEURALNETWORK_HPP_
#define _NEURALNETWORK_HPP_

#include <iostream>
#include "Matrix.hpp"
#include "Layer.hpp"
#include "utils/MultiplyMatrix.hpp"
#include <vector>

using namespace std;

class NeuralNetwork {
    public:
        NeuralNetwork(vector<int> topology);
        void setCurrentInput(vector<double> input);
        void printToConsole();
        void feedForward();

        Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); }
        Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }
        Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); }
        Matrix *getWeightMatrix(int index) { return this->weightMatrices.at(index); }
        void setNeuronValue(int indexLayer, int indexNeuron, double val) { return this->layers.at(indexLayer)->setVal(indexNeuron, val); }

    private:
        int             topologySize;
        vector<int>     topology;
        vector<Layer *>  layers;
        vector<Matrix *> weightMatrices;
        vector<double>   input;
};

#endif
