#ifndef _NEURALNETWORK_HPP_
#define _NEURALNETWORK_HPP_

#include <iostream>
#include "Matrix.hpp"
#include "Layer.hpp"
#include <algorithm>
#include "utils/MultiplyMatrix.hpp"
#include <vector>

using namespace std;

class NeuralNetwork {
    public:
        NeuralNetwork(vector<int> topology);
        void setCurrentInput(vector<double> input);
        void setCurrentTarget(vector<double> target);
        void printToConsole();
        void feedForward();
        void backPropagation();
        void setErrors();

        Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); }
        Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }
        Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); }
        Matrix *getWeightMatrix(int index) { return this->weightMatrices.at(index); }
        void setNeuronValue(int indexLayer, int indexNeuron, double val) { return this->layers.at(indexLayer)->setVal(indexNeuron, val); }

        double getTotalError() { return this->error; }
        vector<double> getErrors() { return this->errors; }

        void printInputToConsole();
        void printOutputToConsole();
        void printTargetToConsole();
    private:
        int             topologySize;
        vector<int>     topology;
        vector<Layer *>  layers;
        vector<Matrix *> weightMatrices;
        vector<Matrix *> gradientMatrices;
        vector<double>   input;
        vector<double>   target;
        double           error;
        vector<double>   errors;
        vector<double>   historicalErrors;
};

#endif
