#include "../include/Layer.hpp"

Layer::Layer(int size) {
    this->size = size;

    for(int i=0;i<size;i++){
        Neuron *n = new Neuron(0.0);
        this->neurons.push_back(n);
    }
}

void Layer::setVal(int i, double v) {
    this->neurons.at(i)->setVal(v);
}

Matrix *Layer::matrixifyVals() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for(int i=0;i<this->neurons.size();i++) {
        m->setValue(1, i, this->neurons.at(i)->getVal());
    }
}

Matrix *Layer::matrixifyActivatedVals() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for(int i=0;i<this->neurons.size();i++) {
        m->setValue(1, i, this->neurons.at(i)->getActivatedVal());
    }
}

Matrix *Layer::matrixifyDerivedVals() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for(int i=0;i<this->neurons.size();i++) {
        m->setValue(1, i, this->neurons.at(i)->getDerivedVal());
    }
}
