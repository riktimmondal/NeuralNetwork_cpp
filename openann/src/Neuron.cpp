#include "../include/Neuron.hpp"

//constructor
Neuron::Neuron(double val) {
    this->val = val;
    activate();
    derive();
}

void Neuron::setVal(double val) {
    this->val = val;
    activate();
    derive();
}
void Neuron::activate() {
    this->activatedVal = this->val / (1 + abs(this->val));
}

void Neuron::derive() {
    this->derivedVal =  this->activatedVal * (1 - this->activatedVal);
}