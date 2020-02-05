#include <iostream>
#include "../include/Neuron.hpp"

using namespace std;

int main(int argc, char **argv) {
    Neuron *n = new Neuron(0.9);
    cout<< "val: " << n->getVal() <<endl;
    cout<< "Activated Val: " << n->getActivatedVal() <<endl;
    cout<< "Derivedval: " << n->getDerivedVal() <<endl;

    return 0;
}