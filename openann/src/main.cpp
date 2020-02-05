#include <iostream>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"

using namespace std;

int main(int argc, char **argv) {
    /*Neuron *n = new Neuron(0.9);
    cout<< "val: " << n->getVal() <<endl;
    cout<< "Activated Val: " << n->getActivatedVal() <<endl;
    cout<< "Derivedval: " << n->getDerivedVal() <<endl;*/

    Matrix *m = new Matrix(3,2,true);
    m->printToConsole();

    cout<<" ^^^^^^^^^^^^^^^^^^^^^^^\n";

    Matrix *t = m->transpose();
    t->printToConsole();
    return 0;
}