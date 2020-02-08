#include <iostream>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/MultiplyMatrix.hpp"

using namespace std;

int main(int argc, char **argv) {
    /*Neuron *n = new Neuron(0.9);
    cout<< "val: " << n->getVal() <<endl;
    cout<< "Activated Val: " << n->getActivatedVal() <<endl;
    cout<< "Derivedval: " << n->getDerivedVal() <<endl;*/

    /*Matrix *m = new Matrix(3,2,true);
    m->printToConsole();

    cout<<" ^^^^^^^^^^^^^^^^^^^^^^^\n";

    Matrix *t = m->transpose();
    t->printToConsole();
    return 0;*/

    vector<int> topology{3,2,3};

    vector<double> input{1.0,0.0,1.0};
    NeuralNetwork *nn = new NeuralNetwork(topology);
    nn->setCurrentInput(input);
    nn->setCurrentTarget(input);
    
     //train
    for(int i=0;i<20;i++) {
        cout<<"EPOCH: "<<i<<endl;
        nn->feedForward();
        nn->setErrors();
        cout<< "Total Error "<< nn->getTotalError()<<endl;
        nn->backPropagation();

        cout<< "=============\n";
        cout<<"OUTPUT\n";
        nn->printOutputToConsole();

        cout<<"TARGET\n";
        nn->printTargetToConsole();
        cout<< "=============\n\n";


    }
    return 0;

}