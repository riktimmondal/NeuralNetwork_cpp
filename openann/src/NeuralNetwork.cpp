#include "../include/NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(vector<int> topology) {
    this->topology = topology;
    this->topologySize = topology.size();

    for(int i=0;i<topologySize;i++){
        Layer *l = new Layer(topology.at(i));
        this->layers.push_back(l);
    }

     for(int i=0;i< (topologySize-1);i++){
        Matrix *m = new Matrix(topology.at(i), topology.at(i+1), true);
        this->weightMatrices.push_back(m);
    }
}

void NeuralNetwork::setCurrentInput(vector<double> input) {
    this->input = input;

    for(int i=0;i<input.size();i++) {
        this->layers.at(0)->setVal(i, input.at(i));
    }
}

void NeuralNetwork::setCurrentTarget(vector<double> target) {
    this->target = target;
    for(int i=0;i<target.size();i++) {
        this->layers.at(this->layers.size()-1)->setVal(i, target.at(i));
    }
}

void NeuralNetwork::printToConsole() {
     for(int i=0;i<this->layers.size();i++) {
         cout<<"LAYER: " << i << endl;
         if(i==0) {
             Matrix *m = this->layers.at(i)->matrixifyVals();
             m->printToConsole();
         } else {
             Matrix *m = this->layers.at(i)->matrixifyActivatedVals();
             m->printToConsole();
         }

         cout<< "+++++++++++++++++++++\n";
         if(i < this->layers.size() -1) {
             cout<< "Weight Matrix of " << i <<endl;
             this->getWeightMatrix(i)->printToConsole();
         }
         cout<< "+++++++++++++++++++++\n";
     }
}

void NeuralNetwork::printInputToConsole() {
    for(int i=0;i<this->input.size();i++) {
        cout<<this->input.at(i)<<"\t";
    }
    cout<<endl;
}

void NeuralNetwork::printOutputToConsole() {
    int indexOfOutputLayer = this->layers.size()-1;
    Matrix *outputValues = this->layers.at(indexOfOutputLayer)->matrixifyActivatedVals();
    for(int i=0;i<outputValues->getNumCols();i++) {
        cout<<outputValues->getValue(0,i)<<"\t";
    }
    cout<<endl;
}

void NeuralNetwork::printTargetToConsole() {
     for(int i=0;i<this->target.size();i++) {
        cout<<this->target.at(i)<<"\t";
    }
    cout<<endl;
}

void NeuralNetwork::feedForward() {
    for(int i=0;i< (this->layers.size()-1);i++) {
        Matrix *a = this->getNeuronMatrix(i);
        
        if(i!=0) {
            a = this->getActivatedNeuronMatrix(i);
        }
        Matrix *b = this->getWeightMatrix(i);
        Matrix *c = (new utils::MultiplyMatrix(a,b))->execute();

        for(int c_index=0;c_index< c->getNumCols();c_index++) {
            this->setNeuronValue(i+1, c_index, c->getValue(0, c_index));
        }
    }
}

void NeuralNetwork::setErrors() {
    if(this->target.size() == 0 ) {
        cerr<< "No target for this neural network"<<endl;
        assert(false);
    }
    if(this->target.size() != this->layers.at(this->layers.size()-1)->getNeurons().size()) {
        cerr<< "Target size is not the same as output size: "<< this->layers.at(this->layers.size()-1)->getNeurons().size()<<endl;
        assert(false);
    }
    this->error = 0.0;
    int outputLayerIndex = this->layers.size() - 1;
    vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNeurons();
    errors.clear();
    for(int i=0;i<this->target.size();i++) {
         double tempErr = (outputNeurons.at(i)->getActivatedVal() - target.at(i));
         errors.push_back(tempErr);
         this->error += pow(tempErr,2);
    } 
    this->error *= 0.5;

    this->historicalErrors.push_back(this->error);
    //this->errors.clear();
}

void NeuralNetwork::backPropagation() {

    vector<Matrix *> newWeights;
    Matrix *gradients;

    //output to hidden
    int outputLayerIndex = this->layers.size()-1;
    Matrix *derivedValuesYToZ = this->layers.at(outputLayerIndex)->matrixifyDerivedVals();
    Matrix *gradientsYToZ = new Matrix(1, this->layers.at(outputLayerIndex)->getNeurons().size(), false);
    for(int i=0;i< this->errors.size();i++) {
        double d = derivedValuesYToZ->getValue(0,i);
        double e = this->errors.at(i);
        double g = d*e;
        gradientsYToZ->setValue(0, i, g);
    }

    int lastHiddenLayerIndex = outputLayerIndex - 1;
    Layer *lastHiddenLayer = this->layers.at(lastHiddenLayerIndex);
    Matrix *weightsOutputToHidden = this->weightMatrices.at(lastHiddenLayerIndex);
    Matrix *deltaOutputToHidden = (new utils::MultiplyMatrix(gradientsYToZ->transpose(),
                                        lastHiddenLayer->matrixifyActivatedVals()))->execute()->transpose();

    Matrix *newWeightsOutputToHidden = new Matrix(deltaOutputToHidden->getNumRows(), 
                                                deltaOutputToHidden->getNumCols(), false);                                    
    
    for(int r=0;r<deltaOutputToHidden->getNumRows();r++) {
        for(int c=0;c<deltaOutputToHidden->getNumCols();c++) {
            double originalWeight = weightsOutputToHidden->getValue(r,c);
            double deltaWeight = deltaOutputToHidden->getValue(r,c);
            newWeightsOutputToHidden->setValue(r,c, (originalWeight-deltaWeight));
        }
    }

    newWeights.push_back(newWeightsOutputToHidden);

    //cout<<"Ouput to Hidden new weights"<<endl;
    //newWeightsOutputToHidden->printToConsole();

    gradients = new Matrix(gradientsYToZ->getNumRows(), gradientsYToZ->getNumCols(), false);
    for(int r=0; r<gradientsYToZ->getNumRows();r++) {
        for(int c=0; c<gradientsYToZ->getNumCols();c++) {
            gradients->setValue(r,c, gradientsYToZ->getValue(r,c));
        }
    }
    
    //Moving from last hidden layer towards input layer
    for(int i = (outputLayerIndex-1) ; i>0; i--) {
        Layer *l= this->layers.at(i);
        Matrix *derivedHidden = l->matrixifyDerivedVals();
        Matrix *activatedHidden = l->matrixifyActivatedVals();
        Matrix *derivedGradients = new Matrix(1, l->getNeurons().size(), false); 

        Matrix *weightMatrix = this->weightMatrices.at(i);
        Matrix *originalWeight = this->weightMatrices.at(i-1);

        for(int r=0;r<weightMatrix->getNumRows();r++) {
            double sum = 0;
            for(int c=0;c<weightMatrix->getNumCols();c++) {
                double p = gradients->getValue(0,c) * weightMatrix->getValue(r,c);
                sum += p;
            }
            double g = sum * activatedHidden->getValue(0,r);
            derivedGradients->setValue(0, r, g);
        }

        Matrix *leftNeurons = (i-1) == 0 ? this->layers.at(0)->matrixifyVals() : this->layers.at(i-1)->matrixifyActivatedVals();
        Matrix *deltaWeights = (new utils::MultiplyMatrix(derivedGradients->transpose(),leftNeurons))->execute()->transpose();  
        Matrix *newWeightsHidden = new Matrix(deltaWeights->getNumRows(), deltaWeights->getNumCols(), false);

        for(int r=0;r<newWeightsHidden->getNumRows();r++) {
            for(int c=0;c<newWeightsHidden->getNumCols();c++) {
                newWeightsHidden->setValue(r, c, originalWeight->getValue(r,c) - deltaWeights->getValue(r,c));
            }
        }

        gradients = new Matrix(derivedGradients->getNumRows(), derivedGradients->getNumCols(), false);
        for(int r=0;r<derivedGradients->getNumRows();r++) {
            for(int c=0;c<derivedGradients->getNumCols();c++) {
                gradients->setValue(r,c, derivedGradients->getValue(r,c));
            }
        }

        newWeights.push_back(newWeightsHidden);

    }

   // cout<< "Back prop complete!"<<endl;
    //cout<< "New weights size " <<newWeights.size()<<endl;
    //cout<< "Old weights size" <<this->weightMatrices.size()<<endl;
    // for(int i=0;i<newWeights.size();i++) {
    //     cout<<i<<":"<<endl;
    //     newWeights.at(i)->printToConsole();
    // }
// cout<<"OLd weights"<<endl;
//     for(int i=0;i<this->weightMatrices.size();i++) {
//         cout<<i<<":"<<endl;
//         this->weightMatrices.at(i)->printToConsole();
//     }

    reverse(newWeights.begin(), newWeights.end());
    this->weightMatrices = newWeights;

    // cout<<"Reverse"<<endl;
    // for(int i=0;i<newWeights.size();i++) {
    //     cout<<i<<":"<<endl;
    //     newWeights.at(i)->printToConsole();
    // }

    
    // cout<<"New weights"<<endl;
    //  for(int i=0;i<this->weightMatrices.size();i++) {
    //     cout<<i<<":"<<endl;
    //     this->weightMatrices.at(i)->printToConsole();
    // }

    // cout<<"Updated"<<endl;
    //     for(int i=0;i<newWeights.size();i++) {
    //     cout<<i<<":"<<endl;
    //     newWeights.at(i)->printToConsole();
    // }
}