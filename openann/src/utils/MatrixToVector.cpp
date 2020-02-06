#include "../../include/utils/MatrixToVector.hpp"

utils::MatrixToVector::MatrixToVector(Matrix *a) {
    this->a = a;
}

vector<double> utils::MatrixToVector::execute() {
    vector<double> result;

    for(int i=0;i<a->getNumRows();i++) {
        for(int j=0;j<a->getNumCols();j++) {
            result.push_back(a->getValue(i,j));
        }
    }
    return result;
}