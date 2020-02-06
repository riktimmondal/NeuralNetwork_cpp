#include "../../include/utils/MultiplyMatrix.hpp"

utils::MultiplyMatrix::MultiplyMatrix(Matrix *a, Matrix *b) {
    this->a = a;
    this->b = b;
    if(a->getNumCols() != b->getNumRows()) {
        cout<<"Multiply matrix dimension mismatch\n";
        assert(false);
    }

    this->c = new Matrix(a->getNumRows(), b->getNumCols(), false);
}

Matrix *utils::MultiplyMatrix::execute() {
    for(int i=0;i<a->getNumRows();i++) {
        for(int j=0;j<b->getNumCols() ;j++) {
            for(int k=0;k<b->getNumRows();k++) {
                double pro = this->a->getValue(i,k) * this->b->getValue(k,j);
                double newVal = this->c->getValue(i,j) + pro;
                this->c->setValue(i,j,newVal);
            }
        }
    }
    return this->c;
}