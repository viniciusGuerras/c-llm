#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// i need to do tensors, they are a generalization of arrays (nd-array) 
//i should use data sequentially, since they need to store a bunch of data

typedef struct Tensor{
    int* data;
    unsigned* strides;
    int* dim;
    unsigned int dim_size;
} Tensor;

int get_len(int* dim, unsigned int dim_size){
    unsigned int total = 1;
    for(int i = (dim_size - 1); i >= 0; i--){
        total *= dim[i];
    }
    return total;
}
void tensor_calculate_strides(Tensor* tensor){
    unsigned int stride_size = tensor->dim_size;
    int last = 1;
    for (int i = (stride_size - 1); i >= 0; i--){
        tensor->strides[i] = tensor->dim[i] * last;
        last = tensor->dim[i];
    }
}

Tensor* tensor_zeroes(int* dim, unsigned int dim_size){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    unsigned int tensor_len = get_len(dim, dim_size);
    tensor->data = calloc(tensor_len, sizeof(int));
    tensor->dim = dim;
    tensor->dim_size = dim_size;
    tensor->strides = malloc(dim_size * sizeof(int));
    tensor_calculate_strides(tensor);
    return tensor;
}

Tensor* tensor_copy(Tensor* tensor){
    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    copy->data = tensor->data;
    copy->strides = tensor->strides;
    copy->dim = tensor->dim;
    copy->dim_size = tensor->dim_size;
    return copy;
}

void tensor_print_data(Tensor tensor){
    int tensor_len = get_len(tensor.dim, tensor.dim_size);
    for(int i=0; i < (tensor_len - 1); i++){
        printf("%d", tensor.data[i]);
    }
}

void tensor_print_strides(Tensor tensor){
    for(int i=0; i < tensor.dim_size; i++){
        printf("%d", tensor.strides[i]);
    }
}

int main(){ 
    int array[3] = {3, 2, 2};
    Tensor* my = tensor_zeroes(array, 3);
    Tensor* copy = tensor_copy(my);
    tensor_print_strides(*copy);

}
