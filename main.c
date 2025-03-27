#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// i need to do tensors, they are a generalization of arrays (nd-array) 
//i should use data sequentially, since they need to store a bunch of data

typedef enum{
    TAG_INT = (1 << 0),
    TAG_FLOAT = (1 << 1), 
    TAG_CHAR = (1 << 2),
    TAG_STRING = (1 << 3)
} t_flag;

typedef struct{
    void* data;
    t_flag d_type;
    int* dim;
    size_t dim_size;
    unsigned int* strides;
    size_t data_size;
} Tensor;

#define ELEMENTWISE_OPERATION(x, y, size, operation_char, type) {            \
    for (int i = 0; i < size; i++) {                                         \
        ((type*)(x->data))[i] operation_char##= ((type*)(y->data))[i];       \
    }                                                                        \
}

void* tensor_see_index(Tensor* tensor, int* indices, int indices_size){
    if(indices_size > tensor->dim_size){
        printf("tensor dimensional depth is %zu and indices are %d.", tensor->dim_size, indices_size);
        return 0;
    }
    int position = 0;
    for(int i=0; i<(indices_size - 1); i++){
        position += (indices[i] * tensor->strides[i]);
    }
    if(tensor->d_type & TAG_INT){
        printf("%d\n", ((int*)tensor->data)[position]);
        return &((int*)tensor->data)[position];
    }
    if(tensor->d_type & TAG_FLOAT){
        printf("%f\n", ((float*)tensor->data)[position]);
        return &((float*)tensor->data)[position];
    }
    if(tensor->d_type & TAG_CHAR){
        printf("%c\n", ((char*)tensor->data)[position]);
        return &((char*)tensor->data)[position];
    }
    if(tensor->d_type & TAG_STRING){
        printf("%s\n", ((char**)tensor->data)[position]);
        return &((char**)tensor->data)[position];
    }
    return NULL;
}

void tensor_set_index(Tensor* tensor, int* indices, int indices_size, void* new_value){
    if(indices_size > tensor->dim_size){
        printf("tensor dimensional depth is %zu and indices are %d.", tensor->dim_size, indices_size);
        return;
    }
    int position = 0;
    for(int i=0; i< indices_size; i++){
        position += (indices[i] - 1) * tensor->strides[i];
    }

    if (tensor->d_type & TAG_INT) {
        ((int*)tensor->data)[position] = *(int*)new_value;
    }
    else if (tensor->d_type & TAG_FLOAT) {
        ((float*)tensor->data)[position] = *(float*)new_value;
    }
    else if(tensor->d_type & TAG_CHAR){
        ((char*)tensor->data)[position] = *(char*)new_value;
    }
    else if(tensor->d_type & TAG_STRING){
        ((char**)tensor->data)[position] = *(char**)new_value;
    }
    else 
    {
        printf("Unsupported data type.\n");
    }
}

size_t tensor_get_len(int* dim, unsigned int dim_size){
    unsigned int total = 1;
    for(int i = (dim_size - 1); i >= 0; i--){
        total *= dim[i];
    }
    return total;
}

Tensor* tensor_copy(Tensor tensor){
    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    if (copy == NULL){
        printf("error allocating Tensor.\n");
        return NULL;
    }

    size_t data_size = tensor.data_size;
    copy->data = (int*)malloc(data_size * sizeof(int));
    if(copy->data == NULL){
        printf("error allocating Tensor data.\n");
        return NULL;
    }

    copy->data_size = data_size;

    if(tensor.d_type & TAG_INT){
        memcpy(copy->data, tensor.data, data_size * sizeof(int));
    }
    else if(tensor.d_type & TAG_FLOAT){
        memcpy(copy->data, tensor.data, data_size * sizeof(float));
    }
    else if(tensor.d_type & TAG_CHAR){
        memcpy(copy->data, tensor.data, data_size * sizeof(char));
    }
    else if(tensor.d_type & TAG_STRING){
        memcpy(copy->data, tensor.data, data_size * sizeof(char*));
    }

    copy->strides = (unsigned int*)malloc(tensor.dim_size * sizeof(unsigned int));
    if(copy->strides == NULL){
        printf("error allocating Tensor strides.\n");
        return NULL;
    }

    memcpy(copy->strides, tensor.strides, tensor.dim_size * sizeof(unsigned int));
    copy->dim = (int*)malloc(tensor.dim_size * sizeof(int));
    if(copy->dim == NULL){
        printf("error allocating Tensor dims.\n");
        return NULL;
    }

    copy->d_type = tensor.d_type;

    memcpy(copy->dim, tensor.dim, tensor.dim_size * sizeof(int*));
    copy->dim_size = tensor.dim_size;
    return copy;
}

//tensor_random()
Tensor* tensor_like(Tensor tensor){
    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    if (copy == NULL){
        printf("error allocating Tensor.\n");
        return NULL;
    }

    size_t data_size = tensor_get_len(tensor.dim, tensor.dim_size);
    copy->data = (int*)malloc(data_size * sizeof(int));
    if(copy->data == NULL){
        printf("error allocating Tensor data.\n");
        return NULL;
    }

    copy->data_size = data_size;

    if(tensor.d_type & TAG_INT){
        memcpy(copy->data, tensor.data, data_size * sizeof(int));
    }
    else if(tensor.d_type & TAG_FLOAT){
        memcpy(copy->data, tensor.data, data_size * sizeof(float));
    }
    else if(tensor.d_type & TAG_CHAR){
        memcpy(copy->data, tensor.data, data_size * sizeof(char));
    }
    else if(tensor.d_type & TAG_STRING){
        memcpy(copy->data, tensor.data, data_size * sizeof(char*));
    }

    copy->strides = (unsigned int*)malloc(tensor.dim_size * sizeof(unsigned int));
    if(copy->strides == NULL){
        printf("error allocating Tensor strides.\n");
        return NULL;
    }

    memcpy(copy->strides, tensor.strides, tensor.dim_size * sizeof(unsigned int));
    copy->dim = (int*)malloc(tensor.dim_size * sizeof(int));
    if(copy->dim == NULL){
        printf("error allocating Tensor dims.\n");
        return NULL;
    }

    memcpy(copy->dim, tensor.dim, tensor.dim_size * sizeof(int*));
    copy->dim_size = tensor.dim_size;
    return copy;
}

//correct this print method
//
void tensor_print_data(Tensor tensor){
    size_t tensor_len = tensor.data_size;

    if (tensor.d_type & TAG_INT) {
        for(int i=0; i < tensor_len; i++){
            printf("%d ", ((int*)tensor.data)[i]);
        }
    } 
    else if (tensor.d_type & TAG_FLOAT) {
        for(int i=0; i < tensor_len; i++){
            printf("%f ", ((float*)tensor.data)[i]);
        }
    }
    else if (tensor.d_type & TAG_CHAR){
        for(int i=0; i < tensor_len; i++){
            if(((char*)tensor.data)[i] == '\0'){
                printf("\" \", ");
            }
            else{
                printf("%c ", ((char*)tensor.data)[i]);
            }
        }
    }
    else if(tensor.d_type & TAG_STRING){
        for(int i=0; i < tensor_len; i++){
            if(strlen(((char**)tensor.data)[i]) == 0){
                printf("\" \", ");
            }
            else{
                printf("%s, ", ((char**)tensor.data)[i]);
            }
        }
    }
    else {
        printf("Unsupported data type.\n");
    }
    printf("\n");
}

void tensor_print_strides(Tensor tensor){
    for(int i=0; i < tensor.dim_size; i++){
        printf("%d", tensor.strides[i]);
    }
    printf("\n");
}

void tensor_calculate_strides(Tensor* tensor){
    unsigned int stride_size = tensor->dim_size;

    //se the last stride as one
    tensor->strides[stride_size - 1] = 1;

    int last = 1;
    int counter = 1;
    for (int i = (stride_size - 2); i >= 0; i--){
        last = tensor->dim[i + 1];
        tensor->strides[i] = counter * last;
        counter++;
    }
}

Tensor* tensor_zeroes(int* dim, unsigned int dim_size, t_flag d_type){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    size_t tensor_len = tensor_get_len(dim, dim_size);

    if(d_type & TAG_INT){
        tensor->data = calloc(tensor_len, sizeof(int));
    }
    else if(d_type & TAG_FLOAT){
        tensor->data = calloc(tensor_len, sizeof(float));
    }
    else if(d_type & TAG_CHAR){
        tensor->data = calloc(tensor_len, sizeof(char));
    }
    else if(d_type & TAG_STRING){
        tensor->data = (char*)malloc(tensor_len * sizeof(char));
        for(int i = 0; i < tensor_len; i++){
            ((char**)tensor->data)[i] = "";
        }
    }

    tensor->dim = dim;
    tensor->dim_size = dim_size;
    tensor->data_size = tensor_len;
    tensor->strides = malloc(dim_size * sizeof(int));
    tensor->d_type = d_type;
    tensor_calculate_strides(tensor);
    return tensor;
}

// Tensor* tensor_rand(int* dim, unsigned int dim_size, data_type d_type){
//     //i need to learn to generate pseudo-random numbers
//     return NULL;
// }

int tensor_check_dimension_equality(Tensor* goal, Tensor* source){
    if(goal->dim_size != source->dim_size){
        printf("tamanho das dimensões não correspondem");
        return 0;
    }
    for(int i = 0; i < goal->dim_size; i++){
        if(goal->dim[i] != source->dim[i]){
            printf("dimensões não correspondem");
            return 0;
        }
    }
    return 1;
}

int tensor_check_dtype_equality(Tensor* goal, Tensor* source){
    if(goal->d_type != source->d_type){
        return 0;
    }
    return 1;
}

void tensor_elementwise_operation(Tensor* goal, Tensor* source, char operation){
    if(tensor_check_dimension_equality(goal, source)){
        size_t size = goal->data_size;

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                    goal->d_type == TAG_INT ? "TAG_INT" : "TAG_FLOAT",
                    source->d_type == TAG_INT ? "TAG_INT" : "TAG_FLOAT");
            return;
        } 

        t_flag type = goal->d_type;

        if(type & TAG_INT){
            switch (operation)
            {
            case '+':
                ELEMENTWISE_OPERATION(goal, source, size, +, int); break;
            case '-':
                ELEMENTWISE_OPERATION(goal, source, size, -,  int); break;
            case '*':
                ELEMENTWISE_OPERATION(goal, source, size, *, int); break;
            case '/':
                ELEMENTWISE_OPERATION(goal, source, size, /, int); break;
            default:
                break;
            }
        }
        else{
            switch (operation)
            {
            case '+':
                ELEMENTWISE_OPERATION(goal, source, size, +, float); break;
            case '-':
                ELEMENTWISE_OPERATION(goal, source, size, -,  float); break;
            case '*':
                ELEMENTWISE_OPERATION(goal, source, size, *, float); break;
            case '/':
                ELEMENTWISE_OPERATION(goal, source, size, /, float); break;
            default:
                break;
            }
        }
    }

}

void tensor_scalar_multiplication(Tensor* tensor, void* scalar){
    size_t data_size = tensor->data_size;

    if(tensor->d_type & TAG_INT){
        for(int i = 0; i < data_size; i++){
            ((int*)tensor->data)[i] *= *(int*)scalar;
        }
    }
    else if(tensor->d_type & TAG_FLOAT){
        for(int i = 0; i < data_size; i++){
            ((float*)tensor->data)[i] *= *(float*)scalar;
        }
    }
}

void tensor_triu(Tensor* tensor){
    if(tensor->dim_size != 2){
        printf("must be a matrice");
        return;
    }
    if(tensor->dim[0] != tensor->dim[1]){
        printf("must be a square mstrice.");
        return;
    }

    int pos = 0;
    if(tensor->d_type & TAG_INT){
        for(int i = 0; i < tensor->dim[0]; i++){
            for(int j = 0; j < tensor->dim[1]; j++){
                if(j > i){
                    ((int*)tensor->data)[j + pos] = 0;
                }
            }
            pos += tensor->dim[0];
        }
    }
    else if(tensor->d_type & TAG_FLOAT){
        for(int i = 0; i < tensor->dim[0]; i++){
            for(int j = 0; j < tensor->dim[1]; j++){
                if(j > i){
                    ((float*)tensor->data)[j + pos] = 0.0;
                }
            }
            pos += tensor->dim[0];
        }
    }
}

void tensor_tril(Tensor* tensor){
    if(tensor->dim_size != 2){
        printf("must be a matrice");
        return;
    }
    if(tensor->dim[0] != tensor->dim[1]){
        printf("must be a square mstrice.");
        return;
    }

    int pos = 0;
    if(tensor->d_type & TAG_INT){
        for(int i = 0; i < tensor->dim[0]; i++){
            for(int j = 0; j < tensor->dim[1]; j++){
                if(j < i){
                    ((int*)tensor->data)[j + pos] = 0;
                }
            }
            pos += tensor->dim[0];
        }
    }
    else if(tensor->d_type & TAG_FLOAT){
        for(int i = 0; i < tensor->dim[0]; i++){
            for(int j = 0; j < tensor->dim[1]; j++){
                if(j < i){
                    ((float*)tensor->data)[j + pos] = 0.0;
                }
            }
            pos += tensor->dim[0];
        }
    }
}

Tensor* tensor_stack(Tensor** tensors, size_t tensors_quantity){
    for(int i = 0; i<(tensors_quantity-1);i++){
        if(tensor_check_dimension_equality(tensors[i], tensors[i+1]) == 0){
           printf("error in stacking process.");
           return NULL;
        }
        if(tensor_check_dtype_equality(tensors[i], tensors[i+1]) == 0){
            printf("data types dont match.");
            return NULL;
        }
    }

    size_t new_dim_size = tensors[0]->dim_size + 1;
    int* dims = (int*)malloc(new_dim_size * sizeof(int));

    if(dims == NULL){
        printf("error allocating Tensor dims.\n");
        return NULL;
    }

    dims[0] = tensors_quantity;

    int* pointer = &dims[1];
    memcpy(pointer, tensors[0]->dim, tensors[0]->dim_size * sizeof(int)); 

    t_flag type = tensors[0]->d_type;
    Tensor* return_tensor = tensor_zeroes(dims, new_dim_size, type);
    size_t tensor_total_size = tensors[0]->data_size;

    if(type & TAG_INT){
        for (int i = 0; i < tensors_quantity; i++){
            for(int j = 0; j < tensor_total_size; j++){
                ((int*)return_tensor->data)[(i * tensor_total_size) + j] = ((int*)tensors[i]->data)[j];
            }
        }
    }
    else if(type & TAG_FLOAT){
        for (int i = 0; i < tensors_quantity; i++){
            for(int j = 0; j < tensor_total_size; j++){
                ((float*)return_tensor->data)[(i * tensor_total_size) + j] = ((float*)tensors[i]->data)[j];
            }
        }
    }
    else if(type & TAG_CHAR){
        for (int i = 0; i < tensors_quantity; i++){
            for(int j = 0; j < tensor_total_size; j++){
                ((char*)return_tensor->data)[(i * tensor_total_size) + j] = ((char*)tensors[i]->data)[j];
            }
        }
    }
    else if(type & TAG_STRING){
        for (int i = 0; i < tensors_quantity; i++){
            for(int j = 0; j < tensor_total_size; j++){
                ((char**)return_tensor->data)[(i * tensor_total_size) + j] = ((char**)tensors[i]->data)[j];
            }
        }
    }
    return return_tensor;
}

//view
//broadcasting
//Reduction operations (sum, max, min) across specific axes.
//Transpose or reshape functionalities.

int tensor_slice(Tensor* tensor, int* indices){
    return 0;
}

int view(Tensor* tensor){
    return 0;
}

int main(){


}