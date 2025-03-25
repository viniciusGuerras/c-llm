#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// i need to do tensors, they are a generalization of arrays (nd-array) 
//i should use data sequentially, since they need to store a bunch of data
typedef enum{
    INT,
    FLOAT
} DataType;

typedef struct{
    void* data;
    DataType d_type;
    unsigned int* strides;
    int* dim;
    unsigned int dim_size;
} Tensor;

void* tensor_see_index(Tensor* tensor, int* indices, int indices_size){
    if(indices_size > tensor->dim_size){
        printf("tensor dimensional depth is %d and indices are %d.", tensor->dim_size, indices_size);
        return 0;
    }
    int position = 0;
    for(int i=0; i<(indices_size - 1); i++){
        position += (indices[i] * tensor->strides[i]);
    }
    if(tensor->d_type == INT){
        printf("%f\n", ((int*)tensor->data)[position]);
        return &((int*)tensor->data)[position];
    }
    else if(tensor->d_type == FLOAT){
        printf("%d\n", ((float*)tensor->data)[position]);
        return &((float*)tensor->data)[position];
    }
    return NULL;
}

void tensor_set_index(Tensor* tensor, int* indices, int indices_size, void* new_value){
    if(indices_size > tensor->dim_size){
        printf("tensor dimensional depth is %d and indices are %d.", tensor->dim_size, indices_size);
        return;
    }
    int position = 0;
    for(int i=0; i< indices_size; i++){
        position += (indices[i] - 1) * tensor->strides[i];
    }

    if (tensor->d_type == INT) {
        ((int*)tensor->data)[position] = *(int*)new_value;
    } else if (tensor->d_type == FLOAT) {
        ((float*)tensor->data)[position] = *(float*)new_value;
    } else {
        printf("Unsupported data type.\n");
    }
}

int tensor_get_len(int* dim, unsigned int dim_size){
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

    size_t data_size = (size_t)tensor_get_len(tensor.dim, tensor.dim_size);
    copy->data = (int*)malloc(data_size * sizeof(int));
    if(copy->data == NULL){
        printf("error allocating Tensor data.\n");
        return NULL;
    }

    memcpy(copy->data, tensor.data, data_size * sizeof(int));
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

Tensor* tensor_like(Tensor tensor){
    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    if (copy == NULL){
        printf("error allocating Tensor.\n");
        return NULL;
    }

    size_t data_size = (size_t)tensor_get_len(tensor.dim, tensor.dim_size);
    copy->data = calloc(data_size, sizeof(int));
    if(copy->data == NULL){
        printf("error allocating Tensor data.\n");
        return NULL;
    }

    memcpy(copy->data, tensor.data, data_size * sizeof(int));
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

void tensor_print_data(Tensor tensor){
    int tensor_len = tensor_get_len(tensor.dim, tensor.dim_size);
    if (tensor.d_type == INT) {
        for(int i=0; i < tensor_len; i++){
            printf("%d", ((int*)tensor.data)[i]);
        }
    } else if (tensor.d_type == FLOAT) {
        for(int i=0; i < tensor_len; i++){
            printf("%f", ((float*)tensor.data)[i]);
        }
    } else {
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

Tensor* tensor_zeroes(int* dim, unsigned int dim_size){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    unsigned int tensor_len = tensor_get_len(dim, dim_size);
    tensor->data = calloc(tensor_len, sizeof(int));
    tensor->dim = dim;
    tensor->dim_size = dim_size;
    tensor->strides = malloc(dim_size * sizeof(int));
    tensor_calculate_strides(tensor);
    return tensor;
}

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

void tensor_elementwise_add(Tensor* goal, Tensor* source){
    if(tensor_check_dimension_equality(goal, source)){
        int size = tensor_get_len(goal->dim, goal->dim_size);

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                   goal->d_type == INT ? "INT" : "FLOAT",
                   source->d_type == INT ? "INT" : "FLOAT");
            return;
        }
    
        int size = tensor_get_len(goal->dim, goal->dim_size);
    
        if (goal->d_type == INT) {
            int* goal_data = (int*)goal->data;
            int* source_data = (int*)source->data;
            for (int i = 0; i < size; i++) {
                goal_data[i] += source_data[i];
            }
        } else if (goal->d_type == FLOAT) {
            float* goal_data = (float*)goal->data;
            float* source_data = (float*)source->data;
            for (int i = 0; i < size; i++) {
                goal_data[i] += source_data[i];
            }
        } else {
            printf("Unsupported data type.\n");
        }
    }
}

void tensor_elementwise_subtract(Tensor* goal, Tensor* source){
    if(tensor_check_dimension_equality(goal, source)){
        int size = tensor_get_len(goal->dim, goal->dim_size);

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                   goal->d_type == INT ? "INT" : "FLOAT",
                   source->d_type == INT ? "INT" : "FLOAT");
            return;
        }
    
        int size = tensor_get_len(goal->dim, goal->dim_size);
    
        if (goal->d_type == INT) {
            int* goal_data = (int*)goal->data;
            int* source_data = (int*)source->data;
            for (int i = 0; i < size; i++) {
                goal_data[i] -= source_data[i];
            }
        } else if (goal->d_type == FLOAT) {
            float* goal_data = (float*)goal->data;
            float* source_data = (float*)source->data;
            for (int i = 0; i < size; i++) {
                goal_data[i] -= source_data[i];
            }
        } else {
            printf("Unsupported data type.\n");
        }
    }
}

void tensor_elementwise_multiply(Tensor* goal, Tensor* source){
    if(tensor_check_dimension_equality(goal, source)){
        int size = tensor_get_len(goal->dim, goal->dim_size);

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                   goal->d_type == INT ? "INT" : "FLOAT",
                   source->d_type == INT ? "INT" : "FLOAT");
            return;
        }
    
        int size = tensor_get_len(goal->dim, goal->dim_size);
    
        if (goal->d_type == INT) {
            int* goal_data = (int*)goal->data;
            int* source_data = (int*)source->data;
            for (int i = 0; i < size; i++) {
                goal_data[i] *= source_data[i];
            }
        } else if (goal->d_type == FLOAT) {
            float* goal_data = (float*)goal->data;
            float* source_data = (float*)source->data;
            for (int i = 0; i < size; i++) {
                goal_data[i] *= source_data[i];
            }
        } else {
            printf("Unsupported data type.\n");
        }
    }
}


void tensor_elementwise_divide(Tensor* goal, Tensor* source){
    if(tensor_check_dimension_equality(goal, source)){
        int size = tensor_get_len(goal->dim, goal->dim_size);

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                   goal->d_type == INT ? "INT" : "FLOAT",
                   source->d_type == INT ? "INT" : "FLOAT");
            return;
        }
    
        int size = tensor_get_len(goal->dim, goal->dim_size);
    
        if (goal->d_type == INT) {
            int* goal_data = (int*)goal->data;
            int* source_data = (int*)source->data;
            for (int i = 0; i < size; i++) {
                goal_data[i] /= source_data[i];
            }
        } else if (goal->d_type == FLOAT) {
            float* goal_data = (float*)goal->data;
            float* source_data = (float*)source->data;
            for (int i = 0; i < size; i++) {
                goal_data[i] /= source_data[i];
            }
        } else {
            printf("Unsupported data type.\n");
        }
    }
}

Tensor* tensor_stack(Tensor** tensors, size_t tensors_quantity){
    for(int i = 0; i<(tensors_quantity-1);i++){
        if(tensor_check_dimension_equality(tensors[i], tensors[i+1]) == 0){
           printf("error in stacking process.");
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

    Tensor* return_tensor = tensor_zeroes(dims, new_dim_size);
    size_t tensor_total_size = (size_t)tensor_get_len(tensors[0]->dim, tensors[0]->dim_size);

    for (int i = 0; i < tensors_quantity; i++){
        for(int j = 0; j < tensor_total_size; j++){
            return_tensor->data[(i * tensor_total_size) + j] = tensors[i]->data[j];
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
    int array[3] = {3, 2, 2};
    Tensor* my = tensor_zeroes(array, 3);
    size_t size = tensor_get_len(my->dim, my->dim_size);
    int indices[3] = {2, 2, 1};
    tensor_set_index(my, indices, 3, 5);
    Tensor* my2 = tensor_zeroes(array, 3);
    int indices2[3] = {1, 1, 1};
    tensor_set_index(my2, indices2, 3, 9);
    Tensor* my3 = tensor_zeroes(array, 3);
    int indices3[3] = {1, 2, 2};
    tensor_set_index(my3, indices3, 3, 1);
    tensor_print_data(*my);
    tensor_print_data(*my2);
    tensor_print_data(*my3);
    Tensor* tensor_array[3] = {my, my2, my3};

    Tensor* stack = tensor_stack(tensor_array, 3);
    tensor_print_data(*stack);   
}
