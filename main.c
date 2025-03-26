#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// i need to do tensors, they are a generalization of arrays (nd-array) 
//i should use data sequentially, since they need to store a bunch of data

typedef enum{
    INT,
    FLOAT, 
    CHAR,
    STRING
} data_type;

typedef struct{
    void* data;
    data_type d_type;
    int* dim;
    size_t dim_size;
    unsigned int* strides;
    size_t data_size;
} Tensor;

void* tensor_see_index(Tensor* tensor, int* indices, int indices_size){
    if(indices_size > tensor->dim_size){
        printf("tensor dimensional depth is %zu and indices are %d.", tensor->dim_size, indices_size);
        return 0;
    }
    int position = 0;
    for(int i=0; i<(indices_size - 1); i++){
        position += (indices[i] * tensor->strides[i]);
    }
    if(tensor->d_type == INT){
        printf("%d\n", ((int*)tensor->data)[position]);
        return &((int*)tensor->data)[position];
    }
    else if(tensor->d_type == FLOAT){
        printf("%f\n", ((float*)tensor->data)[position]);
        return &((float*)tensor->data)[position];
    }
    else if(tensor->d_type == CHAR){
        printf("%c\n", ((char*)tensor->data)[position]);
        return &((char*)tensor->data)[position];
    }
    else if(tensor->d_type == STRING){
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

    if (tensor->d_type == INT) {
        ((int*)tensor->data)[position] = *(int*)new_value;
    }
    else if (tensor->d_type == FLOAT) {
        ((float*)tensor->data)[position] = *(float*)new_value;
    }
    else if(tensor->d_type == CHAR){
        ((char*)tensor->data)[position] = *(char*)new_value;
    }
    else if(tensor->d_type == STRING){
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

    if(tensor.d_type == INT){
        memcpy(copy->data, tensor.data, data_size * sizeof(int));
    }
    else if(tensor.d_type == FLOAT){
        memcpy(copy->data, tensor.data, data_size * sizeof(float));
    }
    else if(tensor.d_type == CHAR){
        memcpy(copy->data, tensor.data, data_size * sizeof(char));
    }
    else if(tensor.d_type == STRING){
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
    copy->data = calloc(data_size, sizeof(int));
    if(copy->data == NULL){
        printf("error allocating Tensor data.\n");
        return NULL;
    }

    copy->data_size = data_size;

    if(tensor.d_type == INT){
        memcpy(copy->data, tensor.data, data_size * sizeof(int));
    }
    else if(tensor.d_type == FLOAT){
        memcpy(copy->data, tensor.data, data_size * sizeof(float));
    }
    else if(tensor.d_type == CHAR){
        memcpy(copy->data, tensor.data, data_size * sizeof(char));
    }
    else if(tensor.d_type == STRING){
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

    if (tensor.d_type == INT) {
        for(int i=0; i < tensor_len; i++){
            printf("%d ", ((int*)tensor.data)[i]);
        }
    } 
    else if (tensor.d_type == FLOAT) {
        for(int i=0; i < tensor_len; i++){
            printf("%f ", ((float*)tensor.data)[i]);
        }
    }
    else if (tensor.d_type == CHAR){
        for(int i=0; i < tensor_len; i++){
            if(((char*)tensor.data)[i] == '\0'){
                printf("\" \", ");
            }
            else{
                printf("%c ", ((char*)tensor.data)[i]);
            }
        }
    }
    else if(tensor.d_type == STRING){
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

Tensor* tensor_zeroes(int* dim, unsigned int dim_size, data_type d_type){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    size_t tensor_len = tensor_get_len(dim, dim_size);

    if(d_type == INT){
        tensor->data = calloc(tensor_len, sizeof(int));
    }
    else if(d_type == FLOAT){
        tensor->data = calloc(tensor_len, sizeof(float));
    }
    else if(d_type == CHAR){
        tensor->data = calloc(tensor_len, sizeof(char));
    }
    else if(d_type == STRING){
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

Tensor* tensor_rand(int* dim, unsigned int dim_size, data_type d_type){
    //i need to learn to generate pseudo-random numbers
    return NULL;
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

int tensor_check_dtype_equality(Tensor* goal, Tensor* source){
    if(goal->d_type != source->d_type){
        return 0;
    }
    return 1;
}

void tensor_elementwise_add(Tensor* goal, Tensor* source){
    if(tensor_check_dimension_equality(goal, source)){
        size_t size =  goal->data_size;

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                   goal->d_type == INT ? "INT" : "FLOAT",
                   source->d_type == INT ? "INT" : "FLOAT");
            return;
        }
    
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
        size_t size = goal->data_size;

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                   goal->d_type == INT ? "INT" : "FLOAT",
                   source->d_type == INT ? "INT" : "FLOAT");
            return;
        }
    
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
        size_t size = goal->data_size;

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                   goal->d_type == INT ? "INT" : "FLOAT",
                   source->d_type == INT ? "INT" : "FLOAT");
            return;
        }
    
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
        size_t size = goal->data_size;

        if (goal->d_type != source->d_type) {
            printf("Data type mismatch. Goal is %s and Source is %s.\n",
                   goal->d_type == INT ? "INT" : "FLOAT",
                   source->d_type == INT ? "INT" : "FLOAT");
            return;
        }
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

void tensor_scalar_multiplication(Tensor* tensor, void* scalar){
    size_t data_size = tensor->data_size;

    if(tensor->d_type == INT){
        for(int i = 0; i < data_size; i++){
            ((int*)tensor->data)[i] *= *(int*)scalar;
        }
    }
    else if(tensor->d_type == FLOAT){
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
    if(tensor->d_type == INT){
        for(int i = 0; i < tensor->dim[0]; i++){
            for(int j = 0; j < tensor->dim[1]; j++){
                if(j > i){
                    ((int*)tensor->data)[j + pos] = 0;
                }
            }
            pos += tensor->dim[0];
        }
    }
    else if(tensor->d_type == FLOAT){
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
    if(tensor->d_type == INT){
        for(int i = 0; i < tensor->dim[0]; i++){
            for(int j = 0; j < tensor->dim[1]; j++){
                if(j < i){
                    ((int*)tensor->data)[j + pos] = 0;
                }
            }
            pos += tensor->dim[0];
        }
    }
    else if(tensor->d_type == FLOAT){
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

    data_type type = tensors[0]->d_type;
    Tensor* return_tensor = tensor_zeroes(dims, new_dim_size, type);
    size_t tensor_total_size = tensors[0]->data_size;

    if(type == INT){
        for (int i = 0; i < tensors_quantity; i++){
            for(int j = 0; j < tensor_total_size; j++){
                ((int*)return_tensor->data)[(i * tensor_total_size) + j] = ((int*)tensors[i]->data)[j];
            }
        }
    }
    else if(type == FLOAT){
        for (int i = 0; i < tensors_quantity; i++){
            for(int j = 0; j < tensor_total_size; j++){
                ((float*)return_tensor->data)[(i * tensor_total_size) + j] = ((float*)tensors[i]->data)[j];
            }
        }
    }
    else if(type == CHAR){
        for (int i = 0; i < tensors_quantity; i++){
            for(int j = 0; j < tensor_total_size; j++){
                ((char*)return_tensor->data)[(i * tensor_total_size) + j] = ((char*)tensors[i]->data)[j];
            }
        }
    }
    else if(type == STRING){
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
    /*
    int array[3] = {3, 2, 2};
    Tensor* my = tensor_zeroes(array, 3, FLOAT);
    int indices[3] = {2, 2, 1};
    float value = 1;
    void* my_int3 = &value;
    tensor_set_index(my, indices, 3, my_int3);
    Tensor* my2 = tensor_zeroes(array, 3, FLOAT);
    int indices2[3] = {1, 1, 1};
    float value2 = 9;
    void* my_int2 = &value2;
    tensor_set_index(my2, indices2, 3, my_int2);
    Tensor* my3 = tensor_zeroes(array, 3, FLOAT);
    int indices3[3] = {1, 2, 2};
    float value3 = 5;
    void* my_int = &value3;
    tensor_set_index(my3, indices3, 3, my_int);

    Tensor* tensor_array[3] = {my, my2, my3};
    Tensor* stack = tensor_stack(tensor_array, 3);
    printf("\n");
    float scalar = 5.0;
    void* my_scalar = &scalar;
    tensor_scalar_multiplication(stack, my_scalar);
    printf("\n");
    tensor_print_data(*stack);
    */
    int dims[2] = {3, 3};
    Tensor* square = tensor_zeroes(dims, 2, INT);
    /*
    int indices_1[2] = {1, 1};
    int indices_2[2] = {1, 2};
    int indices_3[2] = {1, 3};
    int indices_4[2] = {2, 1};
    int indices_5[2] = {2, 2};
    int indices_6[2] = {2, 3};
    int indices_7[2] = {3, 1};
    int indices_8[2] = {3, 2};
    int indices_9[2] = {3, 3};

    int value_1 = 1;
    void* my_int1 = &value_1;
    int value_2 = 3;
    void* my_int2 = &value_2;
    int value_3 = 8;
    void* my_int3 = &value_3;
    int value_4 = 1;
    void* my_int4 = &value_4;

    tensor_set_index(square, indices_1, 2, my_int1);
    tensor_set_index(square, indices_2, 2, my_int2);
    tensor_set_index(square, indices_3, 2, my_int3);
    tensor_set_index(square, indices_4, 2, my_int4);
    tensor_set_index(square, indices_5, 2, my_int1);
    tensor_set_index(square, indices_6, 2, my_int2);
    tensor_set_index(square, indices_7, 2, my_int3);
    tensor_set_index(square, indices_8, 2, my_int4);
    tensor_set_index(square, indices_9, 2, my_int1);
    */

    tensor_print_data(*square);

    
    /*
    int array[3] = {3, 2, 2};
    Tensor* my = tensor_zeroes(array, 3, STRING);
    char* value = "heyyy";
    void* my_int3 = &value;
    int indices[3] = {2, 2, 2};
    tensor_set_index(my, indices, 3, my_int3);
    tensor_print_data(*my);
    char* anything = "hello, how are you";
    */



}

