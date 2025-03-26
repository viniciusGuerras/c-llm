#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE* file;

//at the moment i will deal only with .txt files for simplicity
int main(){
    file = fopen("dataset/tinishakespeare.txt", "r");
    if (file == NULL) {
        printf("The file is not opened. The program will "
               "now exit.");
        exit(0);
    }

    printf("found sucessfuly");
    char read_char;
    while ((read_char = (fgetc(file))) != EOF)
    {
        printf("%c", read_char);
    }
    
    fclose(file);
    return 0;
}
