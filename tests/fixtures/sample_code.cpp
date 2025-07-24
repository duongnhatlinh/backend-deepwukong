#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Sample C++ code with potential vulnerabilities for testing

void vulnerable_function(char* input) {
    char buffer[100];
    
    // Potential buffer overflow vulnerability
    strcpy(buffer, input);  // Line 10 - unsafe string copy
    
    printf("Buffer content: %s\n", buffer);
}

int* return_local_pointer() {
    int local_var = 42;
    
    // Potential null pointer dereference  
    return &local_var;  // Line 18 - returning address of local variable
}

void integer_overflow_example(int a, int b) {
    int result;
    
    // Potential integer overflow
    result = a * b;  // Line 25 - no overflow checking
    
    printf("Result: %d\n", result);
}

void unsafe_memory_access() {
    int* ptr = NULL;
    
    // Potential null pointer dereference
    *ptr = 10;  // Line 33 - dereferencing null pointer
    
    int array[10];
    
    // Potential buffer overflow
    array[15] = 100;  // Line 38 - out of bounds access
}

int main() {
    char large_input[1000];
    memset(large_input, 'A', 999);
    large_input[999] = '\0';
    
    // Call vulnerable functions
    vulnerable_function(large_input);
    
    int* dangerous_ptr = return_local_pointer();
    
    integer_overflow_example(2147483647, 2);
    
    unsafe_memory_access();
    
    return 0;
}