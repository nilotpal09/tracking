#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N_TRIPLET 1000


// The plan so far...

// module_triplets is a 1D arrays
// ((1,2,3), (6,7,8)) will be represented as (1,2,3,6,7,8)

// A hit is an array of three numbers (floats)
// (x,y,z)

// hits in module will be stored in a 1D array (flattened hit x,y,zs)
// similar to module_triplets

// all the hits (in all the modules) will be in an object of some class
// Need to develop the structure of the class as we progress




class InputHits {



    // do the selection and stuff here
    // it's easier this way, since everything is already here
    // and then decorate the final result (yet to figure out the format)


    // makes edges for one module triplet
    // can call it inside the cuda kernal later (parallelizable)
    void make_edges(){

    }


}



// runs on module triplets
// will become the kernal later
void fun_mTriplet(float *t_arr, float *a, float *b, int n) {
    
    for(int i = 0; i < n; i++){

        int module1 = t_arr[i*3];
        int module2 = t_arr[i*3 + 1];
        int module3 = t_arr[i*3 + 2];




    }
}




int main(){
    float *a, *b, *out; 

    // Allocate memory for module triple array
    int *triplet_array = (int*)malloc(sizeof(int) * 3*N_TRIPLET);

    // Initializing the triplet array
    for(int i = 0; i < 3*N_TRIPLET; i++){
        a[i] = rand() % 100;

    // Allocate memory for result


    // create the inputHits object
    inputHits = InputHits()


    fun_mTriplet(triplet_array, inputHits, N_TRIPLET)


    printf("PASSED\n");
}