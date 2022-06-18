#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <random>


#define N_TRIPLET 1000
#define N_MODULES 100
#define N_MAX_HITS 10


// runs on module triplets
// will become the kernal later
void fun_mTriplet(
    std::vector<unsigned long int> const &flatten_triplets, 
    std::vector<std::vector<float> > const &input_hits, 
    unsigned long int n) {

    for(int i = 0; i < n; i++){

        unsigned long int module1_idx = flatten_triplets[i*3];
        unsigned long int module2_idx = flatten_triplets[i*3 + 1];
        unsigned long int module3_idx = flatten_triplets[i*3 + 2];

        int sum = input_hits.at(module1_idx).size() + input_hits.at(module2_idx).size() + input_hits.at(module3_idx).size();
    }
}


int main(){

    // random seeding (module triple indices)
    std::random_device rd1;
    std::mt19937 rng1(rd1());
    std::uniform_int_distribution<unsigned long int> uni1(0,N_MODULES-1);


    // random seeding (num hits)
    std::random_device rd2;
    std::mt19937 rng2(rd2());
    std::uniform_int_distribution<int> uni2(0,N_MAX_HITS);


    // random seeding (hit values)
    std::random_device rd3;
    std::mt19937 rng3(rd3());
    std::uniform_real_distribution<float> uni3(-5,5);


    // contains only the module indices of the triplets (flatten)
    std::vector<unsigned long int> flatten_triplets;
    for (unsigned long int i=0; i<3*N_TRIPLET; i++){
        unsigned long int random_integer = uni1(rng1);
        flatten_triplets.push_back(random_integer);
    }


    // contains the actual data (hits on each module)
    std::vector<std::vector<float> > input_hits;
    for (unsigned long int i=0; i<N_MODULES; i++){

        int num_hits = uni2(rng2);
        std::vector<float> hit_vector;

        for (int j=0; j<num_hits; j++){
            float random_float = uni3(rng3);
            hit_vector.push_back(random_float);
        }

        input_hits.push_back(hit_vector);
    }

    fun_mTriplet(flatten_triplets, input_hits, N_TRIPLET);

    printf("PASSED\n");
}