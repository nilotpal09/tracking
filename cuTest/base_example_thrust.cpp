#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>

#define N 100000
#define MAX_ERR 1


void kernel(std::vector<int> const &x, std::vector<int> const &y, std::vector<int> &out, int n){

    for (int i=0; i<n; i++){
        out[i] = x[i] + y[i];         
    }
}


int main(){

    std::vector<int> x;
    std::vector<int> y;
    std::vector<int> out;

    for (int i=0; i<N; i++){
        x.push_back(rand() % 101);
        y.push_back(rand() % 101);
        out.push_back(13);
    }

    kernel(x, y, out, N);

    // Verification
    for(int i=0; i<N; i++){
        assert(fabs(out[i] - x[i] - y[i]) < MAX_ERR);
    }
    
    return 0;

}
