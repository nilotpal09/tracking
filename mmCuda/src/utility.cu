#include "../include/utility.cuh"


unsigned findPairIndex(unsigned a, unsigned b, std::vector<unsigned>& offsets, std::vector<unsigned>& indices) {
  unsigned offset = offsets[a];
  unsigned end = offsets[a + 1] - offset;

  unsigned idx;
  bool found = false;
  for (unsigned i = 0; i < end; i++) {
    if (indices[offset + i] == b) {
      idx = i;
      found = true;
    }
  }
  if (!found) {
    std::cout << "Pair not found" << std::endl;
  }
  //std::cout << "Pair " << a << " " << b << " found at " << offset + idx << std::endl;
  return offset + idx;
}

void test_PrefixSum() {
  int *d_in, *d_out;
  int size = 10;
  int *h_in = new int[size];
  int *h_out = new int[size + 1];

  for (int i = 0; i < size; i++) {
    h_in[i] = i;
  }

  cudaMalloc(&d_in, size * sizeof(int));
  cudaMalloc(&d_out, (size + 1) * sizeof(int));
  cudaMemset(d_out, 0, (size + 1) * sizeof(int)); // Memset to 0 is important

  cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

  PrefixSum(&d_in, &d_out, size);

  cudaMemcpy(h_out, d_out, (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

  // int sum = 0;
  for (int i = 0; i < size; i++) {
    assert(h_out[i + 1] - h_out[i] == h_in[i]);
    // sum += h_in[i];
    // assert(h_out[i] == sum);
    std::cout << h_out[i + 1] - h_out[i] << " " << h_in[i] << std::endl;
  }

  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  delete[] h_out;
}

void test_Sort() {
  int *d_keys_in, *d_keys_out;
  int *d_values_in, *d_values_out;
  int size = 10;
  int *h_keys_in = new int[size];
  int *h_keys_out = new int[size];
  int *h_values_in = new int[size];
  int *h_values_out = new int[size];

  for (int i = 0; i < size; i++) {
    h_keys_in[i] = size - i;
    h_values_in[i] = i;
  }

  cudaMalloc(&d_keys_in, size * sizeof(int));
  cudaMalloc(&d_keys_out, size * sizeof(int));
  cudaMalloc(&d_values_in, size * sizeof(int));
  cudaMalloc(&d_values_out, size * sizeof(int));

  cudaMemcpy(d_keys_in, h_keys_in, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values_in, h_values_in, size * sizeof(int),
             cudaMemcpyHostToDevice);

  Sort(&d_keys_in, &d_values_in, &d_keys_out, &d_values_out, size);

  cudaMemcpy(h_keys_out, d_keys_out, size * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_values_out, d_values_out, size * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++) {
    std::cout << h_keys_out[i] << " " << h_values_out[i] << std::endl;
  }

  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  delete[] h_keys_in;
  delete[] h_keys_out;
  delete[] h_values_in;
  delete[] h_values_out;
}

void test_ReduceFlagged() {
  int *d_flags_in, *d_values_in, *d_values_out;
  int size = 10;
  int *h_flags_in = new int[size];
  int *h_values_in = new int[size];
  int *h_values_out = new int[size];

  for (int i = 0; i < size; i++) {
    h_flags_in[i] = i % 2;
    h_values_in[i] = i;
  }

  cudaMalloc(&d_flags_in, size * sizeof(int));
  cudaMalloc(&d_values_in, size * sizeof(int));
  cudaMalloc(&d_values_out, size * sizeof(int));

  cudaMemcpy(d_flags_in, h_flags_in, size * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values_in, h_values_in, size * sizeof(int),
             cudaMemcpyHostToDevice);

  ReduceFlagged(&d_flags_in, &d_values_in, &d_values_out, size);

  cudaMemcpy(h_values_out, d_values_out, size * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++) {
    std::cout << h_values_out[i] << std::endl;
  }

  cudaFree(d_flags_in);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  delete[] h_flags_in;
  delete[] h_values_in;
  delete[] h_values_out;
}