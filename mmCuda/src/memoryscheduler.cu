#include "../include/memoryscheduler.cuh"

unsigned MemoryScheduler::_total_memory;

cub::CachingDeviceAllocator MemoryScheduler::_allocator(true);