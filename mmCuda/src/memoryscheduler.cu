#include "../include/memoryscheduler.cuh"

cub::CachingDeviceAllocator MemoryScheduler::_allocator(true);