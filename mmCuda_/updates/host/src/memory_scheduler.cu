#include "host/include/memory_scheduler.cuh"

unsigned MemoryScheduler::_total_memory;

cub::CachingDeviceAllocator MemoryScheduler::_allocator(true);