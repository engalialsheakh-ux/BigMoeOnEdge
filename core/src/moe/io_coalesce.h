#pragma once

#include <cstddef>
#include <cstdint>

namespace bmoe::detail {

struct CoalesceSpan {
    uint64_t offset;
    uint64_t bytes;
    uintptr_t destination;
    int file;
    int layer;
    int projection;
    int first_expert;
    int count;
    int flag;
    bool speculative;
};

// Exact adjacency in both file and fixed expert arena means one read and one memcpy
// serve all slices, with no gap bytes or additional scatter buffer.
inline bool can_coalesce(const CoalesceSpan & left, const CoalesceSpan & right, uint64_t max_bytes) {
    return !left.speculative && !right.speculative && left.file == right.file && left.layer == right.layer &&
           left.projection == right.projection && left.first_expert + left.count == right.first_expert &&
           left.flag >= 0 && right.flag == left.flag + left.count && left.offset <= UINT64_MAX - left.bytes &&
           left.offset + left.bytes == right.offset && left.destination <= UINTPTR_MAX - left.bytes &&
           left.destination + left.bytes == right.destination && left.bytes <= max_bytes &&
           right.bytes <= max_bytes - left.bytes;
}

inline uint64_t aligned_window(uint64_t offset, uint64_t bytes, uint64_t alignment) {
    const uint64_t begin = offset / alignment * alignment;
    const uint64_t end = ((offset + bytes + alignment - 1) / alignment) * alignment;
    return end - begin;
}

} // namespace bmoe::detail
