#include "io_coalesce.h"
#include "file_reader.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

int main() {
    using bmoe::detail::CoalesceSpan;
    constexpr uint64_t slice = 512 * 1024, limit = 4 * 1024 * 1024, first_offset = 3584;
    CoalesceSpan first{first_offset, slice, 0x100000, 0, 1, 0, 7, 1, 7, false};
    CoalesceSpan next{first_offset + slice, slice, 0x100000 + slice, 0, 1, 0, 8, 1, 8, false};
    if (!bmoe::detail::can_coalesce(first, next, limit)) return 1;
    CoalesceSpan gap = next;
    ++gap.offset;
    if (bmoe::detail::can_coalesce(first, gap, limit)) return 2;
    gap = next;
    ++gap.destination;
    if (bmoe::detail::can_coalesce(first, gap, limit)) return 3;
    gap = next;
    ++gap.layer;
    if (bmoe::detail::can_coalesce(first, gap, limit)) return 4;
    if (bmoe::detail::can_coalesce(first, next, slice)) return 5;
    const uint64_t separate = bmoe::detail::aligned_window(first.offset, first.bytes, 4096) +
                              bmoe::detail::aligned_window(next.offset, next.bytes, 4096);
    const uint64_t merged = bmoe::detail::aligned_window(first.offset, first.bytes + next.bytes, 4096);
    if (separate - merged != 4096) return 6;

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("bmoe-coalesce-" + std::to_string(nonce) + ".bin");
    std::vector<char> input(first_offset + 2 * slice + 4096);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = char((i * 37 + 11) & 255);
    {
        std::ofstream output(path, std::ios::binary);
        output.write(input.data(), std::streamsize(input.size()));
    }
    bmoe::FileReader reader;
    if (!reader.open(path.string(), 1, true, 4096, limit + 4096)) return 7;
    std::vector<char> separate_bytes(2 * slice), merged_bytes(2 * slice);
    if (reader.read(0, separate_bytes.data(), first.offset, slice) < 0 ||
        reader.read(0, separate_bytes.data() + slice, next.offset, slice) < 0 ||
        reader.read(0, merged_bytes.data(), first.offset, 2 * slice) < 0 || separate_bytes != merged_bytes)
        return 8;
    reader.close();
    std::filesystem::remove(path);
    std::puts("adjacent exact slices merge; gaps, destination gaps, and oversized ranges stay separate");
    return 0;
}
