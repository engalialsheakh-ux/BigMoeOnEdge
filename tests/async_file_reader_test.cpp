#include "file_reader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

int main() {
#if !defined(_WIN32)
    return 77;
#else
    const auto path = std::filesystem::temp_directory_path() / "bmoe-async-reader-test.bin";
    std::vector<uint8_t> source(8 * 1024 * 1024);
    for (size_t i = 0; i < source.size(); ++i)
        source[i] = uint8_t((i * 131u + i / 97u) & 255u);
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(source.data()), std::streamsize(source.size()));
        if (!out) return 1;
    }

    bmoe::FileReader reader;
    if (!reader.open(path.string(), 4, false, 4096, 0, true) || !reader.asynchronous()) return 2;
    std::atomic<bool> valid{true};
    std::array<std::thread, 4> workers;
    for (int lane = 0; lane < 4; ++lane) {
        workers[size_t(lane)] = std::thread([&, lane] {
            for (int iteration = 0; iteration < 3; ++iteration) {
                const size_t off = size_t(lane) * 1024 * 1024 + size_t(iteration) * 197;
                std::vector<uint8_t> got(512 * 1024 + size_t(iteration));
                if (reader.read(lane, got.data(), off, got.size()) != static_cast<long long>(got.size()) ||
                    !std::equal(got.begin(), got.end(), source.begin() + ptrdiff_t(off)))
                    valid.store(false);
            }
        });
    }
    for (auto & worker : workers)
        worker.join();
    if (!valid || reader.async_completions() < 12 || reader.async_high_water() < 1) return 3;
    if (!reader.reopen() || !reader.asynchronous()) return 4;
    std::array<uint8_t, 257> tail{};
    if (reader.read(0, tail.data(), source.size() - tail.size(), tail.size()) != static_cast<long long>(tail.size()) ||
        !std::equal(tail.begin(), tail.end(), source.end() - ptrdiff_t(tail.size())))
        return 5;
    reader.close();
    std::filesystem::remove(path);
    return 0;
#endif
}
