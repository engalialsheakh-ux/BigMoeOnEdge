#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dstorage.h>
#include <cuda_runtime_api.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;

static void check(HRESULT hr, const char * what) {
    if (FAILED(hr)) throw std::runtime_error(std::string(what) + ": HRESULT=" + std::to_string((uint32_t) hr));
}

static void check(cudaError_t rc, const char * what) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
}

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static uint64_t process_cpu_100ns() {
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user))
        throw std::runtime_error("GetProcessTimes failed");
    ULARGE_INTEGER k{}, u{};
    k.LowPart = kernel.dwLowDateTime;
    k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;
    u.HighPart = user.dwHighDateTime;
    return k.QuadPart + u.QuadPart;
}

static double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    return values[std::min(values.size() - 1, size_t(std::ceil(q * values.size()) - 1))];
}

static std::vector<uint64_t> offsets_from(const char * path, uint64_t max_offset) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open expert offsets file");
    std::vector<uint64_t> offsets;
    uint64_t offset = 0;
    while (in >> offset) {
        offset &= ~uint64_t(4095);
        if (offset <= max_offset) offsets.push_back(offset);
    }
    if (offsets.size() < 32) throw std::runtime_error("need at least 32 valid aligned expert offsets");
    return offsets;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: bmoe_dstorage_probe MODEL_GGUF EXPERT_OFFSETS_TXT\n";
            return 2;
        }
        const std::filesystem::path model_path(argv[1]);
        const uint64_t file_size = std::filesystem::file_size(model_path);
        const auto offsets = offsets_from(argv[2], file_size - 4 * 1024 * 1024);

        int cuda_count = 0;
        check(cudaGetDeviceCount(&cuda_count), "cudaGetDeviceCount");
        if (cuda_count == 0) throw std::runtime_error("no CUDA device");
        ComPtr<IDXGIFactory1> dxgi;
        check(CreateDXGIFactory1(IID_PPV_ARGS(dxgi.GetAddressOf())), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> adapter;
        int cuda_device = -1;
        for (UINT i = 0; dxgi->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            check(adapter->GetDesc1(&desc), "GetDesc1");
            for (int j = 0; j < cuda_count; ++j) {
                cudaDeviceProp prop{};
                check(cudaGetDeviceProperties(&prop, j), "cudaGetDeviceProperties");
                if (std::memcmp(&desc.AdapterLuid, prop.luid, sizeof(LUID)) == 0) {
                    cuda_device = j;
                    break;
                }
            }
            if (cuda_device >= 0) break;
        }
        if (cuda_device < 0) throw std::runtime_error("D3D12/CUDA adapter LUID mismatch");
        check(cudaSetDevice(cuda_device), "cudaSetDevice");
        ComPtr<ID3D12Device> device;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.GetAddressOf())),
              "D3D12CreateDevice");

        ComPtr<IDStorageFactory> factory;
        check(DStorageGetFactory(IID_PPV_ARGS(factory.GetAddressOf())), "DStorageGetFactory");
        ComPtr<IDStorageFile> file;
        check(factory->OpenFile(model_path.c_str(), IID_PPV_ARGS(file.GetAddressOf())), "DirectStorage OpenFile");
        DSTORAGE_QUEUE_DESC qdesc{};
        qdesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        qdesc.Capacity = 128;
        qdesc.Priority = DSTORAGE_PRIORITY_NORMAL;
        qdesc.Device = device.Get();
        ComPtr<IDStorageQueue> queue;
        check(factory->CreateQueue(&qdesc, IID_PPV_ARGS(queue.GetAddressOf())), "DirectStorage CreateQueue");

        const UINT64 capacity = 32 * 1024 * 1024;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC resource{};
        resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resource.Width = capacity;
        resource.Height = 1;
        resource.DepthOrArraySize = 1;
        resource.MipLevels = 1;
        resource.Format = DXGI_FORMAT_UNKNOWN;
        resource.SampleDesc.Count = 1;
        resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> buffer;
        check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &resource, D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, IID_PPV_ARGS(buffer.GetAddressOf())),
              "CreateCommittedResource");
        HANDLE resource_handle = nullptr;
        check(device->CreateSharedHandle(buffer.Get(), nullptr, GENERIC_ALL, nullptr, &resource_handle),
              "CreateSharedHandle(buffer)");
        cudaExternalMemoryHandleDesc memory_desc{};
        memory_desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        memory_desc.handle.win32.handle = resource_handle;
        memory_desc.size = capacity;
        memory_desc.flags = cudaExternalMemoryDedicated;
        cudaExternalMemory_t external_memory = nullptr;
        check(cudaImportExternalMemory(&external_memory, &memory_desc), "cudaImportExternalMemory");
        CloseHandle(resource_handle);
        cudaExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.size = capacity;
        void * cuda_shared = nullptr;
        check(cudaExternalMemoryGetMappedBuffer(&cuda_shared, external_memory, &buffer_desc),
              "cudaExternalMemoryGetMappedBuffer");

        ComPtr<ID3D12Fence> fence;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence.GetAddressOf())), "CreateFence");
        HANDLE fence_handle = nullptr;
        check(device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &fence_handle),
              "CreateSharedHandle(fence)");
        cudaExternalSemaphoreHandleDesc semaphore_desc{};
        semaphore_desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        semaphore_desc.handle.win32.handle = fence_handle;
        cudaExternalSemaphore_t external_semaphore = nullptr;
        check(cudaImportExternalSemaphore(&external_semaphore, &semaphore_desc), "cudaImportExternalSemaphore");
        CloseHandle(fence_handle);
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("CreateEvent failed");
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        void * cuda_baseline = nullptr;
        check(cudaMalloc(&cuda_baseline, capacity), "cudaMalloc baseline");
        void * host = _aligned_malloc(capacity, 4096);
        if (!host) throw std::runtime_error("aligned host allocation failed");
        HANDLE win_file = CreateFileW(model_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                      FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (win_file == INVALID_HANDLE_VALUE) throw std::runtime_error("CreateFileW direct failed");
        HANDLE async_file = CreateFileW(model_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (async_file == INVALID_HANDLE_VALUE) throw std::runtime_error("CreateFileW overlapped failed");
        HANDLE read_events[8]{};
        for (HANDLE & read_event : read_events) {
            read_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!read_event) throw std::runtime_error("CreateEventW for overlapped read failed");
        }

        std::cout << "size_kib,path,p50_ms,p95_ms,throughput_mib_s,submit_wall_ms,ready_wait_ms,process_cpu_ms,n\n";
        UINT64 fence_value = 0;
        for (size_t size : {size_t(512) * 1024, size_t(1024) * 1024, size_t(2048) * 1024, size_t(4096) * 1024}) {
            for (int arm = 0; arm < 2; ++arm) {
                std::vector<double> elapsed, submit, wait, process_cpu;
                constexpr size_t n = 48;
                for (size_t i = 0; i < n; ++i) {
                    const uint64_t offset = offsets[(i * 7919 + size / 4096) % offsets.size()];
                    const auto t0 = Clock::now();
                    const uint64_t cpu0 = process_cpu_100ns();
                    if (arm == 0) {
                        LARGE_INTEGER position{};
                        position.QuadPart = offset;
                        if (!SetFilePointerEx(win_file, position, nullptr, FILE_BEGIN))
                            throw std::runtime_error("SetFilePointerEx failed");
                        DWORD got = 0;
                        if (!ReadFile(win_file, host, DWORD(size), &got, nullptr) || got != size)
                            throw std::runtime_error("ReadFile direct failed");
                        const auto t_submit = Clock::now();
                        check(cudaMemcpyAsync(cuda_baseline, host, size, cudaMemcpyHostToDevice, stream),
                              "cudaMemcpyAsync");
                        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize baseline");
                        const auto t_ready = Clock::now();
                        submit.push_back(ms(t0, t_submit));
                        wait.push_back(ms(t_submit, t_ready));
                        elapsed.push_back(ms(t0, t_ready));
                        process_cpu.push_back(double(process_cpu_100ns() - cpu0) / 10000.0);
                    } else {
                        DSTORAGE_REQUEST request{};
                        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                        request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                        request.Source.File.Source = file.Get();
                        request.Source.File.Offset = offset;
                        request.Source.File.Size = UINT32(size);
                        request.Destination.Buffer.Resource = buffer.Get();
                        request.Destination.Buffer.Offset = 0;
                        request.Destination.Buffer.Size = UINT32(size);
                        ++fence_value;
                        queue->EnqueueRequest(&request);
                        queue->EnqueueSignal(fence.Get(), fence_value);
                        queue->Submit();
                        const auto t_submit = Clock::now();
                        check(fence->SetEventOnCompletion(fence_value, event), "SetEventOnCompletion");
                        if (WaitForSingleObject(event, 10000) != WAIT_OBJECT_0)
                            throw std::runtime_error("DirectStorage fence timeout");
                        cudaExternalSemaphoreWaitParams wait_params{};
                        wait_params.params.fence.value = fence_value;
                        check(cudaWaitExternalSemaphoresAsync(&external_semaphore, &wait_params, 1, stream),
                              "cudaWaitExternalSemaphoresAsync");
                        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize DirectStorage");
                        const auto t_ready = Clock::now();
                        submit.push_back(ms(t0, t_submit));
                        wait.push_back(ms(t_submit, t_ready));
                        elapsed.push_back(ms(t0, t_ready));
                        process_cpu.push_back(double(process_cpu_100ns() - cpu0) / 10000.0);
                        if (WaitForSingleObject(queue->GetErrorEvent(), 0) == WAIT_OBJECT_0)
                            throw std::runtime_error("DirectStorage queue reported an error");
                        if (i == 0) {
                            LARGE_INTEGER position{};
                            position.QuadPart = offset;
                            if (!SetFilePointerEx(win_file, position, nullptr, FILE_BEGIN))
                                throw std::runtime_error("verification seek failed");
                            DWORD got = 0;
                            if (!ReadFile(win_file, host, DWORD(size), &got, nullptr) || got != size)
                                throw std::runtime_error("verification read failed");
                            std::vector<uint8_t> sample(size);
                            check(cudaMemcpy(sample.data(), cuda_shared, size, cudaMemcpyDeviceToHost),
                                  "verification CUDA copy");
                            if (std::memcmp(sample.data(), host, size) != 0)
                                throw std::runtime_error("DirectStorage/CUDA bytes differ from GGUF source");
                        }
                    }
                }
                const double mean = std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();
                const double mean_submit = std::accumulate(submit.begin(), submit.end(), 0.0) / submit.size();
                const double mean_wait = std::accumulate(wait.begin(), wait.end(), 0.0) / wait.size();
                const double mean_cpu =
                    std::accumulate(process_cpu.begin(), process_cpu.end(), 0.0) / process_cpu.size();
                std::cout << size / 1024 << ',' << (arm == 0 ? "direct_read_cuda_upload" : "directstorage_cuda_shared")
                          << ',' << quantile(elapsed, 0.5) << ',' << quantile(elapsed, 0.95) << ','
                          << double(size) / 1048576.0 / (mean / 1000.0) << ',' << mean_submit << ',' << mean_wait << ','
                          << mean_cpu << ',' << n << '\n';
            }
        }
        for (size_t size : {size_t(512) * 1024, size_t(1024) * 1024, size_t(2048) * 1024, size_t(4096) * 1024}) {
            for (int arm = 0; arm < 2; ++arm) {
                std::vector<double> elapsed, submit, wait, process_cpu;
                constexpr size_t requests_per_batch = 8;
                constexpr size_t batches = 16;
                for (size_t batch = 0; batch < batches; ++batch) {
                    const auto t0 = Clock::now();
                    const uint64_t cpu0 = process_cpu_100ns();
                    OVERLAPPED pending[requests_per_batch]{};
                    for (size_t i = 0; i < requests_per_batch; ++i) {
                        const uint64_t offset =
                            offsets[((batch * requests_per_batch + i) * 7919 + size / 4096) % offsets.size()];
                        if (arm == 0) {
                            ResetEvent(read_events[i]);
                            pending[i].Offset = DWORD(offset);
                            pending[i].OffsetHigh = DWORD(offset >> 32);
                            pending[i].hEvent = read_events[i];
                            if (!ReadFile(async_file, static_cast<uint8_t *>(host) + i * size, DWORD(size), nullptr,
                                          &pending[i]) &&
                                GetLastError() != ERROR_IO_PENDING)
                                throw std::runtime_error("queued overlapped read submission failed");
                        } else {
                            DSTORAGE_REQUEST request{};
                            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                            request.Source.File.Source = file.Get();
                            request.Source.File.Offset = offset;
                            request.Source.File.Size = UINT32(size);
                            request.Destination.Buffer.Resource = buffer.Get();
                            request.Destination.Buffer.Offset = i * size;
                            request.Destination.Buffer.Size = UINT32(size);
                            queue->EnqueueRequest(&request);
                        }
                    }
                    if (arm == 1) {
                        ++fence_value;
                        queue->EnqueueSignal(fence.Get(), fence_value);
                        queue->Submit();
                    }
                    const auto t_submit = Clock::now();
                    if (arm == 0) {
                        if (WaitForMultipleObjects(DWORD(requests_per_batch), read_events, TRUE, 10000) !=
                            WAIT_OBJECT_0)
                            throw std::runtime_error("queued overlapped read timeout");
                        for (size_t i = 0; i < requests_per_batch; ++i) {
                            DWORD got = 0;
                            if (!GetOverlappedResult(async_file, &pending[i], &got, FALSE) || got != size)
                                throw std::runtime_error("queued overlapped read failed");
                            check(cudaMemcpyAsync(static_cast<uint8_t *>(cuda_baseline) + i * size,
                                                  static_cast<uint8_t *>(host) + i * size, size, cudaMemcpyHostToDevice,
                                                  stream),
                                  "queued baseline H2D");
                        }
                        check(cudaStreamSynchronize(stream), "queued baseline synchronize");
                    } else {
                        check(fence->SetEventOnCompletion(fence_value, event), "queued SetEventOnCompletion");
                        if (WaitForSingleObject(event, 10000) != WAIT_OBJECT_0)
                            throw std::runtime_error("queued DirectStorage fence timeout");
                        cudaExternalSemaphoreWaitParams wait_params{};
                        wait_params.params.fence.value = fence_value;
                        check(cudaWaitExternalSemaphoresAsync(&external_semaphore, &wait_params, 1, stream),
                              "queued CUDA fence wait");
                        check(cudaStreamSynchronize(stream), "queued CUDA synchronize");
                        if (WaitForSingleObject(queue->GetErrorEvent(), 0) == WAIT_OBJECT_0)
                            throw std::runtime_error("queued DirectStorage error");
                    }
                    const auto t_ready = Clock::now();
                    submit.push_back(ms(t0, t_submit));
                    wait.push_back(ms(t_submit, t_ready));
                    elapsed.push_back(ms(t0, t_ready));
                    process_cpu.push_back(double(process_cpu_100ns() - cpu0) / 10000.0);
                }
                const double mean = std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();
                const double mean_submit = std::accumulate(submit.begin(), submit.end(), 0.0) / submit.size();
                const double mean_wait = std::accumulate(wait.begin(), wait.end(), 0.0) / wait.size();
                const double mean_cpu =
                    std::accumulate(process_cpu.begin(), process_cpu.end(), 0.0) / process_cpu.size();
                std::cout << size / 1024 << ','
                          << (arm == 0 ? "direct_read_cuda_upload_queued8" : "directstorage_cuda_shared_queued8") << ','
                          << quantile(elapsed, 0.5) << ',' << quantile(elapsed, 0.95) << ','
                          << requests_per_batch * double(size) / 1048576.0 / (mean / 1000.0) << ',' << mean_submit
                          << ',' << mean_wait << ',' << mean_cpu << ',' << batches << '\n';
            }
        }
        CloseHandle(win_file);
        CloseHandle(async_file);
        for (HANDLE read_event : read_events)
            CloseHandle(read_event);
        _aligned_free(host);
        cudaFree(cuda_baseline);
        cudaStreamDestroy(stream);
        CloseHandle(event);
        cudaDestroyExternalSemaphore(external_semaphore);
        cudaFree(cuda_shared);
        cudaDestroyExternalMemory(external_memory);
        queue->Close();
        file->Close();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "probe failed: " << error.what() << '\n';
        return 1;
    }
}
