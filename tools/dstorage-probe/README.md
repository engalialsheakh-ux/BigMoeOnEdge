# Isolated DirectStorage/CUDA probe

This diagnostic is not linked into BigMoeOnEdge. It compares aligned unbuffered
Windows reads followed by CUDA upload with DirectStorage reads into a shared
D3D12 buffer imported into CUDA. It uses 512 KiB, 1 MiB, 2 MiB, and 4 MiB
requests at real routed expert offsets, both singly and in batches of eight.
The source bytes are verified for the first DirectStorage request at each size.

Install Microsoft's DirectStorage SDK NuGet package separately (tested with
`Microsoft.Direct3D.DirectStorage` 1.3.0), then configure this isolated folder:

```powershell
cmake -S tools/dstorage-probe -B build-cuda/dstorage-probe -A x64 `
  -DDSTORAGE_NATIVE='D:/path/to/DirectStorage/package/native'
cmake --build build-cuda/dstorage-probe --config Release
py tools/dstorage-probe/make_offsets.py TRACE.access build-cuda/dstorage-probe/offsets.txt
build-cuda/dstorage-probe/Release/bmoe_dstorage_probe.exe MODEL.gguf build-cuda/dstorage-probe/offsets.txt
```

The trace's GGUF projection offsets require a single-file GGUF. The probe
compares two *GPU destination* paths; the production expert cache currently
loads SSD misses for CPU work and promotes to GPU only from RAM. The benchmark
therefore cannot justify a production DirectStorage integration by itself.
The baseline's eight-request mode uses overlapped Windows I/O and eight
in-flight reads, which is an optimistic comparison against the production
four-lane reader. `process_cpu_ms` is process CPU time sampled by Windows and
is coarse for sub-millisecond single requests.
