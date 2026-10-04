#include "bmoe/route_trace.h"
#include "bmoe/runtime.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Trace final : bmoe::IRouteTraceSink {
    std::vector<bmoe::RouteTraceRow> rows;
    void on_static(const bmoe::RouteTraceStatic &) override {}
    void on_rows(const bmoe::RouteTraceRow * first, size_t count) override {
        rows.insert(rows.end(), first, first + count);
    }
};

using Route = std::tuple<int, int, int, int, int>;

std::vector<Route> routes(const Trace & trace) {
    std::vector<Route> out;
    out.reserve(trace.rows.size());
    for (const auto & r : trace.rows)
        out.emplace_back(r.phase, r.step, r.layer, r.slot, r.expert);
    return out;
}

bool run_case(const bmoe::RunConfig & cfg, bmoe::RunResult & result, Trace & trace) {
    result = bmoe::run(cfg, nullptr, nullptr, &trace);
    if (result) return true;
    std::fprintf(stderr, "prefill cache gate run failed: %s\n", result.error.c_str());
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    bmoe::RunConfig baseline;
    baseline.model_path = argv[1];
    baseline.prompt = std::string(96, 'a');
    baseline.n_ctx = 256;
    baseline.n_ubatch = 8;
    baseline.n_predict = 4;
    baseline.n_threads = 2;
    baseline.moe.enabled = true;
    baseline.moe.io_threads = 2;

    Trace old_trace, serial_trace, overlap_trace, two_wave_trace;
    bmoe::RunResult old_result, serial_result, overlap_result, two_wave_result;
    if (!run_case(baseline, old_result, old_trace)) return 1;

    bmoe::RunConfig serial = baseline;
    serial.moe.cache_mb = 2;
    serial.moe.force_cache = true;
    if (!run_case(serial, serial_result, serial_trace)) return 1;

    bmoe::RunConfig overlap = serial;
    overlap.moe.overlap = true;
    overlap.moe.io_coalesce = true;
    overlap.moe.access_trace_path = overlap.model_path + ".access";
    if (!run_case(overlap, overlap_result, overlap_trace)) return 1;
    bmoe::RunConfig two_wave = overlap;
    two_wave.moe.io_two_wave = true;
    two_wave.moe.access_trace_path.clear();
    if (!run_case(two_wave, two_wave_result, two_wave_trace)) return 1;

    const auto & s = serial_result.summary;
    if (old_result.generated_text != serial_result.generated_text ||
        old_result.generated_text != overlap_result.generated_text ||
        old_result.generated_text != two_wave_result.generated_text || routes(old_trace) != routes(serial_trace) ||
        routes(old_trace) != routes(overlap_trace) || routes(old_trace) != routes(two_wave_trace)) {
        std::fprintf(stderr, "prefill cache changed output or routing\n");
        return 1;
    }
    int earlier_layer_hits = 0;
    for (const auto & row : serial_trace.rows)
        if (row.phase == 0 && row.layer == 0 && row.step >= 8 && row.residency != bmoe::route_miss)
            ++earlier_layer_hits;
    if (earlier_layer_hits == 0 || s.prefill_retained_hits == 0 || s.prefill_cache_lookups == 0 ||
        s.prefill_transient_loads == 0 || s.prefill_cache_peak_mib <= 0 || s.prefill_retained_resident_mib <= 0 ||
        overlap_result.summary.prefill_retained_hits == 0 || std::fabs(s.cache_budget_mib - 2.0) > 0.01 ||
        s.prefill_cache_peak_mib > 2.01 || overlap_result.summary.prefill_cache_peak_mib > 2.01 ||
        s.cache_resident_mib > 2.01 || overlap_result.summary.cache_resident_mib > 2.01 ||
        s.prefill_protected_after_generation_mib != 0.0 ||
        overlap_result.summary.prefill_protected_after_generation_mib != 0.0 ||
        two_wave_result.summary.prefill_cache_peak_mib > 2.01) {
        std::fprintf(stderr, "retention/budget gate failed: layer0=%d retained=%lld/%lld resident=%.2f/%.2f MiB\n",
                     earlier_layer_hits, s.prefill_retained_hits, overlap_result.summary.prefill_retained_hits,
                     s.cache_resident_mib, s.cache_budget_mib);
        return 1;
    }
    std::printf("prefill retention: layer0 hits=%d serial=%lld overlap=%lld transient=%lld peak=%.2f MiB; "
                "outputs and routes exact\n",
                earlier_layer_hits, s.prefill_retained_hits, overlap_result.summary.prefill_retained_hits,
                s.prefill_transient_loads, s.prefill_cache_peak_mib);
    return 0;
}
