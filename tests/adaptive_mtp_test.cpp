#include "bmoe/adaptive_mtp.h"

int main() {
    bmoe::AdaptiveMtpWidth c;
    if (c.request(4, 10) != 2) return 1;
    for (int i = 0; i < 3; ++i)
        c.observe(2, 2, 2, 4);
    if (c.request(4, 10) != 3) return 2;
    c.observe(3, 3, 0, 4);
    if (c.request(4, 10) != 1) return 3;
    for (int i = 0; i < 9; ++i) {
        const int width = c.request(4, 10);
        c.observe(width, width, width, 4);
    }
    if (c.width != 4) return 4;
    if (c.request(4, 2) != 2) return 5;
    c.observe(2, 0, 0, 4); // confidence floor can abstain without another decision pass
    if (c.width != 3) return 6;
    if (c.max_requested > 4) return 7;

    bmoe::AdaptiveMtpWidth expensive;
    expensive.width = 2;
    for (int i = 0; i < 2; ++i) {
        expensive.observe(1, 1, 1, 4, {10.0, 0.0, 8.0, 0.0, 0.0});
        expensive.observe(2, 2, 2, 4, {50.0, 30.0, 22.0, 14.0, 16.0});
    }
    if (expensive.request(4, 10) != 1 || expensive.last_proposed != 2 || expensive.cost_truncations != 1) return 8;

    bmoe::AdaptiveMtpWidth useful;
    useful.width = 2;
    for (int i = 0; i < 2; ++i) {
        useful.observe(1, 1, 1, 4, {20.0, 5.0, 8.0, 2.0, 0.0});
        useful.observe(2, 2, 2, 4, {24.0, 5.0, 12.0, 3.0, 0.0});
    }
    if (useful.request(4, 10) != 2) return 9;

    // Drafting and accepted-prefix catch-up can reverse the choice made from target cost alone.
    bmoe::AdaptiveMtpWidth full_cost;
    full_cost.request(2, 10, true);
    for (int i = 0; i < 2; ++i) {
        full_cost.observe(1, 1, 1, 2, {20.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        full_cost.observe(2, 2, 2, 2, {24.0, 0.0, 0.0, 0.0, 0.0, 20.0});
    }
    if (full_cost.request(2, 10, true) != 1) return 10;
    if (full_cost.request(2, 10) != 2) return 11; // measured production policy remains the default

    const auto seed_cold = [](bmoe::AdaptiveMtpWidth & controller) {
        controller.request(2, 10, true);
        for (int i = 0; i < 2; ++i) {
            controller.observe(1, 1, 1, 2, {10.0, 0.0, 0.0, 0.0, 0.0, 1.0});
            controller.observe(2, 2, 2, 2, {80.0, 0.0, 0.0, 0.0, 0.0, 2.0});
        }
    };

    // Expensive wider batches get bounded refreshes, not repeated speculative sweeps.
    bmoe::AdaptiveMtpWidth bounded;
    seed_cold(bounded);
    int previous_probe = -bmoe::AdaptiveMtpWidth::probe_interval;
    int narrow_requests = 0;
    for (int i = 0; i < 40; ++i) {
        const int probes = bounded.cost_probes;
        const int requested = bounded.request(2, 10, true);
        if (requested < 1 || requested > 2) return 12;
        if (bounded.cost_probes != probes) {
            if (i - previous_probe < bmoe::AdaptiveMtpWidth::probe_interval) return 13;
            previous_probe = i;
        } else if (requested != 1) {
            return 14;
        }
        narrow_requests += requested == 1;
        bounded.observe(requested, requested, requested, 2,
                        {requested == 1 ? 10.0 : 80.0, 0.0, 0.0, 0.0, 0.0, double(requested)});
    }
    if (bounded.cost_probes < 3 || narrow_requests < 35) return 15;

    // Once warming makes the wider batch cheaper, stale cold cost must expire and widening recover.
    bmoe::AdaptiveMtpWidth warmed;
    seed_cold(warmed);
    for (int i = 0; i < 30; ++i) {
        const int requested = warmed.request(3, 10, true);
        warmed.observe(requested, requested, requested, 3,
                       {10.0 + 2.0 * (requested - 1), 0.0, 0.0, 0.0, 0.0, double(requested)});
    }
    if (warmed.cost_probes == 0 || warmed.width != 3 || warmed.max_requested != 3) return 16;

    // Narrow successful passes must not erase the evidence collected by wider probes.
    bmoe::AdaptiveMtpWidth streak;
    streak.request(4, 10, true);
    streak.observe(2, 2, 2, 4);
    for (int i = 0; i < 7; ++i)
        streak.observe(1, 1, 1, 4);
    if (streak.width != 2 || streak.streak != 1) return 17;
    streak.observe(2, 2, 2, 4);
    streak.observe(1, 1, 1, 4);
    streak.observe(2, 2, 2, 4);
    if (streak.width != 3) return 18;
    streak.observe(3, 2, 2, 4); // confidence-limited drafts cannot justify the wider request
    if (streak.streak != 0) return 19;
    streak.observe(3, 3, 0, 4);
    if (streak.width != 1 || streak.request(4, 10, true) != 1) return 20;
    if (warmed.request(3, 1, true) != 1 || warmed.request(3, 0, true) != 0) return 21;
}
