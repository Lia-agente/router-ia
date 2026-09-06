// residency_plan.cpp — Host-side residency budget planner for the native
// Qwen3.6-35B-A3B decode runtime (Project: 1 token/s on 3.8GB VRAM + ~5GB RAM).
//
// This is the FIRST native deliverable because "1 tk/s" is fundamentally a
// data-delivery problem, not a flops problem. This planner computes whether
// the model's hot working set actually FITS in the available memory, how it
// should be split across (VRAM resident / RAM hot / SSD cold), and the
// projected warm-decode wall clock. If the budget doesn't close, no kernel
// speeds it up — the bytes physically cannot be delivered in time.
//
// Pure host C++ (no CUDA) so it builds and runs anywhere:
//   g++ -O2 -std=c++17 native_runtime/residency_plan.cpp -o /tmp/resplan && /tmp/resplan
//

#include <cstdint>
#include <cstdio>
#include <algorithm>

namespace {

// ---- Qwen3.6-35B-A3B model constants (from the project README) ----
constexpr uint64_t kHidden      = 2048;
constexpr uint64_t kVocab       = 248320;
constexpr uint32_t kLayers      = 40;
constexpr uint32_t kFullAttnLayers = 10;    // validated full-attention
constexpr uint32_t kDeltaLayers = 30;       // Gated DeltaNet / linear-attn
constexpr uint32_t kExpertsPerLayer = 256;
constexpr uint32_t kTopK        = 8;
constexpr bool    kFp8          = true;     // 1 byte/param checkpoint

// RFC numbers — we derive these from the total param count rather than guess.
constexpr uint64_t kTotalParams = 35'000'000'000ULL;  // ~35B
constexpr uint64_t kB          = 1000ULL;
constexpr uint64_t kMiB        = 1024ULL * 1024ULL;
constexpr uint64_t kGiB        = 1024ULL * kMiB;

constexpr double kBytesPerParam = kFp8 ? 1.0 : 1.0;  // FP8 checkpoint, 1 byte

// Elective constants (user-tunable for the host)
double g_vram_gib   = 3.8;    // ~3.8 GB usable VRAM
double g_ram_gib    = 8.0;    // 8 GB DDR3 (H61 board — single channel)
double g_pcie_gbs   = 4.5;    // PCIe Gen2 x16 (H61) effective H2D bandwidth (~4-5 GB/s)
double g_ssd_gbs    = 0.3;    // SATA-2 SSD on H61 (contemporary boards cap at SATA2)
double g_hdd_gbs    = 0.11;    // SATA-2 HDD worst case (~110 MB/s)

// Quantization policy the user is implementing (automatic Q4 to save RAM/VRAM).
// 1.0 = FP8 (1 byte/param). 0.5 = Q4 (nibble/param). This HALVES the bytes
// that must move, which is the single biggest lever on a Gen2/SATA2 rig.
double g_bytes_per_param = 0.5;

uint64_t params_embeddings() { return kVocab * kHidden; }        // ~508M
uint64_t params_lm_head()    { return kVocab * kHidden; }        // ~508M

// Per-layer fixed (non-routed) weights: attention QKV+O projections + norms.
uint64_t params_attn_per_layer() {
    // 4 projections hidden x hidden = 4 * 2048 * 2048 = 16.8M
    return 4ull * kHidden * kHidden;
}

uint64_t params_fixed_per_layer() {
    return params_attn_per_layer();
}

// Total fixed (attention + embeddings + lm_head + shared experts + norms).
uint64_t params_fixed_total() {
    return kLayers * params_fixed_per_layer()
         + params_embeddings() + params_lm_head();
}

// The remaining params are the routed MoE experts (attention is fixed).
uint64_t params_moe_total() {
    uint64_t f = params_fixed_total();
    return kTotalParams > f ? kTotalParams - f : 0;
}

uint64_t params_shared_total() {
    // 1 shared expert per layer (approx physiological size of a routed expert)
    uint64_t per = params_moe_total() / (kLayers * kExpertsPerLayer);
    return kLayers * per;  // shared experts are the same block size
}

uint64_t bytes_of(uint64_t params) { return (uint64_t)(params * g_bytes_per_param); }

uint64_t moe_expert_bytes() {
    return bytes_of(params_moe_total() / (kLayers * kExpertsPerLayer));
}

void header() {
    std::printf("===========================================================\n"
                " RESIDENCY BUDGET PLANNER — native decode for Qwen3.6-A3B\n"
                " Goal: 1 token/s @ 3.8GB VRAM + ~5GB RAM\n"
                "===========================================================\n\n");
}

uint64_t f2b(double gib) { return (uint64_t)(gib * kGiB); }

} // namespace

int main() {
    header();

    const uint64_t vram_b = f2b(g_vram_gib);
    const uint64_t ram_b  = f2b(g_ram_gib);

    // ---- Derive per-block bytes ----
    const uint64_t emb_b    = bytes_of(params_embeddings());   // ~508MB
    const uint64_t lmhead_b = bytes_of(params_lm_head());      // ~508MB
    const uint64_t attn_per = bytes_of(params_attn_per_layer()); // ~17MB
    const uint64_t moe_ex   = moe_expert_bytes();              // ~3.3MB FP8
    const uint64_t fixed_t  = bytes_of(params_fixed_total());
    const uint64_t moe_t    = bytes_of(params_moe_total());    // ~33.8GB -> never all resident

    std::printf("--- Derived model sizes (FP8) ---\n");
    std::printf(" embeddings (row: only 1 needed per step) : %4.1f MB (%4.1f KB/row)\n",
                emb_b / (double)kMiB, emb_b / (double)kVocab / 1024.0);
    std::printf(" lm_head    (needed EVERY token)          : %4.1f MB\n", lmhead_b / (double)kMiB);
    std::printf(" attention  per layer                     : %4.1f MB x%u\n",
                attn_per / (double)kMiB, kLayers);
    std::printf(" MoE routed expert per block              : %4.1f MB\n", moe_ex / (double)kMiB);
    std::printf(" full fixed total (attn+emb+lmhead, no MoE): %4.1f GB\n",
                fixed_t / (double)kGiB);
    std::printf(" full MoE total (routed experts)          : %4.1f GB  <-- the beast\n\n",
                moe_t / (double)kGiB);

    // ---- Physical feasibility check: can attention + shared + lm_head be resident? ----
    const uint64_t shared_t = bytes_of(params_shared_total());
    const uint64_t must_hot_in_vram = attn_per * kLayers + shared_t + lmhead_b;
    std::printf("--- Residency split ---\n");
    std::printf(" ALWAYS-HOT (VRAM) attention+shared+lm_head: %4.1f GB  (must fit in VRAM %.2f GB)\n",
                must_hot_in_vram / (double)kGiB, g_vram_gib);
    std::printf(" VRAM left for active-expert slots         : %4.1f GB\n",
                (vram_b > must_hot_in_vram ? (vram_b - must_hot_in_vram) / (double)kGiB : 0.0));
    std::printf(" expert slots that fit in that VRAM head   : %4.0f\n",
                vram_b > must_hot_in_vram ? double(vram_b - must_hot_in_vram) / moe_ex : 0.0);
    std::printf(" expert slots that fit in 5GB RAM (hot)    : %4.0f\n",
                ram_b / (double)moe_ex);
    std::printf(" routed-expert loads per single token      : %u (top-%u x %u layers)\n\n",
                kLayers * kTopK, kTopK, kLayers);

    // ---- Per-token data that must arrive from RAM/VRAM (decode) ----
    const uint64_t per_token_moe = kLayers * kTopK * moe_ex;
    std::printf("--- Per-token decode physics ---\n");
    std::printf(" routed-expert bytes touched per token     : %4.2f GB\n",
                per_token_moe / (double)kGiB);
    std::printf("  if 100%% served from RAM @ %.1f GB/s (Gen2): %.0f ms\n",
                g_pcie_gbs, per_token_moe / g_pcie_gbs / 1e6);
    std::printf("  if 50%% cold from SSD @ %.2f GB/s (SATA2)  : ~%.0f ms  <-- danger\n",
                g_ssd_gbs,
                (0.5 * per_token_moe / g_ssd_gbs + 0.5 * per_token_moe / g_pcie_gbs) / 1e6);

    // ---- Quantization lever: FP8 vs Q4 ----
    const double fp8_moe_t = bytes_of(params_moe_total() / 0.5);      // back out FP8 bytes
    const double q4_moe_t  = fp8_moe_t * 0.5;                         // half
    std::printf("\n--- Quantization lever (auto-Q4 the user is building) ---\n");
    std::printf(" full MoE @ FP8 : %4.1f GB   @ Q4 : %4.1f GB  (2x smaller)\n",
                fp8_moe_t / (double)kGiB, q4_moe_t / (double)kGiB);
    std::printf(" bytes/token    : FP8 %.2f GB  |  Q4 %.2f GB\n",
                kLayers * kTopK * fp8_moe_t / (double)(kLayers * kExpertsPerLayer) / (double)kGiB,
                kLayers * kTopK * q4_moe_t  / (double)(kLayers * kExpertsPerLayer) / (double)kGiB);
    std::printf(" RAM-held (8GB) expert slots: FP8 %.0f | Q4 %.0f\n",
                8.0 * kGiB / fp8_moe_t * (kLayers * kExpertsPerLayer),
                8.0 * kGiB / q4_moe_t  * (kLayers * kExpertsPerLayer));

    // ---- Compute side: is 1 token/s flop-bound or bandwidth-bound? ----
    // ~3B active params/token -> ~6 GFLOP; consumers fp8 do ~200-500 GFLOPS.
    const double active_gflop = 6.0;
    const double gpu_gflops   = 250.0;   // conservative fp8 on a 4GB consumer card
    std::printf(" active compute ~%.0f GFLOP/token @ %.0f GFLOPS : ~%.0f ms (not the bottleneck)\n",
                active_gflop, gpu_gflops, active_gflop / gpu_gflops * 1000.0);

    // ---- Verdict ----
    // Q4 halves the per-token bytes, so use the Q4 numbers for the estimate.
    const double q4_moe_ex = q4_moe_t / (double)(kLayers * kExpertsPerLayer);
    const double per_token_q4 = kLayers * kTopK * q4_moe_ex;
    const double estimate_ms =
        must_hot_in_vram <= vram_b
        ? (per_token_q4 / g_pcie_gbs / 1e6) + (active_gflop / gpu_gflops * 1000.0)
        : 1e9;

    std::printf("\n--- Verdict ---\n");
    if (must_hot_in_vram <= vram_b) {
        std::printf(" ALWAYS-HOT set FITS in VRAM (%.1f/%.1f GB).\n",
                    must_hot_in_vram / (double)kGiB, g_vram_gib);
        std::printf(" Projected warm-decode, Q4 experts served from RAM: ~%.0f ms/token.\n\n",
                    estimate_ms);
        std::printf(" CONCLUSION: even on PCIe Gen2 + DDR3 (8GB RAM, 3.8GB VRAM),\n");
        std::printf(" 1 token/s is PHYSICALLY REACHABLE for the warm-decode regime IF:\n");
        std::printf("  (1) attention + shared experts + lm_head stay resident in VRAM\n");
        std::printf("      (%.1f GB of 3.8 GB),\n", must_hot_in_vram / (double)kGiB);
        std::printf("  (2) the 320 active experts/token are served Q4 from RAM (hot)\n");
        std::printf("      via routing-predictor prefetch (never from the SATA2 SSD),\n");
        std::printf("  (3) Python overhead is removed from the per-token hot loop.\n");
        std::printf(" The killer is SSD leakage: at SATA2, even 10%% cold costs ~%.0f ms.\n",
                    (0.1 * per_token_q4 / g_ssd_gbs + 0.9 * per_token_q4 / g_pcie_gbs) / 1e6);
    } else {
        std::printf(" ALWAYS-HOT set does NOT fit (%.1f/%.1f GB) -> need to demote\n",
                    must_hot_in_vram / (double)kGiB, g_vram_gib);
        std::printf(" lm_head to RAM or refit. 1 tk/s requires a smaller lm_head trade\n");
        std::printf(" or an SSD pipeline; otherwise the budget does not close.\n");
    }
    std::printf("===========================================================\n");
    return 0;
}