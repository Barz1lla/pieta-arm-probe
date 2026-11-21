/* i8mm correctness test — V2 stretch path.
 * Correction 2026-11-21: the first version modelled the reference as a 2x2 GEMM and FAILED on silicon. The fault was the
 * reference, not the CPU: vmmlaq_s32(int32x4 acc, int8x16 a, int8x16 b) accumulates FOUR independent 4-byte dot products
 * into the four lanes — acc[i] += sum(a[4i..4i+3] * b[4i..4i+3]). The reference below matches the instruction.
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>

#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "i8mm not enabled: this CPU or this -march does not support the Armv8.6 int8 matmul extension"
#endif

int main(void) {
    int8_t a[16], b[16];
    for (int i = 0; i < 16; i++) {
        a[i] = (int8_t)((i * 5 + 3) % 31 - 15);
        b[i] = (int8_t)((i * 7 + 1) % 23 - 11);
    }

    /* reference: four lanes, each a 4-wide dot product */
    int32_t ref[4] = {0, 0, 0, 0};
    for (int lane = 0; lane < 4; lane++)
        for (int k = 0; k < 4; k++)
            ref[lane] += (int32_t)a[lane * 4 + k] * (int32_t)b[lane * 4 + k];

    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
    int32x4_t acc = vdupq_n_s32(0);
    acc = vmmlaq_s32(acc, va, vb);

    int32_t got[4];
    vst1q_s32(got, acc);

    int ok = 1;
    for (int i = 0; i < 4; i++) if (got[i] != ref[i]) ok = 0;

    printf("ref=[%d %d %d %d] i8mm=[%d %d %d %d] %s\n",
           ref[0], ref[1], ref[2], ref[3], got[0], got[1], got[2], got[3], ok ? "PASS" : "FAIL");
    printf("features: i8mm=1 dotprod=%d\n",
#if defined(__ARM_FEATURE_DOTPROD)
           1
#else
           0
#endif
    );
    return ok ? 0 : 1;
}
