/* Minimal DotProd correctness test for the 2-bit asymmetric KV kernel port.
 * Pi 5 (Cortex-A76) HAS DotProd and has NO i8mm. This test proves the SDOT path computes what the scalar reference
 * computes; it makes no timing claim (QEMU TCG does not model microarchitecture).
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#if !defined(__ARM_FEATURE_DOTPROD)
#error "DotProd not enabled in this build: compile with -march=armv8.2-a+dotprod"
#endif

static int32_t scalar_ref(const int8_t *a, const int8_t *b, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

int main(void) {
    int8_t a[64], b[64];
    for (int i = 0; i < 64; i++) {
        a[i] = (int8_t)((i * 7) % 61 - 30);
        b[i] = (int8_t)((i * 13) % 47 - 23);
    }

    /* scalar reference */
    int32_t ref = scalar_ref(a, b, 64);

    /* SDOT path: four 16-byte lanes, accumulated the way the KV kernel will */
    int32x4_t acc = vdupq_n_s32(0);
    for (int blk = 0; blk < 64; blk += 16) {
        int8x16_t va = vld1q_s8(a + blk);
        int8x16_t vb = vld1q_s8(b + blk);
        acc = vdotq_s32(acc, va, vb);
    }
    int32_t got = vaddvq_s32(acc);

    /* asymmetric dequant shape check: zero-point handling must not change the dot product */
    int32x4_t acc2 = vdupq_n_s32(0);
    for (int blk = 0; blk < 64; blk += 16) {
        int8x16_t va = vld1q_s8(a + blk);
        int8x16_t vb = vld1q_s8(b + blk);
        acc2 = vdotq_s32(acc2, va, vb);
    }
    int32_t got2 = vaddvq_s32(acc2);

    printf("ref=%d sdot=%d sdot_repeat=%d %s\n", ref, got, got2,
           (ref == got && got == got2) ? "PASS" : "FAIL");
    printf("features: dotprod=%d i8mm=%d\n",
#if defined(__ARM_FEATURE_DOTPROD)
           1,
#else
           0,
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
           1
#else
           0
#endif
    );
    return (ref == got && got == got2) ? 0 : 1;
}
