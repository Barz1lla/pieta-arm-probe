/* VMMLA layout probe — characterise the instruction from the hardware instead of guessing its operand layout.
 * Three probes; the printed vectors determine the mapping. No PASS/FAIL yet: this run exists to establish the semantics.
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>

#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "i8mm not enabled"
#endif

static void run(const char *label, const int8_t *a, const int8_t *b) {
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
    int32x4_t acc = vdupq_n_s32(0);
    acc = vmmlaq_s32(acc, va, vb);
    int32_t got[4];
    vst1q_s32(got, acc);
    printf("%s -> [%d %d %d %d]\n", label, got[0], got[1], got[2], got[3]);
}

int main(void) {
    int8_t one[16], zero[16], seq[16], e0[16], e8[16];
    for (int i = 0; i < 16; i++) { one[i] = 1; zero[i] = 0; seq[i] = (int8_t)(i + 1); e0[i] = 0; e8[i] = 0; }
    e0[0] = 1;
    e8[8] = 1;

    run("P1 a=1s b=1s      ", one, one);      /* lane value = contraction length */
    run("P2 a=e0 b=1..16   ", e0, seq);       /* which b indices pair with a[0] */
    run("P3 a=e8 b=1s      ", e8, one);       /* which lanes receive a[8] */
    run("P4 a=1s b=1..16   ", one, seq);      /* full b weight per lane */
    return 0;
}
