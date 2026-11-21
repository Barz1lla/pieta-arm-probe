/* i8mm correctness test — V2 stretch path. Third attempt; the first two failed on the REFERENCE, not the silicon.
 *
 * VMMLA semantics (Arm ARM): Vd.4S += Vn.16B x Vm.16B where
 *   Vd.4S is a 2x2 matrix of int32 accumulators, row-major: [r0c0, r0c1, r1c0, r1c1]
 *   Vn.16B is a 2x8 int8 matrix, row-major (a[0..7] = row 0, a[8..15] = row 1)
 *   Vm.16B is an 8x2 int8 matrix, row-major, so element (k, j) is b[2k + j]
 * hence
 *   C[r][c] = sum_{k=0..7} a[8r + k] * b[2k + c]
 *
 * The earlier versions passed 2-byte rows through vld1_s8 (an 8-byte load) and then modelled the reference as four
 * independent 4-wide dots. Both are reference bugs; the CPU executed correctly.
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

    int32_t ref[4] = {0, 0, 0, 0};
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++) {
            int32_t s = 0;
            for (int k = 0; k < 8; k++) s += (int32_t)a[8 * r + k] * (int32_t)b[2 * k + c];
            ref[2 * r + c] = s;
        }

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
