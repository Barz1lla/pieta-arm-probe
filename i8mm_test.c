/* i8mm correctness test — V2 stretch path. Reference derived from probes ON THE HARDWARE, not from recollection.
 *
 * Probe results (Azure Cobalt-class Neoverse, i8mm=1):
 *   P1 a=1s b=1s     -> [8 8 8 8]        each lane contracts 8 terms
 *   P2 a=e0 b=1..16  -> [1 9 0 0]        a[0] pairs with b[0] in lane 0 and b[8] in lane 1
 *   P3 a=e8 b=1s     -> [0 0 1 1]        a[8] feeds lanes 2 and 3
 *   P4 a=1s b=1..16  -> [36 100 36 100]  lane0 = sum b[0..7], lane1 = sum b[8..15]
 *
 * Hence, with A = 2x8 row-major (a[8r+k]) and B = 8x2 stored as two contiguous 8-byte columns (b[8c+k]):
 *   lane (r,c) = sum_{k=0..7} a[8r + k] * b[8c + k]
 * which is a 2x2 matrix product, A x B, in column-major B.
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
            for (int k = 0; k < 8; k++) s += (int32_t)a[8 * r + k] * (int32_t)b[8 * c + k];
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
