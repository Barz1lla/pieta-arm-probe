/* i8mm correctness test — the V2 stretch path. Compiled only on silicon that advertises i8mm (Armv8.6).
 * If __ARM_FEATURE_MATMUL_INT8 is not defined the build fails here, by design: a silent scalar fallback would turn a
 * "the feature is absent" result into a false pass.
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>

#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "i8mm not enabled: this CPU or this -march does not support the Armv8.6 int8 matmul extension"
#endif

static int32_t scalar_ref(const int8_t *a, const int8_t *b, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

int main(void) {
    /* 2x8 A tile against an 8x2 B panel — the shape the KV kernel uses for asymmetric dequantise-and-dot. */
    int8_t A[2][8], B[8][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 8; j++) A[i][j] = (int8_t)((i * 5 + j * 3) % 31 - 15);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 2; j++) B[i][j] = (int8_t)((i * 7 + j * 11) % 23 - 11);

    int32_t ref[2][2] = {{0, 0}, {0, 0}};
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++)
            for (int k = 0; k < 8; k++) ref[r][c] += (int32_t)A[r][k] * (int32_t)B[k][c];

    int8x16_t va = vcombine_s8(vld1_s8(A[0]), vld1_s8(A[1]));
    int8x16_t vb = vcombine_s8(vld1_s8(B[0]), vld1_s8(B[1]));
    int32x4_t acc = vdupq_n_s32(0);
    acc = vmmlaq_s32(acc, va, vb);

    int32_t got[4];
    vst1q_s32(got, acc);
    int32_t refv[4] = {ref[0][0], ref[0][1], ref[1][0], ref[1][1]};

    int ok = 1;
    for (int i = 0; i < 4; i++) if (got[i] != refv[i]) ok = 0;

    printf("ref=[%d %d %d %d] i8mm=[%d %d %d %d] %s\n",
           refv[0], refv[1], refv[2], refv[3], got[0], got[1], got[2], got[3], ok ? "PASS" : "FAIL");
    printf("features: i8mm=1 dotprod=%d\n",
#if defined(__ARM_FEATURE_DOTPROD)
           1
#else
           0
#endif
    );
    return ok ? 0 : 1;
}
