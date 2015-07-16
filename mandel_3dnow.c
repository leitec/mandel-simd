#include <xmmintrin.h>
#include "mandel.h"

typedef float v2sf __attribute__ ((vector_size(8)));
typedef int v2si __attribute__ ((vector_size(8)));

void
mandel_3dnow(unsigned char *image, const struct spec *s)
{
    v2sf xmin, ymin, xscale, yscale, iter_scale, depth_scale;
    v2sf threshold = (v2sf) { 4.0, 4.0 };
    v2si one = (v2si) { 1, 1 };

    xmin = (v2sf) { s->xlim[0], s->xlim[0] };
    ymin = (v2sf) { s->ylim[0], s->ylim[0] };
    xscale = (v2sf) { (s->xlim[1] - s->xlim[0]) / s->width,
        (s->xlim[1] - s->xlim[0]) / s->width };
    yscale = (v2sf) { (s->ylim[1] - s->ylim[0]) / s->height,
        (s->ylim[1] - s->ylim[0]) / s->height };
    iter_scale = (v2sf) { 1.0f / s->iterations,
        1.0f / s->iterations };
    depth_scale = (v2sf) { s->depth - 1, s->depth - 1 };

    __builtin_ia32_femms();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < s->height; y++) {
        for (int x = 0; x < s->width; x += 2) {
            v2sf mx = (v2sf) { x, x+1 };
            v2sf my = (v2sf) { y, y };
            v2sf cr = __builtin_ia32_pfadd(__builtin_ia32_pfmul(mx, xscale), xmin);
            v2sf ci = __builtin_ia32_pfadd(__builtin_ia32_pfmul(my, yscale), ymin);
            v2sf zr = cr;
            v2sf zi = ci;

            int k = 1;
            v2si mk = (v2si) { 1, 1 };
            while (++k < s->iterations) {
                /* Compute z1 from z0 */
                v2sf zr2 = __builtin_ia32_pfmul(zr, zr);
                v2sf zi2 = __builtin_ia32_pfmul(zi, zi);
                v2sf zrzi = __builtin_ia32_pfmul(zr, zi);

                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = __builtin_ia32_pfadd(__builtin_ia32_pfsub(zr2, zi2), cr);
                zi = __builtin_ia32_pfadd(__builtin_ia32_pfadd(zrzi, zrzi), ci);

                /* Increment k */
                zr2 = __builtin_ia32_pfmul(zr, zr);
                zi2 = __builtin_ia32_pfmul(zi, zi);
                v2sf mag2 = __builtin_ia32_pfadd(zr2, zi2);
                v2si mask = __builtin_ia32_pfcmpge(mag2, threshold);
		mk = __builtin_ia32_paddd(mk, one);
                mk = __builtin_ia32_psubd(mk, __builtin_ia32_pand(one, mask));

                /* Early bailout? */
                int imask[2];
                *(v2si *)&imask = mask;

                if (imask[0] && imask[1])
                    break;
            }

            v2sf fmk = __builtin_ia32_pi2fd(mk);

            fmk = __builtin_ia32_pfmul(fmk, iter_scale);
            fmk = __builtin_ia32_pfmul(__builtin_ia32_pfrsqrt(fmk), fmk);
            fmk = __builtin_ia32_pfmul(fmk, depth_scale);

            v2si pixels = __builtin_ia32_pf2id(fmk);

            unsigned char *dst = image + y * s->width * 3 + x * 3;
            unsigned char *src = (unsigned char *)&pixels;

            for (int i = 0; i < 2; i++) {
                dst[i * 3 + 0] = src[i * 4];
                dst[i * 3 + 1] = src[i * 4];
                dst[i * 3 + 2] = src[i * 4];
            }
        }
    }

    __builtin_ia32_femms();
}
