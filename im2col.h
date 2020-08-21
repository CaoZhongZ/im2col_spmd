#pragma once

void im2col(const float *im, float *col, int h, int w,
            int c, int kh, int kw, int hp, int wp);
void im2col_simd(const float *im, float *col, int h, int w,
            int c, int kh, int kw, int hp, int wp);
