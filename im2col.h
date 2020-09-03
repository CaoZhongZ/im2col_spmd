#pragma once

void im2col(const float *im, float *col, int h, int w, int c, int kh, int kw,
            int hp, int wp);
void im2col_simd(const float *im, float *col, int h, int w, int c, int kh,
                 int kw, int hp, int wp);
void im2col_simd_unroll(const float *im, float *col, int h, int w, int c,
                        int kh, int kw, int hp, int wp);

#define declare_copy_from_2d_array_simd_c_unroll(x)                            \
  void copy_from_2d_array_simd_##x(float *dest, int d_height, int d_width,     \
                                   const float *src, int s_height,             \
                                   int s_width, int h_off, int w_off)
