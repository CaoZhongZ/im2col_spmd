#pragma once

void im2col(const float *im, float *col, int h, int w, int c, int kh, int kw,
            int hp, int wp);
void im2col_simd(const float *im, float *col, int h, int w, int c, int kh,
                 int kw, int hp, int wp);
void im2col_simd_unroll(const float *im, float *col, int h, int w, int c,
                        int kh, int kw, int hp, int wp);

#define declare_copy_from_2d_array_simd_c_unroll(x)                            \
  void copy_from_2d_array_simd_##x(                                            \
      float *uniform dest, uniform int d_height, uniform int d_width,          \
      const float *uniform src, uniform int s_height, uniform int s_width,     \
      uniform int h_off, uniform int w_off) {
