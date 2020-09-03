#include <cstdio>
#include <cstdlib>

#include "im2col.h"

extern "C" {
void copy_from_2d_array_simd(float *dst, int d_height, int d_width,
                             const float *src, int s_height, int s_width,
                             int h_off, int w_off);
declare_copy_from_2d_array_simd_c_unroll(4);
declare_copy_from_2d_array_simd_c_unroll(8);
declare_copy_from_2d_array_simd_c_unroll(16);
}

void copy_from_2d_array(float *dst, int d_height, int d_width, const float *src,
                        int s_height, int s_width, int h_off, int w_off) {
  for (int j = 0; j < d_height; ++j) {
    auto j_off = j + h_off;
    if (j_off < 0 || j_off >= s_height) {
      for (int i = 0; i < d_width; ++i)
        (dst + j * d_width)[i] = 0.;
    } else {
      for (int pid = 0; pid < d_width; ++pid) {
        auto p_off = pid + w_off;
        (dst + j * d_width)[pid] = (p_off >= 0 && p_off < s_width)
                                       ? (src + j_off * s_width)[p_off]
                                       : 0.;
      }
    }
  }
}

// simple single batch im2col interface
/* col[c][kh][kw][oh][ow] <-- im2col(im[c][h][w]) */
void im2col(const float *im, float *col, int h, int w, int c, int kh, int kw,
            int hp, int wp) {

  auto oh = h + 2 * hp - kh + 1;
  auto ow = w + 2 * wp - kw + 1;

  // for every image
  for (int ic = 0; ic < c; ++ic) {
    auto *im_c = im + ic * (h * w);

    for (int k1 = 0; k1 < kh; ++k1) {
      for (int k0 = 0; k0 < kw; ++k0) {
        // Copy h x w with offset according to k1/k0
        auto *col_c = col + (k0 + k1 * kw + ic * kw * kh) * (oh * ow);
        auto h_off = 0 - hp + k1;
        auto w_off = 0 - wp + k0;

        copy_from_2d_array(col_c, oh, ow, im_c, h, w, h_off, w_off);
      }
    }
  }
}

// simple single batch im2col interface
void im2col_simd(const float *im, float *col, int h, int w, int c, int kh,
                 int kw, int hp, int wp) {

  auto oh = h + 2 * hp - kh + 1;
  auto ow = w + 2 * wp - kw + 1;

  // for every image
  for (int ic = 0; ic < c; ++ic) {
    auto *im_c = im + ic * (h * w);

    for (int k1 = 0; k1 < kh; ++k1) {
      for (int k0 = 0; k0 < kw; ++k0) {
        // Copy h x w with offset according to k1/k0
        auto *col_c = col + (k0 + k1 * kw + ic * kw * kh) * (oh * ow);
        auto h_off = 0 - hp + k1;
        auto w_off = 0 - wp + k0;

        copy_from_2d_array_simd(col_c, oh, ow, im_c, h, w, h_off, w_off);
      }
    }
  }
}

// simple single batch im2col interface
void im2col_simd_unroll(const float *im, float *col, int h, int w, int c,
                        int kh, int kw, int hp, int wp) {

  auto oh = h + 2 * hp - kh + 1;
  auto ow = w + 2 * wp - kw + 1;

  // for every image
  for (int ic = 0; ic < c / 4; ++ic) {
    auto *im_c = im + ic * (h * w);

    for (int k1 = 0; k1 < kh; ++k1) {
      for (int k0 = 0; k0 < kw; ++k0) {
        // Copy h x w with offset according to k1/k0
        auto *col_c = col + (k0 + k1 * kw + ic * kw * kh) * (oh * ow);
        auto h_off = 0 - hp + k1;
        auto w_off = 0 - wp + k0;

        copy_from_2d_array_simd_4(col_c, oh, ow, im_c, h, w, h_off, w_off);
      }
    }
  }
}
