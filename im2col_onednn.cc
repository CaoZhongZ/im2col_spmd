#include <cstdlib>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include "copy_1d_simd.h"

void im2col_unroll(int j_oh, int j_ow, int j_os, int j_ih, int j_iw, int j_is,
            int j_kh, int j_kw, int j_ks, int tp, int lp,
            const float *__restrict im, float *__restrict col, int ss, int sb,
            int cs, int cb) {
  const float *__restrict _im = reinterpret_cast<const float *__restrict>(im);
  float *__restrict _col = reinterpret_cast<float *__restrict>(col);

  const size_t im_step = j_is;
  const size_t col_step = j_ks * sb;
  const int dh = 1;
  const int dw = 1;
  const int sh = 1;
  const int first_oh = ss / j_ow;
  const int last_oh = (ss + sb - 1) / j_ow;
  const int oh_begin = first_oh;
  const int oh_end = last_oh + 1;
  const int first_ow = ss % j_ow;
  const int last_ow = (ss + sb - 1) % j_ow;

  for (int ic = 0; ic < cb; ic += 8) {
    const float *__restrict im_ic = _im + (ic + cs) * im_step;
    for (int kh = 0; kh < j_kh; kh++) {
      for (int kw = 0; kw < j_kw; kw++) {
        float *__restrict col_k =
            _col + ic * col_step + (kh * j_kw + kw) * sb;
        for (int oh = oh_begin; oh < oh_end; oh++) {
          const int ih = oh * sh - tp + kh * dh;
          const float *__restrict im_ = im_ic + ih * j_iw;
          const int ow_begin = (oh == first_oh) ? first_ow : 0;
          const int ow_end = (oh == last_oh) ? (last_ow + 1) : j_ow;
          float *__restrict col_ = col_k + oh * j_ow - ss;
          if (ih < 0 || ih >= j_ih)
            zero_1d_simd_c_unroll_8(col_ + ow_begin, col_step,
                                    ow_end - ow_begin);
          else {
            auto start_off = ow_begin - lp + kw * dw;
            auto len = ow_end - ow_begin;
            copy_1d_simd_c_unroll_8(col_ + ow_begin, col_step, im_, im_step,
                                    start_off, j_iw, len);
          }
        }
      }
    }
  }
}

void im2col(int j_oh, int j_ow, int j_os, int j_ih, int j_iw, int j_is,
            int j_kh, int j_kw, int j_ks, int tp, int lp,
            const float *__restrict im, float *__restrict col, int ss, int sb,
            int cs, int cb) {
  const float *__restrict _im = reinterpret_cast<const float *__restrict>(im);
  float *__restrict _col = reinterpret_cast<float *__restrict>(col);

  const size_t im_step = j_is;
  const size_t col_step = j_ks * sb;
  const int dh = 1;
  const int dw = 1;
  const int sh = 1;
  const int first_oh = ss / j_ow;
  const int last_oh = (ss + sb - 1) / j_ow;
  const int oh_begin = first_oh;
  const int oh_end = last_oh + 1;
  const int first_ow = ss % j_ow;
  const int last_ow = (ss + sb - 1) % j_ow;

  for (int ic = 0; ic < cb; ic++) {
    const float *__restrict im_ic = _im + (ic + cs) * im_step;
    for (int kh = 0; kh < j_kh; kh++) {
      for (int kw = 0; kw < j_kw; kw++) {
        float *__restrict col_k =
            _col + ic * col_step + (kh * j_kw + kw) * sb;
        for (int oh = oh_begin; oh < oh_end; oh++) {
          const int ih = oh * sh - tp + kh * dh;
          const float *__restrict im_ = im_ic + ih * j_iw;
          const int ow_begin = (oh == first_oh) ? first_ow : 0;
          const int ow_end = (oh == last_oh) ? (last_ow + 1) : j_ow;
          float *__restrict col_ = col_k + oh * j_ow - ss;
          if (ih < 0 || ih >= j_ih)
            zero_1d_simd(col_ + ow_begin, ow_end - ow_begin);
          else {
            auto start_off = ow_begin - lp + kw * dw;
            auto len = ow_end - ow_begin;
            copy_1d_simd(col_ + ow_begin, im_, start_off, j_iw, len);
          }
        }
      }
    }
  }
}

template <typename dtype> struct params {
  int c, h, w;
  int hp, wp;

  size_t i_nelem() const { return c * h * w; }

  size_t o_nelem() const {
    auto oh = h + 2 * hp - 3 + 1;
    auto ow = w + 2 * wp - 3 + 1;

    return c * oh * ow * 9;
  }

  size_t i_size() const { return i_nelem() * sizeof(dtype); }
  size_t o_size() const { return o_nelem() * sizeof(dtype); }
  size_t oh() const { return h + 2 * hp - 3 + 1; }
  size_t ow() const { return w + 2 * wp - 3 + 1; }
};

static params<float> shapes[] = {
    {256, 56, 56, 1, 1},
    {512, 4, 4, 1, 1},  {1024, 4, 4, 1, 1},   {512, 7, 7, 1, 1},
    {256, 14, 14, 1, 1}, {256, 28, 28, 1, 1}, {128, 28, 28, 1, 1},
    {128, 56, 56, 1, 1},
};

void fill_float(float *f, size_t nelem) {
  std::random_device rd_;
  std::mt19937 gen_{rd_()};
  std::uniform_real_distribution<float> dstr_{-1., 1.};

  for (size_t i = 0; i < nelem; ++i) {
    f[i] = dstr_(gen_);
  }
}

void seq_float(float *f, size_t nelem) {
  for (size_t i = 0; i < nelem; ++i) {
    f[i] = (float)i;
  }
}

void zero_float(float *f, size_t nelem) {
  for (size_t i = 0; i < nelem; ++i) {
    f[i] = 0.;
  }
}

void show_3d(float *a, int oh, int ow) {
}

bool exact(float *a, float *b, size_t nelem) {
  for (size_t i = 0; i < nelem; ++i) {
    if (*(a+i) != *(b+i))
      return false;
  }

  return true;
}

using Time = std::chrono::high_resolution_clock;

void bench_im2col(const params<float> *shape, int times) {
  float *t_in, *t_out, *t_cmp;

  posix_memalign((void **)&t_in, 4096, shape->i_size());
  posix_memalign((void **)&t_out, 4096, shape->o_size());
  posix_memalign((void **)&t_cmp, 4096, shape->o_size());

  seq_float(t_in, shape->i_nelem());
  zero_float(t_out, shape->o_nelem());
  zero_float(t_cmp, shape->o_nelem());

  std::cout << "Workset i:" << (double)shape->i_size() / 1024 << "K"
            << " o:" << (double)shape->o_size() / 1024 << "K" << std::endl;

  im2col(shape->oh(), shape->ow(), shape->oh() * shape->ow(),
      shape->h, shape->w, shape->h * shape->w,
      3, 3, 9, shape->hp, shape->w,
      t_in, t_out, 0, 120, 0, 32);
  im2col_unroll(shape->oh(), shape->ow(), shape->oh() * shape->ow(),
      shape->h, shape->w, shape->h * shape->w,
      3, 3, 9, shape->hp, shape->w,
      t_in, t_cmp, 0, 120, 0, 32);

  if (!exact(t_out, t_cmp, shape->o_nelem()))
    std::cout << "Error in Copy!" << std::endl;

  auto start = Time::now();
  for (int i = 0; i < times; ++i) {
    im2col(shape->oh(), shape->ow(), shape->oh() * shape->ow(),
        shape->h, shape->w, shape->h * shape->w,
        3, 3, 9, shape->hp, shape->w,
        t_in, t_out, 0, 120, 0, 32);
  }
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() - start)
          .count();
  std::cout << "SPMD execution time:" << duration << "ms";

  start = Time::now();
  for (int i = 0; i < times; ++i) {
    im2col_unroll(shape->oh(), shape->ow(), shape->oh() * shape->ow(),
        shape->h, shape->w, shape->h * shape->w,
        3, 3, 9, shape->hp, shape->w,
        t_in, t_cmp, 0, 120, 0, 32);
  }
  duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() - start)
          .count();
  std::cout << " vs. " << duration << "ms";
  std::cout << std::endl;

  free(t_in);
  free(t_out);
  free(t_cmp);
}

int main() {
  for (int sample = 0; sample < sizeof(shapes) / sizeof(params<float>);
       ++sample) {
    bench_im2col(&shapes[sample], 1000);
  }
}
