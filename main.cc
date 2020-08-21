#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "im2col.h"

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
};

static params<float> shapes[] = {
    {1024, 4, 4, 1, 1},  {512, 4, 4, 1, 1},   {512, 7, 7, 1, 1},
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

bool exact(float *a, float *b, size_t nelem) {
  for (size_t i = 0; i < nelem; ++i) {
    if (*a++ != *b++)
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

  std::cout << "Workset i:" << (double)shape->i_size() / 1024 << "K"
            << " o:" << (double)shape->o_size() / 1024 << "K" << std::endl;

  im2col(t_in, t_out, shape->h, shape->w, shape->c, 3, 3, shape->hp, shape->wp);
  im2col_simd(t_in, t_cmp, shape->h, shape->w, shape->c, 3, 3, shape->hp,
              shape->wp);

  if (!exact(t_out, t_cmp, shape->o_nelem()))
    std::cout << "Error in Copy!" << std::endl;

  auto start = Time::now();
  for (int i = 0; i < times; ++i) {
    im2col_simd(t_in, t_cmp, shape->h, shape->w, shape->c, 3, 3, shape->hp,
                shape->wp);
  }
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() - start)
          .count();
  std::cout << "SPMD execution time:" << (double)duration/times << "ms" << std::endl;

  start = Time::now();
  for (int i = 0; i < times; ++i) {
    im2col(t_in, t_cmp, shape->h, shape->w, shape->c, 3, 3, shape->hp,
           shape->wp);
  }
  duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() - start)
          .count();
  std::cout << "Normal execution time:" << (double)duration/times << "ms" << std::endl;

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
