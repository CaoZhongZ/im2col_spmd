// Copy from a 2d array[h][w], start from h_off, w_off to
// dest[d_height][d_width] h_off, w_off could be negative to indicate up/left
// padding (0) s_height/s_width could be smaller than hdd/ldd indidate
// down/right padding (0)

export void copy_from_2d_array_simd(float *uniform dest,
                                    uniform int d_height,
                                    uniform int d_width,
                                    const float *uniform src,
                                    uniform int s_height,
                                    uniform int s_width,
                                    uniform int h_off,
                                    uniform int w_off) {
  for (uniform int j = 0; j < d_height; ++j) {
    // Transform to src coordinate
    uniform int j_off = j + h_off;

    // Out of bound
    if (j_off < 0 || j_off >= s_height) {
      for (int pid = programIndex; pid < d_width; pid += programCount) {
        (dest + j * d_width)[pid] = 0.;
      }
    } else {
      for (int pid = programIndex; pid < d_width; pid += programCount) {
        // Transform to src coordinate
        int p_off = pid + w_off;

        // Test boundaries
        (dest + j * d_width)[pid] = (p_off >= 0 && p_off < s_width)
                                        ? (src + j_off * s_width)[p_off]
                                        : 0;
      }
    }
  }
}

// export void copy_from_1d_array(float *uniform dst, uniform int d_width,
//                                float *uniform src, uniform int w_off,
//                                uniform int s_width) {
// 
//   for (int pid = programIndex; pid < d_width; pid += programCount) {
//     int p_off = pid + w_off;
// 
//     dst[pid] = (p_off >= 0 || p_off < s_width) ? src[p_off] : 0;
//   }
// }
