#include "/home/andrey/Codes/paradigms/parallel/opencl/myopencl/Color.h"

kernel void sepia( global Color* A, global Color* B, int size) {
    const int idx = get_global_id(0);
    if (size <= idx) return;
    B[idx].r = (A[idx].r * 0.393 + A[idx].g * 0.769 + 0.189 * A[idx].b) > 255 ? 255 : (A[idx].r * 0.393 + A[idx].g * 0.769 + 0.189 * A[idx].b);
    B[idx].g = (A[idx].r * 0.349 + A[idx].g * 0.686 + 0.168 * A[idx].b) > 255 ? 255 : (A[idx].r * 0.349 + A[idx].g * 0.686 + 0.168 * A[idx].b);
    B[idx].b = (A[idx].r * 0.272 + A[idx].g * 0.534 + 0.131 * A[idx].b) > 255 ? 255 : (A[idx].r * 0.272 + A[idx].g * 0.534 + 0.131 * A[idx].b);
}