#define PPMP_CONV2D_MAX_RADIUS 6

__constant__ float filter[PPMP_CONV2D_MAX_RADIUS * 2 + 1];

void conv2d(float *N, float *F, float *P, size_t radius, size_t width, size_t height){

}