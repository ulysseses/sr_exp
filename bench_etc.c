#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// 1600 x 1200
int const H = 1200;
int const W = 1600;
int const PW = 6;
int const OVERLAP = 5;
int const F = 36;
float const THRESH = 0.1;

void init_img(float *img, int h, int w) {
    float ind = 0;
    #pragma omp parallel for collapse(2)
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            img[i*w+j] = ind++;
        }
    }
}

void init_mask(float *mask, int h, int w) {
    #pragma omp parallel for collapse(2)
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            mask[i*w+j] = 1e-8;
        }
    }
}

int num_patches(int h, int w, int pw, int overlap) {
    int const stride = pw - overlap;
    int count = 0;
    #pragma omp parallel for collapse(2)
    for (int i=0; i<h-pw+1; i+=stride) {
        for (int j=0; j<w-pw+1; j+=stride) {
            count += 1;
        }
    }
    return count;
}

void img2patches(float *img, int h, int w, float *patches, int n, int pw, int overlap) {
    int const stride = pw - overlap;
    int ind = 0;
    #pragma omp parallel for collapse(2)
    for (int i=0; i<h-pw+1; i+=stride) {
        for (int j=0; j<w-pw+1; j+=stride) {
            for (int y=0; y<pw; y++) {
                for (int x=0; x<pw; x++) {
                    patches[ind*(pw*pw)+y*pw+x] = img[(i+y)*w+(j+x)];
                }
            }
            ind += 1;
        }
    }
}

void patches2img(float *img, float *mask, int h, int w, float *patches, int n,
                 int pw, int overlap) {
    int const stride = pw - overlap;
    int ind = 0;
    #pragma omp parallel for collapse(2)
    for (int i=0; i<h-pw+1; i+=stride) {
        for (int j=0; j<w-pw+1; j+=stride) {
            for (int y=0; y<pw; y++) {
                for (int x=0; x<pw; x++) {
                    img[(i+y)*w+(j+x)] += patches[ind*(pw*pw)+y*pw+x];
                    mask[(i+y)*w+(j+x)] += 1.0;
                }
            }
            ind += 1;
        }
    }
    
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            img[i*w+j] /= mask[i*w+j];
        }
    }
}

int main() {
    float *img = (float*)malloc(H*W*sizeof(float));
    init_img(img, H, W);
    int n = num_patches(H, W, PW, OVERLAP);
    float *patches = (float*)malloc(n*PW*PW*sizeof(float));
    
    double start_time;
    double duration;
    
    start_time = omp_get_wtime();
    img2patches(img, H, W, patches, n, PW, OVERLAP);
    duration = omp_get_wtime() - start_time;
    printf("img2patches duration: %.3f\n", duration);
    free(img);
    
    img = (float*)calloc(H*W, sizeof(float));
    float *mask = (float*)calloc(H*W, sizeof(float));
    init_mask(mask, H, W);
    start_time = omp_get_wtime();
    patches2img(img, mask, H, W, patches, n, PW, OVERLAP);
    duration = omp_get_wtime() - start_time;
    printf("patches2img duration: %.3f\n", duration);
    
    free(img);
    free(mask);
    free(patches);
    return 0;
}
