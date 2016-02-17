#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cblas.h>

// 8000 x 6000
int const F = 36;
int const N = 18800 * 100;
float const THRESH = 0.1;
int T;
int C;
int K;


void dot(float *x, float *y, float *z, int m, int n, int p) {
	#ifdef USE_OPENBLAS
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, p, n, 1.0, x, m, y,
				n, 0.0, z, m);
	#else
	#ifdef USE_OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int i=0; i<m; i++) {
		for (int k=0; k<p; k++) {
			float acc = 0;
			for (int j=0; j<n; j++) {
				acc += x[i*n+j] * y[n*p+k];
			}
			z[i*p+k] = acc;
		}
	}
	#endif
}

void arr_abs(float *x, int m, int n) {
	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int ind=0; ind<m*n; ind++) {
		x[ind] = fabsf(x[ind]);
	}
}

void nearest_atoms(float *similarities, int *inds, int m, int n) {
	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int i=0; i<m; i++) {
		int best_ind = -1;
		float best_val = -1;
		for (int j=0; j<n; j++) {
			float val = similarities[i*n+j];
			if ((best_ind == -1) || (best_val < val)) {
				best_ind = j;
				best_val = val;
			}
		}
		inds[i] = best_ind;
	}
}

void st(float *x, float *y, int m, int n, float thresh) {
	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int ind=0; ind<m*n; ind++) {
		float v = x[ind];
		y[ind] = ((float)((v > 0) - (v < 0))) * fmaxf(v - thresh, 0);
	}
}

void sub(float *a, float *b, float *c, int m, int n) {
	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int ind=0; ind<m*n; ind++) {
		c[ind] = a[ind] - b[ind];
	}
}

void greedy_ks(float *x, int *y, int m, int n) {
	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int i=0; i<m; i++) {
		int best_k = -1;
		float best_val = -1;
		for (int j=0; j<n; j++) {
			float v = fabsf(x[i*n+j]);
			if ((best_k == -1) || (v > best_val)) {
				best_k = j;
				best_val = v;
			}
		}
		y[i] = best_k;
	}
}

void outer_update(float *x, float *y, float *z, int *ks, int m, int n) {
	#ifdef USE_OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			z[i*n+j] += x[i*n+ks[i]] * y[ks[i]*n+j];
		}
	}
}

void slice_update(float *z, float *z_hat, int *ks, int m, int n) {
	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int i=0; i<m; i++) {
		z[i*n+ks[i]] = z_hat[i*n+ks[i]];
	}
}

double a_plus() {
	// Allocate patches
	float *patches = (float*)malloc(N*K*sizeof(float));
	// Allocate patches_sr
	float *patches_sr = (float*)malloc(N*F*sizeof(float));
	// Allocate LR dictionary
	float *dict_lr = (float*)malloc(K*C*sizeof(float));
	// Allocate Projections
	float* projs = (float*)malloc(C*K*F*sizeof(float));
	// Allocate nearest indices
	int *nn_inds = (int*)malloc(N*sizeof(int));
	// Allocate similarities
	float *similarities = (float*)malloc(N*C*sizeof(float));
	
	// Calculate nearest dictionary atoms
	double start_time = omp_get_wtime();
	dot(patches, dict_lr, similarities, N, K, C);
	arr_abs(similarities, N, C);
	nearest_atoms(similarities, nn_inds, N, C);
	
	// Project back to HR space
	for (int n=0; n<N; n++) {
		int ind = nn_inds[n];
		#ifdef USE_OPENBLAS
		cblas_sgemv(CblasColMajor, CblasTrans, K, F, 1.0, &projs[ind*K*F], K,
					&patches[n*K], 1, 0.0, &patches_sr[n*F], 1);
		#else
		#ifdef USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int f=0; f<F; f++) {
			float acc = 0;
			for (int k=0; k<K; k++)
				acc += patches[n*K+k] * projs[ind*K*F+k*F+f];
			patches_sr[n*F+f] = acc;
		}
		#endif
	}
	
	double duration = omp_get_wtime() - start_time;
	
	// Deallocate
	free(patches);
	free(patches_sr);
	free(dict_lr);
	free(projs);
	free(nn_inds);
	free(similarities);

	return duration;
}

double lcod() {
	// Allocate patches
	float *patches = (float*)malloc(N*K*sizeof(float));
	// Allocate patches_sr
	float *patches_sr = (float*)malloc(N*F*sizeof(float));
	// Allocate LR dictionary
	float *dict_lr = (float*)malloc(K*C*sizeof(float));
	// Allocate HR dictionary
	float *dict_hr = (float*)malloc(C*F*sizeof(float));
	// Allocate S matrix
	float *s = (float*)malloc(C*C*sizeof(float));
	
	// Allocate B
	float *b = (float*)malloc(N*C*sizeof(float));
	// Allocate Z
	float *z = (float*)calloc(N*C, sizeof(float));
	// Allocate Z_hat
	float *z_hat = (float*)malloc(N*C*sizeof(float));
	// Allocate tmp
	float *tmp = (float*)malloc(N*C*sizeof(float));
	// Allocate ks
	int *ks = (int*)malloc(N*sizeof(int));
	
	
	// Perform Coordinate Descent
	double start_time = omp_get_wtime();
	dot(patches, dict_lr, b, N, K, C);
	for (int t=1; t<=T-1; t++) {
		st(b, z_hat, N, C, THRESH);
		int k;
		if (t > 1) {
			sub(z_hat, z, tmp, N, C);
			greedy_ks(tmp, ks, N, C);
			outer_update(tmp, s, b, ks, N, C);
		} else {
			greedy_ks(z_hat, ks, N, C);
			outer_update(z_hat, s, b, ks, N, C);
		}
		slice_update(z, z_hat, ks, N, C);
	}
	st(b, z, N, C, THRESH);
	
	// Project back to HR space
	dot(z, dict_hr, patches_sr, N, C, F);
	
	double duration = omp_get_wtime() - start_time;
	
	// De-allocate
	free(patches);
	free(patches_sr);
	free(dict_lr);
	free(dict_hr);
	free(s);
	free(b);
	free(z);
	free(z_hat);
	free(tmp);
	free(ks);

	return duration;
}

int main_a_plus(int argc, char* argv[]) {
	char *file_name;
	FILE *pFile;
	if (argc > 1) {
		file_name = argv[1];
		pFile = fopen(file_name, "w");
		fprintf(pFile, "A+\n");
	}
	int const times = 2;

	int Cs[7] = {16, 32, 64, 128, 256, 512, 1024};
	int Ks[1] = {28};
	
	for (int ic=0; ic<7; ic++) {
	    for (int ik=0; ik<1; ik++) {
	        C = Cs[ic];
	        K = Ks[ik];
	        
	        double duration_a_plus = 0;
	        for (int ind=0; ind<times; ind++) {
	            duration_a_plus += a_plus();
	        }
			duration_a_plus /= times;
			printf("C: %03d K: %03d duration: %.4f\n",
				   C, K, duration_a_plus);
		   	if (argc > 1) {
				fprintf(pFile,
						"C: %03d K: %03d duration: %.4f\n",
						C, K, duration_a_plus);
		   	}
	    }
	}
	
	if (argc > 1) fclose(pFile);

	return 0;
}

int main_lcod(int argc, char* argv[]) {
	char *file_name;
	FILE *pFile;
	if (argc > 1) {
		file_name = argv[1];
		pFile = fopen(file_name, "a");
		fprintf(pFile, "LCOD\n");
	}
	int const times = 2;

	int Ts[4] = {1, 2, 4, 8};
	int Cs[4] = {32, 64, 128, 256};
	int Ks[2] = {28, 56};

	for (int it=0; it<4; it++) {
		for (int ic=0; ic<4; ic++) {
			for (int ik=0; ik<2; ik++) {
				T = Ts[it];
				C = Cs[ic];
				K = Ks[ik];
				
				if (K >= C) continue;

				double duration_lcod = 0;
				for (int ind=0; ind<times; ind++) {
					duration_lcod += lcod();
				}
				duration_lcod /= times;
				printf("T: %03d C: %03d K: %03d duration: %.4f\n",
					   T, C, K, duration_lcod);
			   	if (argc > 1) {
					fprintf(pFile,
							"T: %03d C: %03d K: %03d duration: %.4f\n",
							T, C, K, duration_lcod);
			   	}
			}
		}
	}
	
	if (argc > 1) fclose(pFile);

	return 0;
}

int main(int argc, char* argv[]) {
    int return_code1 = main_a_plus(argc, argv);
    int return_code2 = main_lcod(argc, argv);
    return return_code1 + return_code2;
}
