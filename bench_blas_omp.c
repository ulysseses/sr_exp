#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cblas.h>

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

void greedy_k1(float *x, float *y, int m, int n) {
	#pragma omp parallel for
	for (int j=0; j<n; j++) {
		float acc = 0;
		for (int i=0; i<m; i++) {
			acc += fabsf(x[i*n+j]);
		}
		y[j] = acc;
	}
}

int greedy_k2(float *y, int n) {
	int best_k = -1;
	float best_val = -1;
	for (int j=0; j<n; j++) {
		float val = y[j];
		if ((best_k == -1) || (val > best_val)) {
			best_k = j;
			best_val = val;
		}
	}
	return best_k;
}

void outer_update(float *x, float *y, float *z, int m, int n, int k) {
	#ifdef USE_OPENBLAS
	cblas_sger(CblasColMajor, m, n, 1.0, &x[k*m], 1, &y[k*m], 1, z, m);
	#else
	#ifdef USE_OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			z[i*n+j] += x[i*n+k]*y[k*n+j];
		}
	}
	#endif
}

void slice_update(float *z, float *z_hat, int m, int n, int k) {
	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int i=0; i<m; i++) {
		z[i*n+k] = z_hat[i*n+k];
	}
}

void add(float *a, float *b, float *c, int m, int n) {
	#ifdef USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int ind=0; ind<m*n; ind++) {
		c[ind] = a[ind] + b[ind];
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
	// Allocate l1s
	float *l1s = (float*)malloc(C*sizeof(float));
	
	
	// Perform Coordinate Descent
	double start_time = omp_get_wtime();
	dot(patches, dict_lr, b, N, K, C);
	for (int t=0; t<T-1; t++) {
		st(b, z_hat, N, C, THRESH);
		int k;
		if (t > 1) {
			sub(z_hat, z, tmp, N, C);
			greedy_k1(tmp, l1s, N, C);
			k = greedy_k2(l1s, C);
			outer_update(tmp, s, b, N, C, k);
		} else {
			greedy_k1(z_hat, l1s, N, C);
			k = greedy_k2(l1s, C);
			outer_update(z_hat, s, b, N, C, k);
		}
		slice_update(z, z_hat, N, C, k);
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
	free(l1s);

	return duration;
}

double lista() {
	// Allocate patches
	float *patches = (float*)malloc(N*K*sizeof(float));
	// Allocate patches_sr
	float *patches_sr = (float*)malloc(N*F*sizeof(float));
	// Allocate LR dictionary
	float *dict_lr = (float*)malloc(K*C*sizeof(float));
	// Allocate HR dictionary
	float *dict_hr = (float*)malloc(C*F*sizeof(float));
	// Allocate S matrix
	float *s1 = (float*)malloc(C*(C/4)*sizeof(float));
	float *s2 = (float*)malloc((C/4)*C*sizeof(float));
	float *tmp = (float*)malloc(N*(C/4)*sizeof(float));
	
	// Allocate B
	float *b = (float*)malloc(N*C*sizeof(float));
	// Allocate Z
	float *z = (float*)malloc(N*C*sizeof(float));
	// Allocate C
	float *c = (float*)malloc(N*C*sizeof(float));
	
	// Perform Iterated Shrinkage-Thresholding
	double start_time = omp_get_wtime();
	dot(patches, dict_lr, b, N, K, C);
	st(b, z, N, C, THRESH);
	for (int t=1; t<=T; t++) {
		dot(z, s1, tmp, N, C, C/4);
		dot(tmp, s2, c, N, C/4, C);
		add(b, c, c, N, C);
		st(c, z, N, C, THRESH);
	}
	
	// Project back to HR space
	dot(z, dict_hr, patches_sr, N, C, F);
	
	double duration = omp_get_wtime() - start_time;

	// Deallocate
	free(patches);
	free(patches_sr);
	free(dict_lr);
	free(dict_hr);
	free(s1);
	free(s2);
	free(tmp);
	free(b);
	free(z);
	free(c);

	return duration;
}

int main(int argc, char* argv[]) {
	char *file_name;
	FILE *pFile;
	if (argc > 1) {
		file_name = argv[1];
		pFile = fopen(file_name, "w");
	}

	int Ts[3] = {1, 2, 4};
	int Cs[4] = {16, 32, 64, 128};
	int Ks[4] = {10, 20, 50, 100};

	int const times = 2;
	for (int it=0; it<3; it++) {
		for (int ic=0; ic<4; ic++) {
			for (int ik=0; ik<4; ik++) {
				T = Ts[it];
				C = Cs[ic];
				K = Ks[ik];

				double duration_a_plus = 0;
				double duration_lcod = 0;
				double duration_lista = 0;
				for (int ind=0; ind<times; ind++) {
					duration_a_plus += a_plus();
					duration_lcod += lcod();
					duration_lista += lista();
				}
				duration_a_plus /= times;
				duration_lcod /= times;
				duration_lista /= times;
				printf("T: %03d C: %03d K: %03d duration (A+/LCoD/LISTA): (%.4f/%.4f/%.4f)\n",
					   T, C, K, duration_a_plus, duration_lcod, duration_lista);
			   	if (argc > 1) {
					fprintf(pFile,
							"T: %03d C: %03d K: %03d duration (A+/LCoD/LISTA): (%.4f/%.4f/%.4f)\n",
							T, C, K, duration_a_plus, duration_lcod, duration_lista);
			   	}
			}
		}
	}
	
	if (argc > 1) fclose(pFile);

	return 0;
}
