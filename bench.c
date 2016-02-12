#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int F = 36;
int N = 18800;
float THRESH = 0.1;
int T = 2;
int C = 16;
int K = 28;

void dot(float *x, float *y, float *z, int m, int n, int p) {
	for (int i=0; i<m; i++) {
		for (int k=0; k<p; k++) {
			float acc = 0;
			for (int j=0; j<n; j++) {
				acc += x[i*n+j] * y[n*p+k];
			}
			z[i*p+k] = acc;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// A+
void arr_abs(float *x, int m, int n) {
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++)
			x[i*n+j] = fabsf(x[i*n+j]);
	}
}

void nearest_atoms(float *similarities, int *inds, int m, int n) {
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

float a_plus() {
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
	float start_time = (float)clock() / CLOCKS_PER_SEC;
	dot(patches, dict_lr, similarities, N, K, C);
	arr_abs(similarities, N, C);
	nearest_atoms(similarities, nn_inds, N, C);
	
	// Project back to HR space
	for (int n=0; n<N; n++) {
		int ind = nn_inds[n];
		for (int f=0; f<F; f++) {
			float acc = 0;
			for (int k=0; k<K; k++)
				acc += patches[n*K+k] * projs[ind*K*F+k*F+f];
			patches_sr[n*F+f] = acc;
		}
	}
	float duration = (float)clock() / CLOCKS_PER_SEC - start_time;
	
	//printf("duration: %.2f\n", duration);
	
	// Deallocate
	free(patches);
	free(patches_sr);
	free(dict_lr);
	free(projs);
	free(nn_inds);
	free(similarities);

	return duration;
}
///////////////////////////////////////////////////////////////////////////////

void st(float *x, float *y, int m, int n, float thresh) {
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			float v = x[i*n+j];
			float sign;
			if (v > 0) sign = 1.0;
			else sign = -1.0;
			float v2 = fabsf(v) - thresh;
			if (v2 > 0) y[i*n+j] = v2 * sign;
			else y[i*n+j] = 0;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// LCoD
void sub(float *a, float *b, float *c, int m, int n) {
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			c[i*n+j] = a[i*n+j] - b[i*n+j];
		}
	}
}

int greedy_k(float *x, int m, int n) {
	int best_k = -1;
	float best_val = -1;
	for (int j=0; j<n; j++) {
		float acc = 0;
		for (int i=0; i<m; i++)
			acc += fabsf(x[i*n+j]);
		if ((best_k == -1) || (acc > best_val)) {
			best_k = j;
			best_val = acc;
		}
	}
	return best_k;
}

void outer_update(float *x, float *y, float *z, int m, int n, int k) {
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			z[i*n+j] += x[i*n+k]*y[k*n+j];
		}
	}
}

void slice_update(float *z, float *z_hat, int m, int n, int k) {
	for (int i=0; i<m; i++)
		z[i*n+k] = z_hat[i*n+k];
}

float lcod() {
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
	
	// Perform Coordinate Descent
	float start_time = (float)clock() / CLOCKS_PER_SEC;
	dot(patches, dict_lr, b, N, K, C);
	for (int t=0; t<T; t++) {
		st(b, z_hat, N, C, THRESH);
		int k;
		if (t > 1) {
			sub(z_hat, z, tmp, N, C);
			k = greedy_k(tmp, N, C);
			outer_update(tmp, s, b, N, C, k);
		} else {
			k = greedy_k(z_hat, N, C);
			outer_update(z_hat, s, b, N, C, k);
		}
		slice_update(z, z_hat, N, C, k);
	}
	st(b, z, N, C, THRESH);
	
	// Project back to HR space
	dot(z, dict_hr, patches_sr, N, C, F);
	float duration = (float)clock() / CLOCKS_PER_SEC - start_time;
	
	//printf("duration: %.2f\n", duration);
	
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

	return duration;
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// LISTA
void add(float *a, float *b, float *c, int m, int n) {
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			c[i*n+j] = a[i*n+j] + b[i*n+j];
		}
	}
}

float lista() {
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
	float *z = (float*)malloc(N*C*sizeof(float));
	// Allocate C
	float *c = (float*)malloc(N*C*sizeof(float));
	
	// Perform Iterated Shrinkage-Thresholding
	float start_time = (float)clock() / CLOCKS_PER_SEC;
	dot(patches, dict_lr, b, N, K, C);
	st(b, z, N, C, THRESH);
	for (int t=1; t<=T; t++) {
		dot(z, s, c, N, C, C);
		add(b, c, c, N, C);
		st(c, z, N, C, THRESH);
	}
	
	// Project back to HR space
	dot(z, dict_hr, patches_sr, N, C, F);
	float duration = (float)clock() / CLOCKS_PER_SEC - start_time;
	
	//printf("duration: %.2f\n", duration);
	
	// Deallocate
	free(patches);
	free(patches_sr);
	free(dict_lr);
	free(dict_hr);
	free(s);
	free(b);
	free(z);
	free(c);

	return duration;
}
///////////////////////////////////////////////////////////////////////////////

int main(void) {
	float duration_a_plus = 0;
	float duration_lcod = 0;
	float duration_lista = 0;

	FILE *pFile;
	pFile = fopen("log.txt", "w");

	int Ts[4] = {1, 2, 4, 8};
	//int Cs[4] = {16, 32, 64, 128};
	//int Ks[4] = {10, 20, 50, 100};
	int Cs[3] = {16, 32, 64};
	int Ks[1] = {28};

	int const times = 2;
	for (int it=0; it<4; it++) {
		for (int ic=0; ic<3; ic++) {
			for (int ik=0; ik<1; ik++) {
				T = Ts[it];
				C = Cs[ic];
				K = Ks[ik];
				if (K > C) continue;

				float duration_a_plus = 0;
				float duration_lcod = 0;
				float duration_lista = 0;
				for (int ind=0; ind<times; ind++) {
					duration_a_plus += a_plus();
					duration_lcod += lcod();
					duration_lista += lista();
				}
				duration_a_plus /= times;
				duration_lcod /= times;
				duration_lista /= times;
				printf("T: %03d C: %03d K: %03d duration (A+/LCoD/LISTA): (%.3f/%.3f/%.3f)\n",
					   T, C, K, duration_a_plus, duration_lcod, duration_lista);
				fprintf(pFile,
						"T: %03d C: %03d K: %03d duration (A+/LCoD/LISTA): (%.3f/%.3f/%.3f)\n",
						T, C, K, duration_a_plus, duration_lcod, duration_lista);
			}
		}
	}

	fclose(pFile);

	return 0;
}
