//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Mark Guevara
 * UCLA ID: 704962920
 * Email id: markgue@g.ucla.edu
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

int OMP_Index(int x, int y, int z)
{
	return ((z * yMax + y) * xMax + x);
}
#define Index(x, y, z) OMP_Index(x, y, z)

double OMP_SQR(double x)
{
	return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
	xMax = xM;
	yMax = yM;
	zMax = zM;
	assert(OMP_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
	assert(OMP_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
}
void OMP_Finish()
{
	free(OMP_conv);
	free(OMP_g);
}
void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
	double lambda = (Ksigma * Ksigma) / (double)(2 * stepCount);
	double nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
	int x, y, z, step, index;
	double boundryScale = 1.0 / (1.0 - nu);
	double postScale = pow(nu / lambda, (double)(3 * stepCount));

	// Performs these operations stepCount times 
	
	for(step = 0; step < stepCount; step++)
	{

		printf("GaussianBlur: step %d\n", step);

		
		// x=0 boundary
		#pragma omp for private (y, index)
		for (z = 0; z < zMax; z++)
		{
			for (y = 0; y < yMax - 1; y+=2)
			{
				index = Index(0, y, z);
				u[index] *= boundryScale;
				u[index + xMax] *= boundryScale;
			}

			for (; y < yMax; y++)
			{
				u[Index(0, y, z)] *= boundryScale;
			}
		}
		// x relies on x-1
		#pragma omp parallel for private (y, x, index)
		for (z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for (x = 1; x < xMax - 1; x+=2)
				{
					index = Index(x, y, z);
					u[index] += u[index - 1] * nu;
					u[index + 1] += u[index] * nu;
				}
				for (; x < xMax; x++)
				{
					index = Index(x, y, z);
					u[index] += u[index - 1] * nu;
				}
			}
		}
		// x=0 boundary
		// Should be this (?):
		// u[Index(xMax - 1, y, z)] *= boundryScale;
		#pragma omp for private (y, index)
		for (z = 0; z < zMax; z++)
		{
			for (y = 0; y < yMax - 1; y += 2)
			{
				index = Index(0, y, z);
				u[index] *= boundryScale;
				u[index + xMax] *= boundryScale;
			}

			for (; y < yMax; y++)
			{
				u[Index(0, y, z)] *= boundryScale;
			}
		}


		// x relies on x + 1
		#pragma omp parallel for private (y, x, index)
		for (z = 0; z < zMax; z++)
		{	
			for (y = 0; y < yMax; y++)
			{
				for (x = xMax - 2; x >= 1; x-=2)
				{
					index = Index(x, y, z);
					u[index] += u[index + 1] * nu;
					u[index - 1] += u[index] * nu;
				}
				for (; x >= 0; x--)
				{
					index = Index(x, y, z);
					u[index] += u[index + 1] * nu;
				}
			}
		}

		// y=0 boundary
		//#pragma omp for private (x, index)
		for (z = 0; z < zMax; z++)
		{
			for (x = 0; x < xMax; x++)
			{
				u[Index(x, 0, z)] *= boundryScale;
			}
		}

		// y relies on y-1
		#pragma omp for private (y, x, index)
		for (z = 0; z < zMax; z++)
		{
			for (y = 1; y < yMax; y++)
			{
				for (x = 0; x < xMax; x++)
				{
					u[Index(x, y, z)] += u[Index(x, y - 1, z)] * nu;
				}
			}
		}
		// y=yMax-1 boundary
		//#pragma omp parallel for private (z, index)
		
		for (z = 0; z < zMax; z++)
		{
			for (x = 0; x < xMax; x++)
			{
				u[Index(x, yMax - 1, z)] *= boundryScale;
			}
		}

		// y relies on y+1
		#pragma omp for private (y, x, index)
		for (z = 0; z < zMax; z++)
		{
			for (y = yMax - 2; y >= 0; y--)
			{
				for (x = 0; x < xMax; x++)
 				{
					u[Index(x, y, z)] += u[Index(x, y + 1, z)] * nu;
				}
			}
		}
		// z=0 boundary
		//#pragma omp for private (y, index)
		for (y = 0; y < yMax; y++)
		{
			for (x = 0; x < xMax; x++)
			{
				u[Index(x, y, 0)] *= boundryScale;
			}
		}

		// z relies on z-1
		#pragma omp for private (y, x, index)
		for (z = 1; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for (x = 0; x < xMax; x++)
				{
					u[Index(x, y, z)] = u[Index(x, y, z - 1)] * nu;
				}
			}
		}

		// z=zMax-1 boundary
		//#pragma omp parallel for private (y, index)
		for (y = 0; y < yMax; y++)
		{
			for (x = 0; x < xMax; x++)
			{
				u[Index(x, y, zMax - 1)] *= boundryScale;
			}
		}

		// z relies on z+1
		#pragma omp for private (y, x, index)
		for (z = zMax - 2; z >= 0; z--)
		{
			for(y = 0; y < yMax; y++)
			{
				for (x = 0; x < xMax; x++)
				{
					u[Index(x, y, z)] += u[Index(x, y, z + 1)] * nu;
				}
			}
		}
	}

	// Affects all
	#pragma omp parallel for private (y, x, index)
	for (z = 0; z < zMax; z++)
	{
		for(y = 0; y < yMax; y++)
		{
			for (x = 0; x < xMax; x+=2)
			{
				u[Index(x, y, z)] *= postScale;
				u[Index(x + 1, y, z)] *= postScale;
			}

			for (; x < xMax; x++)
			{
				u[Index(x, y, z)] *= postScale;
			}
		}
	}
}
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	double epsilon = 1.0e-7;
	double sigma2 = SQR(sigma);
	int x, y, z, iteration;
	int converged = 0;
	int lastConverged = 0;
	int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
	double* conv = OMP_conv;
	double* g = OMP_g;

	for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
	{
		printf("Deblur 1\n");
		#pragma omp parallel for private (y, x)
		for (z = 1; z < zMax - 1; z++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				for (x = 1; x < xMax - 1; x++)
				{
					// Replaced each "Index(x, y, z)" with a single calculation "xyz_index"
					int xyz_index = Index(x, y, z);
					g[xyz_index] = 1.0 / sqrt(epsilon +
						SQR(u[xyz_index] - u[xyz_index + 1]) +
						SQR(u[xyz_index] - u[xyz_index - 1]) +
						SQR(u[xyz_index] - u[xyz_index + xMax]) +
						SQR(u[xyz_index] - u[xyz_index - xMax]) +
						SQR(u[xyz_index] - u[xyz_index + xMax * yMax]) +
						SQR(u[xyz_index] - u[xyz_index - xMax * yMax]));
				}
				//for (z = 1; z < zMax - 1; z++)
				//{
				//	g[Index(x, y, z)] = 1.0 / sqrt(epsilon +
				//		SQR(u[Index(x, y, z)] - u[Index(x + 1, y, z)]) +
				//		SQR(u[Index(x, y, z)] - u[Index(x - 1, y, z)]) +
				//		SQR(u[Index(x, y, z)] - u[Index(x, y + 1, z)]) +
				//		SQR(u[Index(x, y, z)] - u[Index(x, y - 1, z)]) +
				//		SQR(u[Index(x, y, z)] - u[Index(x, y, z + 1)]) +
				//		SQR(u[Index(x, y, z)] - u[Index(x, y, z - 1)]));
				//}

			}
		}
		memcpy(conv, u, sizeof(double) * xMax * yMax * zMax);
		//OMP_GaussianBlur(conv, Ksigma, 3);

		printf("Deblur 2\n");

		#pragma omp parallel for private (y, x)
		for (z = 0; z < zMax; z++)
		{
			for (y = 0; y < yMax; y++)
			{
				for (x = 0; x < xMax; x++)
				{
					// Replaced each "Index(x, y, z)" with a single calculation "xyz_index"
					int xyz_index = Index(x, y, z);

					double r = conv[xyz_index] * f[xyz_index] / sigma2;
					r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
					conv[xyz_index] -= f[xyz_index] * r;
				}
			}
		}
		OMP_GaussianBlur(conv, Ksigma, 1);
		converged = 0;

		printf("Deblur 3\n");

		#pragma omp for private (y, x)
		for (z = 1; z < zMax - 1; z++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				for (x = 1; x < xMax - 1; x++)
				{
					// Replaced each use of Index() with a corresponding value that is calculated only once
					int xyz_index = Index(x, y, z);
					int up = Index(x, y + 1, z);
					int down = Index(x, y - 1, z);
					int left = Index(x - 1, y, z);
					int right = Index(x + 1, y, z);
					int forward = Index(x, y, z + 1);
					int back = Index(x, y, z - 1);

					double oldVal = u[xyz_index];
					double newVal = (u[xyz_index] + dt * ( 
						u[left]		*	g[left]		+ 
						u[right]	*	g[right]	+ 
						u[down]		*	g[down]		+ 
						u[up]		*	g[up]		+ 
						u[back]		*	g[back]		+ 
						u[forward]	*	g[forward] - gamma * conv[xyz_index])) /
						(1.0 + dt * (g[right] + g[left] + g[up] + g[down] + g[forward] + g[back]));
					if(fabs(oldVal - newVal) < epsilon)
					{
						converged++;
					}
					u[xyz_index] = newVal;
				}
			}
		}

		printf("Deblur 4\n");

		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}
}

