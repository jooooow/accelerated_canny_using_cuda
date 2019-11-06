#include <stdio.h>
#include <windows.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <math.h>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <device_functions.h>

using namespace cv;

#define SPLIT_SIZE_X 32
#define SPLIT_SIZE_Y 24
#define BLOCK_SIZE_X 36
#define BLOCK_SIZE_Y 28

/*canny using cuda*/
void CUDA_Canny();
__global__ void CUDA_GaussianAndSobel(unsigned char* img, int width, int height, unsigned char* output);
__device__ void CUDA_Gaussian(unsigned char* img, int width, int height, int idx, unsigned char* output);
__device__ void CUDA_Sobel(unsigned char* img, int width, int height, int idx, unsigned char* output);
__device__ unsigned char CUDA_GetPixelVal(unsigned char* img, int width, int height, int i, int j);


int main(void)
{
	printf("CANNY_CUDA\n");
	CUDA_Canny();
	//system("pause");
	return 0;
}

void CUDA_Canny()
{
	int width = 640 * 2;
	int height = 480 * 2;
	dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid_size(width / SPLIT_SIZE_X, height / SPLIT_SIZE_Y);
	Mat img_src, img_sobel, img_canny;

	VideoCapture camera(1);

	/*cpu memory*/
	unsigned char* cpu_sobel = new unsigned char[width * height];
	unsigned char* cpu_canny   = new unsigned char[width * height];

	/*gpu memory*/
	unsigned char* gpu_img;
	cudaMalloc(&gpu_img, width * height * sizeof(unsigned char));

	unsigned char* gpu_sobel;
	cudaMalloc(&gpu_sobel, width * height * sizeof(unsigned char));

	while (1)
	{
		camera >> img_src;
		resize(img_src, img_src, Size(width, height), 0, 0);
		cvtColor(img_src, img_src, CV_BGR2GRAY);
		imshow("img_src", img_src);

		cudaMemcpy(gpu_img, img_src.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		CUDA_GaussianAndSobel << <grid_size, block_size, 3 * (BLOCK_SIZE_X) * (BLOCK_SIZE_Y) * sizeof(unsigned char)>> > (gpu_img, width, height, gpu_sobel);
		cudaMemcpy(cpu_sobel, gpu_sobel, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		img_sobel = Mat(Size(width, height), CV_8UC1, cpu_sobel);
		imshow("img_guass&sobel", img_sobel);

		if ('q' == waitKey(1))
		{
			destroyAllWindows();
			free(cpu_sobel);
			cpu_sobel = NULL;
			free(cpu_canny);
			cpu_canny = NULL;
			cudaFree(gpu_img);
			cudaFree(gpu_sobel);

			break;
		}
	}
}

__global__ void CUDA_GaussianAndSobel(unsigned char* img, int width, int height, unsigned char* output)
{
	__shared__ unsigned char cache[(BLOCK_SIZE_X) * (BLOCK_SIZE_Y)];
	__shared__ unsigned char gauss[(BLOCK_SIZE_X) * (BLOCK_SIZE_Y)];
	__shared__ unsigned char sobel[(BLOCK_SIZE_X) * (BLOCK_SIZE_Y)];

	/*alloct img to cache*/
	int core_head = SPLIT_SIZE_X * SPLIT_SIZE_Y * blockIdx.y * gridDim.x + blockIdx.x * SPLIT_SIZE_X;
	int core_bias = SPLIT_SIZE_X * gridDim.x * threadIdx.y + threadIdx.x;
	int raw_index = core_head + core_bias;
	int trans_i = raw_index / width - 2;
	int trans_j = raw_index % width - 2;
	int pixel_val = CUDA_GetPixelVal(img, width, height, trans_i, trans_j);
	int cache_index = blockDim.x * threadIdx.y + threadIdx.x;
	cache[cache_index] = pixel_val;
	__syncthreads();

	/*gauss filter*/
	CUDA_Gaussian(cache, blockDim.x, blockDim.y, cache_index, gauss);
	__syncthreads();

	/*sobel filter*/
	CUDA_Sobel(gauss, blockDim.x, blockDim.y, cache_index, sobel);

	/*cute edge*/
	if (threadIdx.y <= 1 || threadIdx.y >= blockDim.y - 2 || 
		threadIdx.x <= 1 || threadIdx.x >= blockDim.x - 2)
		return;
	int raw_head = blockIdx.y * SPLIT_SIZE_X * SPLIT_SIZE_Y * gridDim.x + (threadIdx.y - 2) * SPLIT_SIZE_X * gridDim.x;
	int col_bias = blockIdx.x * SPLIT_SIZE_X + (threadIdx.x - 2);
	int new_id = raw_head + col_bias;

	/*store result*/
	output[new_id] = sobel[cache_index];
}


__device__ void CUDA_Gaussian(unsigned char* img, int width, int height, int idx, unsigned char* output)
{
	int new_pixel_value = 0;
	int i = idx / width;
	int j = idx % width;
	new_pixel_value = CUDA_GetPixelVal(img, width, height, i - 1, j - 1) * 0.07511 +
					  CUDA_GetPixelVal(img, width, height, i - 1, j    ) * 0.12384 +
					  CUDA_GetPixelVal(img, width, height, i - 1, j + 1) * 0.07511 +
					  CUDA_GetPixelVal(img, width, height, i    , j - 1) * 0.12384 +
					  CUDA_GetPixelVal(img, width, height, i    , j    ) * 0.20418 +
					  CUDA_GetPixelVal(img, width, height, i    , j + 1) * 0.12384 +
					  CUDA_GetPixelVal(img, width, height, i + 1, j - 1) * 0.07511 +
					  CUDA_GetPixelVal(img, width, height, i + 1, j    ) * 0.12384 +
					  CUDA_GetPixelVal(img, width, height, i + 1, j + 1) * 0.07511;
	output[idx] = new_pixel_value;
}

__device__ void CUDA_Sobel(unsigned char* img, int width, int height, int idx, unsigned char* output)
{
	int sobel_x = 0;
	int sobel_y = 0;
	int sobel = 0;
	int i = idx / width;
	int j = idx % width;
	sobel_x = CUDA_GetPixelVal(img, width, height, i - 1, j - 1) * (1) +
			  CUDA_GetPixelVal(img, width, height, i - 1, j    ) * (2) +
			  CUDA_GetPixelVal(img, width, height, i - 1, j + 1) * (1) +
			  CUDA_GetPixelVal(img, width, height, i + 1, j - 1) * (-1) +
			  CUDA_GetPixelVal(img, width, height, i + 1, j    ) * (-2) +
			  CUDA_GetPixelVal(img, width, height, i + 1, j + 1) * (-1);
	sobel_y = CUDA_GetPixelVal(img, width, height, i - 1, j - 1) * (-1) +
		      CUDA_GetPixelVal(img, width, height, i - 1, j + 1) * (1) +
		      CUDA_GetPixelVal(img, width, height, i    , j - 1) * (-2) +
		      CUDA_GetPixelVal(img, width, height, i    , j + 1) * (2) +
		      CUDA_GetPixelVal(img, width, height, i + 1, j - 1) * (-1) +
		      CUDA_GetPixelVal(img, width, height, i + 1, j + 1) * (1);
	sobel = sqrtf((float)(sobel_x * sobel_x + sobel_y * sobel_y));
	if(sobel < 255)
		output[idx] = sobel;
	else
		output[idx] = 255;

	if (i <=1 || i >= height - 2)
		output[idx] = 0;
	if(j <= 1 || j >= width - 2)
		output[idx] = 0;
}

__device__ unsigned char CUDA_GetPixelVal(unsigned char* img, int width, int height, int i, int j)
{
	if (i >= height || i < 0)
		return 0;
	else if (j >= width || j < 0)
		return 0;
	return *(img + i * width + j);
}
