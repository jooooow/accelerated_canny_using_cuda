#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <math.h>
//#include <device_launch_parameters.h>
//#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_timer.h>
//#include <device_functions.h>

#define IS_NOT_EDGE(a) (a < min_val)
#define IS_STRONG_EDGE(a) (a >= max_val)
#define IS_WEAK_EDGE(a)   (a >= min_val && a < max_val)

using namespace cv;

#define SPLIT_SIZE_X 32
#define SPLIT_SIZE_Y 24
#define LINK_SIZE_X  16
#define LINK_SIZE_Y  12
#define BLOCK_SIZE_X 36
#define BLOCK_SIZE_Y 28

/*canny using cuda*/
void CUDA_Canny();
__global__ void CUDA_GaussianAndSobel(unsigned char* img, int width, int height, unsigned char* output_sobel, short* output_gradient);
__device__ void CUDA_Gaussian(unsigned char* img, int width, int height, int idx, unsigned char* output);
__device__ void CUDA_Sobel(unsigned char* img, int width, int height, int idx, unsigned char* output_sobel, short* gradient);
__global__ void CUDA_NonMaxSuppress(unsigned char* sobel, int width, int height, short* gradient, unsigned char* output);
__global__ void CUDA_DoubleThreshold2(unsigned char* sobel, int width, int height, int min_val, int max_val, unsigned char* canny);
__device__ void CUDA_SubDoubleThreshold(unsigned char* sobel, int width, int height, int min_val, int max_val, unsigned int* weak_stack, unsigned int* stack_top, unsigned char* output, unsigned char*     visited);
__device__ void CUDA_IsWeakEdge(unsigned char* sobel, int width, int height, int min_val, int max_val, int i, int j, unsigned int* stack, unsigned int* top, unsigned char* output, unsigned char* visited);
__device__ unsigned char CUDA_GetPixelVal(unsigned char* img, int width, int height, int i, int j);
__device__ short GetGradientDirection(int sobel_x, int sobel_y);


void DisplayGradient(short* gradient, int width, int height);
unsigned char GetPixelVal(unsigned char* img, int width, int height, int i, int j);
void NonMaxSuppress(unsigned char* sobel, int width, int height, short* gradient, unsigned char* output);
void DoubleThreshold(unsigned char* sobel, int width, int height, int min_val, int max_val, unsigned char* output);
void IsWeakEdge(unsigned char* sobel, int width, int height, int min_val, int max_val, int i, int j, unsigned short* stack, unsigned short* top, unsigned char* output);
double GetTime()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(void)
{
	printf("CANNY_CUDA\n");
	CUDA_Canny();
	//system("pause");
	return 0;
}

void CUDA_Canny()
{
	int width = 640;
	int height = 480;
	dim3 block_size_extended(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 block_size_normal(SPLIT_SIZE_X, SPLIT_SIZE_Y);
	dim3 block_size_link(LINK_SIZE_X, LINK_SIZE_Y);
	dim3 grid_size(width / SPLIT_SIZE_X, height / SPLIT_SIZE_Y);
	dim3 grid_size_link(width / LINK_SIZE_X, height / LINK_SIZE_Y);
	Mat img_src, img_sobel, img_gradient, img_canny;

	VideoCapture camera(0);

	/*cpu memory*/
	unsigned char* cpu_sobel = new unsigned char[width * height];
	short* cpu_gradient = new short[width * height];
	unsigned char* cpu_canny   = new unsigned char[width * height];

	/*gpu memory*/
	unsigned char* gpu_img;
	cudaMalloc(&gpu_img, width * height * sizeof(unsigned char));
	unsigned char* gpu_sobel;
	cudaMalloc(&gpu_sobel, width * height * sizeof(unsigned char));
	short* gpu_gradient;
	cudaMalloc(&gpu_gradient, width * height * sizeof(short));
	unsigned char* gpu_canny;
	cudaMalloc(&gpu_canny, width * height * sizeof(unsigned char));
	double start_time, end_time;
	while (1)
	{
		camera >> img_src;
		//img_src = imread("F:/img_src/15.jpg");
		resize(img_src, img_src, Size(width, height), 0, 0);
		cvtColor(img_src, img_src, CV_BGR2GRAY);
		//imshow("img_src", img_src);

		start_time = GetTime();

		/*1.copy to gpu memory*/
		cudaMemcpy(gpu_img, img_src.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

		/*2.gauss filter*/
		CUDA_GaussianAndSobel << <grid_size, block_size_extended >> > (gpu_img, width, height, gpu_sobel, gpu_gradient);
		cudaDeviceSynchronize();

		/*3.none max suppress*/
		CUDA_NonMaxSuppress << <grid_size, block_size_normal >> > (gpu_sobel, width, height, gpu_gradient, gpu_sobel);
		cudaDeviceSynchronize();
		
		/*4.double threshold*/
		CUDA_DoubleThreshold2 << <grid_size_link, dim3(1,1) >> > (gpu_sobel, width, height, 50, 90, gpu_canny);

		/*copy to cpu memory*/
		cudaMemcpy(cpu_sobel, gpu_canny, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		//DoubleThreshold(cpu_sobel, width, height, 50, 90, cpu_canny);

		//cudaMemcpy(cpu_gradient, gpu_gradient, width * height * sizeof(short), cudaMemcpyDeviceToHost);
		//NonMaxSuppress(cpu_sobel, width, height, cpu_gradient, cpu_sobel);

		end_time = GetTime();
		printf("elapse : %.2fms\n", (end_time - start_time)*1000);

		img_canny = Mat(Size(width, height), CV_8UC1, cpu_sobel);
		resize(img_canny, img_canny, Size(640, 480), 0, 0);
		imshow("canny", img_canny);

		if ('q' == waitKey(1))
		{
			destroyAllWindows();
			free(cpu_sobel);
			cpu_sobel = NULL;
			free(cpu_canny);
			cpu_canny = NULL;
			free(cpu_gradient);
			cpu_gradient = NULL;
			cudaFree(gpu_img);
			cudaFree(gpu_sobel);
			cudaFree(gpu_gradient);
			cudaFree(gpu_canny);
			break;
		}
	}
}

__global__ void CUDA_GaussianAndSobel(unsigned char* img, int width, int height, unsigned char* output_sobel, short* output_gradient)
{
	__shared__ unsigned char cache[(BLOCK_SIZE_X) * (BLOCK_SIZE_Y)];
	__shared__ unsigned char gauss[(BLOCK_SIZE_X) * (BLOCK_SIZE_Y)];
	__shared__ unsigned char sobel[(BLOCK_SIZE_X) * (BLOCK_SIZE_Y)];
	short gradient = 0;

	/*alloct img to cache*/
	int raw_index = SPLIT_SIZE_X * SPLIT_SIZE_Y * blockIdx.y * gridDim.x + blockIdx.x * SPLIT_SIZE_X + SPLIT_SIZE_X * gridDim.x * threadIdx.y + threadIdx.x;
	int pixel_val = CUDA_GetPixelVal(img, width, height, raw_index / width - 2, raw_index % width - 2);
	int cache_index = blockDim.x * threadIdx.y + threadIdx.x;
	cache[cache_index] = pixel_val;
	__syncthreads();

	/*gauss filter*/
	CUDA_Gaussian(cache, blockDim.x, blockDim.y, cache_index, gauss);
	__syncthreads();

	/*sobel filter*/
	CUDA_Sobel(gauss, blockDim.x, blockDim.y, cache_index, sobel, &gradient);

	/*cute edge*/
	if (threadIdx.y <= 1 || threadIdx.y >= blockDim.y - 2 || 
		threadIdx.x <= 1 || threadIdx.x >= blockDim.x - 2)
		return;
	int new_id = blockIdx.y * SPLIT_SIZE_X * SPLIT_SIZE_Y * gridDim.x + (threadIdx.y - 2) * SPLIT_SIZE_X * gridDim.x + blockIdx.x * SPLIT_SIZE_X + (threadIdx.x - 2);

	/*store result*/
	output_gradient[new_id] = gradient;
	output_sobel[new_id] = sobel[cache_index];
}


__device__ void CUDA_Gaussian(unsigned char* img, int width, int height, int idx, unsigned char* output)
{
	int new_pixel_value = 0;
	new_pixel_value = CUDA_GetPixelVal(img, width, height, threadIdx.y - 1, threadIdx.x - 1) * 0.07511 +
					  CUDA_GetPixelVal(img, width, height, threadIdx.y - 1, threadIdx.x    ) * 0.12384 +
					  CUDA_GetPixelVal(img, width, height, threadIdx.y - 1, threadIdx.x + 1) * 0.07511 +
					  CUDA_GetPixelVal(img, width, height, threadIdx.y    , threadIdx.x - 1) * 0.12384 +
					  CUDA_GetPixelVal(img, width, height, threadIdx.y    , threadIdx.x    ) * 0.20418 +
					  CUDA_GetPixelVal(img, width, height, threadIdx.y    , threadIdx.x + 1) * 0.12384 +
					  CUDA_GetPixelVal(img, width, height, threadIdx.y + 1, threadIdx.x - 1) * 0.07511 +
					  CUDA_GetPixelVal(img, width, height, threadIdx.y + 1, threadIdx.x    ) * 0.12384 +
					  CUDA_GetPixelVal(img, width, height, threadIdx.y + 1, threadIdx.x + 1) * 0.07511;
	output[idx] = new_pixel_value;
}

__device__ void CUDA_Sobel(unsigned char* img, int width, int height, int idx, unsigned char* output_sobel, short* gradient)
{
	int sobel_x = 0;
	int sobel_y = 0;
	int sobel = 0;
	sobel_x = CUDA_GetPixelVal(img, width, height, threadIdx.y - 1, threadIdx.x - 1) * (1) +
			  CUDA_GetPixelVal(img, width, height, threadIdx.y - 1, threadIdx.x    ) * (2) +
			  CUDA_GetPixelVal(img, width, height, threadIdx.y - 1, threadIdx.x + 1) * (1) +
			  CUDA_GetPixelVal(img, width, height, threadIdx.y + 1, threadIdx.x - 1) * (-1) +
			  CUDA_GetPixelVal(img, width, height, threadIdx.y + 1, threadIdx.x    ) * (-2) +
			  CUDA_GetPixelVal(img, width, height, threadIdx.y + 1, threadIdx.x + 1) * (-1);
	sobel_y = CUDA_GetPixelVal(img, width, height, threadIdx.y - 1, threadIdx.x - 1) * (-1) +
		      CUDA_GetPixelVal(img, width, height, threadIdx.y - 1, threadIdx.x + 1) * (1) +
		      CUDA_GetPixelVal(img, width, height, threadIdx.y    , threadIdx.x - 1) * (-2) +
		      CUDA_GetPixelVal(img, width, height, threadIdx.y    , threadIdx.x + 1) * (2) +
		      CUDA_GetPixelVal(img, width, height, threadIdx.y + 1, threadIdx.x - 1) * (-1) +
		      CUDA_GetPixelVal(img, width, height, threadIdx.y + 1, threadIdx.x + 1) * (1);
	sobel = sqrtf((float)(sobel_x * sobel_x + sobel_y * sobel_y));
	sobel = sobel > 255 ? 255 : sobel;

	output_sobel[idx] = sobel;
	*gradient = GetGradientDirection(sobel_x, sobel_y);
}

__global__ void CUDA_NonMaxSuppress(unsigned char* sobel, int width, int height, short* gradient, unsigned char* output)
{
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if (id >= width * height)
		return;
	int i = id / width;
	int j = id % width;
	float weight = 0;
	int g0, g1, g2, g3;
	int temp_gradient = gradient[id] < 0 ? gradient[id] + 180 : gradient[id];
	if (temp_gradient >= 0 && temp_gradient < 45)
	{
		weight = temp_gradient / 45.0;
		g0 = CUDA_GetPixelVal(sobel, width, height, i    , j + 1);
		g1 = CUDA_GetPixelVal(sobel, width, height, i - 1, j + 1);
		g2 = CUDA_GetPixelVal(sobel, width, height, i    , j - 1);
		g3 = CUDA_GetPixelVal(sobel, width, height, i + 1, j - 1);
	}
	else if (temp_gradient >= 45 && temp_gradient < 90)
	{
		weight = (90 - temp_gradient) / 45.0;
		g0 = CUDA_GetPixelVal(sobel, width, height, i - 1, j    );
		g1 = CUDA_GetPixelVal(sobel, width, height, i - 1, j + 1);
		g2 = CUDA_GetPixelVal(sobel, width, height, i + 1, j    );
		g3 = CUDA_GetPixelVal(sobel, width, height, i + 1, j - 1);
	}
	else if (temp_gradient >= 90 && temp_gradient < 135)
	{
		weight = (temp_gradient - 90) / 45.0;
		g0 = CUDA_GetPixelVal(sobel, width, height, i - 1, j    );
		g1 = CUDA_GetPixelVal(sobel, width, height, i - 1, j - 1);
		g2 = CUDA_GetPixelVal(sobel, width, height, i + 1, j    );
		g3 = CUDA_GetPixelVal(sobel, width, height, i + 1, j + 1);
	}
	else if (temp_gradient >= 135 && temp_gradient <= 180)
	{
		weight = (180 - temp_gradient) / 45.0;
		g0 = CUDA_GetPixelVal(sobel, width, height, i    , j - 1);
		g1 = CUDA_GetPixelVal(sobel, width, height, i - 1, j - 1);
		g2 = CUDA_GetPixelVal(sobel, width, height, i    , j + 1);
		g3 = CUDA_GetPixelVal(sobel, width, height, i + 1, j + 1);
	}
	int dot1 = g0 * (1 - weight) + g1 * weight;
	int dot2 = g2 * (1 - weight) + g3 * weight;
	if (sobel[id] >= dot1 && sobel[id] >= dot2)
		output[id] = sobel[id];
	else
		output[id] = 0;
}


__global__ void CUDA_DoubleThreshold2(unsigned char* sobel, int width, int height, int min_val, int max_val, unsigned char* canny)
{
	__shared__ unsigned char cache[LINK_SIZE_X * LINK_SIZE_Y];
	__shared__ unsigned char output[LINK_SIZE_X * LINK_SIZE_Y];
	__shared__ unsigned int weak_stack[LINK_SIZE_X * LINK_SIZE_Y];
	__shared__ unsigned char visited[LINK_SIZE_X * LINK_SIZE_Y];
	unsigned int stack_top = 0;
	memset(visited, 0, LINK_SIZE_X * LINK_SIZE_Y);

	int raw_index = LINK_SIZE_X * LINK_SIZE_Y * blockIdx.y * gridDim.x + blockIdx.x * LINK_SIZE_X + LINK_SIZE_X * gridDim.x * threadIdx.y + threadIdx.x;
	for (int i = 0; i < LINK_SIZE_Y; i++)
	{
		for (int j = 0; j < LINK_SIZE_X; j++)
		{
			cache[i * LINK_SIZE_X + j] = CUDA_GetPixelVal(sobel, width, height, raw_index / width + i, raw_index % width + j);
		}
	}

	CUDA_SubDoubleThreshold(cache, LINK_SIZE_X, LINK_SIZE_Y, min_val, max_val, weak_stack, &stack_top, output, visited);

	for (int i = 0; i < LINK_SIZE_Y; i++)
	{
		for (int j = 0; j < LINK_SIZE_X; j++)
		{
			int new_id = blockIdx.y * LINK_SIZE_X * LINK_SIZE_Y * gridDim.x + i * LINK_SIZE_X * gridDim.x + blockIdx.x * LINK_SIZE_X + j;
			canny[new_id] = output[i * LINK_SIZE_X + j];
		}
	}
}

__device__ void CUDA_SubDoubleThreshold(unsigned char* sobel, int width, int height, int min_val, int max_val, unsigned int* weak_stack, unsigned int* stack_top, unsigned char* output, unsigned char* visited)
{
	unsigned short center_index = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (IS_STRONG_EDGE(CUDA_GetPixelVal(sobel, width, height, i, j)))
			{
				CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, i, j, weak_stack, stack_top, output, visited);
				while ((*stack_top) > 0)
				{
					center_index = weak_stack[(*stack_top) - 1];
					(*stack_top)--;
					CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width - 1, center_index % width - 1, weak_stack, stack_top, output, visited);
					CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width - 1, center_index % width    , weak_stack, stack_top, output, visited);
					CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width - 1, center_index % width + 1, weak_stack, stack_top, output, visited);
					CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width    , center_index % width - 1, weak_stack, stack_top, output, visited);
					CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width    , center_index % width + 1, weak_stack, stack_top, output, visited);
					CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width + 1, center_index % width - 1, weak_stack, stack_top, output, visited);
					CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width + 1, center_index % width    , weak_stack, stack_top, output, visited);
					CUDA_IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width + 1, center_index % width + 1, weak_stack, stack_top, output, visited);
					//__syncthreads();
				}
			}
			else if (IS_NOT_EDGE(CUDA_GetPixelVal(sobel, width, height, i, j)))
			{
				output[i * width + j] = 0;
			}
			else
			{
				if(visited[i * width + j] == 0)
					output[i * width + j] = 0;
			}
		}
	}
}

__device__ void CUDA_IsWeakEdge(unsigned char* sobel, int width, int height, int min_val, int max_val, int i, int j, unsigned int* stack, unsigned int* top, unsigned char* output, unsigned char* visited)
{
	if (i < 0 || i >= height)
		return;
	if (j < 0 || j >= width)
		return;
	if (visited[i * width + j] == 1)
		return;
	visited[i * width + j] = 1;
	if (IS_STRONG_EDGE(CUDA_GetPixelVal(sobel, width, height, i, j)))
	{
		output[i * width + j] = 255;
		stack[*top] = i * width + j;
		(*top)++;
	}
	else if(IS_WEAK_EDGE(CUDA_GetPixelVal(sobel, width, height, i, j)))
	{
		output[i * width + j] = 255;
		stack[*top] = i * width + j;
		(*top)++;
	}
	else
	{
		output[i * width + j] = 0;
	}
}

__device__ unsigned char CUDA_GetPixelVal(unsigned char* img, int width, int height, int i, int j)
{
	if (i >= height || i < 0)
		return 0;
	else if (j >= width || j < 0)
		return 0;
	return *(img + i * width + j);
}

__device__ short GetGradientDirection(int sobel_x, int sobel_y)
{
	short gradient = (atan2f(sobel_x, sobel_y) / 3.1415926 * 180.0);
	//gradient = gradient < 0 ? gradient + 180 : gradient;
	return gradient;
}

void DisplayGradient(short* gradient, int width, int height)
{
	Mat img = Mat::zeros(Size(width, height), CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (abs(*(gradient + i * width + j)) >= 0 && abs(*(gradient + i * width + j)) < 45)
			{
				img.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
			}
			else if (abs(*(gradient + i * width + j)) >= 45 && abs(*(gradient + i * width + j)) < 90)
			{
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			}
			else if (abs(*(gradient + i * width + j)) >= 90 && abs(*(gradient + i * width + j)) < 135)
			{
				img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			}
			else if (abs(*(gradient + i * width + j)) >= 135 && abs(*(gradient + i * width + j)) <= 180)
			{
				img.at<Vec3b>(i, j) = Vec3b(128, 128, 128);
			}
		}
	}
	imshow("gradient", img);
}

unsigned char GetPixelVal(unsigned char* img, int width, int height, int i, int j)
{
	if (i >= height || i < 0)
		return 0;
	else if (j >= width || j < 0)
		return 0;
	return *(img + i * width + j);
}

void NonMaxSuppress(unsigned char* sobel, int width, int height, short* gradient, unsigned char* output)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int id = i * width + j;
			float weight = 0;
			int g0, g1, g2, g3;
			int temp_gradient = gradient[id] < 0 ? gradient[id] + 180 : gradient[id];
			if (temp_gradient >= 0 && temp_gradient < 45)
			{
				weight = temp_gradient / 45.0;
				g0 = GetPixelVal(sobel, width, height, i, j + 1);
				g1 = GetPixelVal(sobel, width, height, i - 1, j + 1);
				g2 = GetPixelVal(sobel, width, height, i, j - 1);
				g3 = GetPixelVal(sobel, width, height, i + 1, j - 1);
			}
			else if (temp_gradient >= 45 && temp_gradient < 90)
			{
				weight = (90 - temp_gradient) / 45.0;
				g0 = GetPixelVal(sobel, width, height, i - 1, j);
				g1 = GetPixelVal(sobel, width, height, i - 1, j + 1);
				g2 = GetPixelVal(sobel, width, height, i + 1, j);
				g3 = GetPixelVal(sobel, width, height, i + 1, j - 1);
			}
			else if (temp_gradient >= 90 && temp_gradient < 135)
			{
				weight = (temp_gradient - 90) / 45.0;
				g0 = GetPixelVal(sobel, width, height, i - 1, j);
				g1 = GetPixelVal(sobel, width, height, i - 1, j - 1);
				g2 = GetPixelVal(sobel, width, height, i + 1, j);
				g3 = GetPixelVal(sobel, width, height, i + 1, j + 1);
			}
			else if (temp_gradient >= 135 && temp_gradient <= 180)
			{
				weight = (180 - temp_gradient) / 45.0;
				g0 = GetPixelVal(sobel, width, height, i, j - 1);
				g1 = GetPixelVal(sobel, width, height, i - 1, j - 1);
				g2 = GetPixelVal(sobel, width, height, i, j + 1);
				g3 = GetPixelVal(sobel, width, height, i + 1, j + 1);
			}
			int dot1 = g0 * (1 - weight) + g1 * weight;
			int dot2 = g2 * (1 - weight) + g3 * weight;
			if (sobel[id] > dot1 && sobel[id] > dot2)
				output[id] = sobel[id];
			else
				output[id] = 0;
		}
	}
}

void DoubleThreshold(unsigned char* sobel, int width, int height, int min_val, int max_val, unsigned char* output)
{
	unsigned short* weak_stack = new unsigned short[width * height];
	unsigned short stack_top = 0;
	unsigned short center_index = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (IS_STRONG_EDGE(GetPixelVal(sobel, width, height, i, j)))
			{
				stack_top = 0;
				IsWeakEdge(sobel, width, height, min_val, max_val, i, j, weak_stack, &stack_top, output);
				while (stack_top > 0)
				{
					center_index = weak_stack[stack_top - 1];
					stack_top--;
					IsWeakEdge(sobel, width, height, min_val, max_val, i - 1, j - 1, weak_stack, &stack_top, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, i - 1, j    , weak_stack, &stack_top, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, i - 1, j + 1, weak_stack, &stack_top, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, i    , j - 1, weak_stack, &stack_top, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, i    , j + 1, weak_stack, &stack_top, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, i + 1, j - 1, weak_stack, &stack_top, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, i + 1, j    , weak_stack, &stack_top, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, i + 1, j + 1, weak_stack, &stack_top, output);
				}
			}
			else if (IS_NOT_EDGE(GetPixelVal(sobel, width, height, i, j)))
			{
				output[i * width + j] = 0;
			}
		}
	}

	delete[] weak_stack;
	weak_stack = nullptr;
}

void IsWeakEdge(unsigned char* sobel, int width, int height, int min_val, int max_val, int i, int j, unsigned short* stack, unsigned short* top, unsigned char* output)
{
	if (IS_WEAK_EDGE(GetPixelVal(sobel, width, height, i, j)) ||
		IS_STRONG_EDGE(GetPixelVal(sobel, width, height, i, j)))
	{
		output[i * width + j] = 255;
		stack[*top] = i * width + j;
		*top++;
	}
	else
	{
		output[i * width + j] = 0;
	}
}



