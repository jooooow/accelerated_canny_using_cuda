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

/*test1*/
void Test1();
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] * B[i];
}
void Addvec(float* A, float* B, float* C, int N)
{
	for (int i = 0; i < N; i++)
		C[i] = A[i] * B[i];
}

/*opencvtest*/
void OpencvTest();
void ConvCPU(unsigned char* img, float* kernel, int img_width, int img_height, int kernel_size, unsigned char* output);
__global__ void ConvGPU(unsigned char* img, float* kernel, int img_width, int img_height, int kernel_size, unsigned char* output);
__global__ void ReverseGPU(unsigned char* img, int img_width, int img_height);

/*cuda array sum*/
void GpuArraySum();
__global__ void ArraySum(int* arr, int* res, int size);

int main(void)
{
	//Test1();
	//OpencvTest();
	GpuArraySum();

	return 0;
}


void Test1()
{
	clock_t s_t, e_t;
	int N = 1920 * 1080 * 3;
	float* a = (float*)malloc(N * sizeof(float));
	float* b = (float*)malloc(N * sizeof(float));
	float* c = (float*)malloc(N * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * 0.01;
		c[i] = 0;
	}

	float* cuda_a;
	float* cuda_b;
	float* cuda_c;

	cudaMalloc(&cuda_a, N * sizeof(float));
	cudaMalloc(&cuda_b, N * sizeof(float));
	cudaMalloc(&cuda_c, N * sizeof(float));

	cudaMemcpy(cuda_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	int thread_cnt = 1024;
	int block_cnt = (thread_cnt + N - 1) / thread_cnt;

	StopWatchInterface * timer_cublas;  //****用来计算GPU核函数耗时
	sdkCreateTimer(&timer_cublas);		//****
	sdkStartTimer(&timer_cublas);		//****

	for (int i = 0; i < 100; i++)
	{
		VecAdd << <block_cnt, thread_cnt >> > (cuda_a, cuda_b, cuda_c, N);
		cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaThreadSynchronize();			//****
	sdkStopTimer(&timer_cublas);		//****
	double dSeconds = sdkGetTimerValue(&timer_cublas) / (1000.0f); //***

	printf("\ngpu_done %.3f\n\n", dSeconds);

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);


	LARGE_INTEGER litmp;	//####//用来计算cpu消耗时间
	LONGLONG qt1, qt2;		//####
	double dft, dff, dfm;

	QueryPerformanceFrequency(&litmp);	//####获得时钟频率
	dff = (double)litmp.QuadPart;			//####
	QueryPerformanceCounter(&litmp);	//####//获得初始值
	qt1 = litmp.QuadPart;

	for (int i = 0; i < 100; i++)
	{
		Addvec(a, b, c, N);
	}

	QueryPerformanceCounter(&litmp);	//####//获得终止值
	qt2 = litmp.QuadPart;					//####
	dfm = (double)(qt2 - qt1);				//####
	dft = dfm / dff;

	printf("\ncpu_done %.3f\n\n", dft);

	free(a);
	free(b);
	free(c);
}

void OpencvTest()
{
	int width  = 640;
	int height = 480;
	VideoCapture camera(1);
	Mat img_src, img_gray, img_sobel;

	float sobel_kernel_cpu[9] = { 1,1,1,0,0,0,-1,-1,-1 };
	unsigned char* sobel_cpu = new unsigned char[width * height];

	unsigned char* gray_gpu;
	unsigned char* sobel_gpu;
	float* sobel_kernel_gpu;
	cudaMalloc(&gray_gpu, width * height * sizeof(unsigned char));
	cudaMalloc(&sobel_gpu, width * height * sizeof(unsigned char));
	cudaMalloc(&sobel_kernel_gpu, 9 * sizeof(float));
	cudaMemcpy(sobel_kernel_gpu, sobel_kernel_cpu, 9 * sizeof(float), cudaMemcpyHostToDevice);

	while (1)
	{
		camera >> img_src;
		imshow("img_src", img_src);
		resize(img_src, img_src, Size(width, height), 0, 0);
		cvtColor(img_src, img_gray, CV_BGR2GRAY);
		imshow("img_gray", img_gray);
		
		/*ConvCPU(img_gray.data, sobel_kernel_cpu, width, height, 3, sobel_cpu);
		ConvCPU(img_gray.data, sobel_kernel_cpu, width, height, 3, sobel_cpu);
		ConvCPU(img_gray.data, sobel_kernel_cpu, width, height, 3, sobel_cpu);
		ConvCPU(img_gray.data, sobel_kernel_cpu, width, height, 3, sobel_cpu);
		img_sobel = Mat(Size(width, height), CV_8UC1, sobel_cpu);
		imshow("sobel_cpu", img_sobel);*/

		int thread_size = 1024;
		int block_size = (width * height + thread_size - 1) / thread_size;
		cudaMemcpy(gray_gpu, img_gray.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		ConvGPU<<<block_size, thread_size>>>(gray_gpu, sobel_kernel_gpu, width, height, 3, sobel_gpu);
		cudaMemcpy(sobel_cpu, sobel_gpu, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		img_sobel = Mat(Size(width, height), CV_8UC1, sobel_cpu);
		imshow("sobel_cpu", img_sobel);

		if ('q' == waitKey(1))
		{
			destroyAllWindows();
			cudaFree(gray_gpu);
			cudaFree(sobel_gpu);
			cudaFree(sobel_kernel_gpu);
			free(sobel_kernel_gpu);
			sobel_kernel_gpu = NULL;
			break;
		}
	}
}

void ConvCPU(unsigned char* img, float* kernel, int img_width, int img_height, int kernel_size, unsigned char* output)
{
	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			float conv_val = 0.0f;
			for (int k = 0; k < kernel_size; k++)
			{
				for (int m = 0; m < kernel_size; m++)
				{
					int pixel_i = k - kernel_size / 2 + i;
					int pixel_j = m - kernel_size / 2 + j;
					unsigned char pixel_val = 0;
					if (pixel_i < 0 || pixel_i >= img_height || pixel_j < 0 || pixel_j >= img_width)
					{

					}
					else
					{
						pixel_val = *(img + pixel_i * img_width + pixel_j);
					}
					int temp = pixel_val * (*(kernel + k * kernel_size + m));
					conv_val += temp;
				}
			}
			*(output + i * img_width + j) = fabs(conv_val);
		}
	}
}

__global__ void ConvGPU(unsigned char* img, float* kernel, int img_width, int img_height, int kernel_size, unsigned char* output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int i = x / img_width;
	int j = x % img_width;

	if (x <= img_width * img_height - 1)
	{
		float conv_val = 0.0f;
		for (int k = 0; k < kernel_size; k++)
		{
			for (int m = 0; m < kernel_size; m++)
			{
				int pixel_i = k - kernel_size / 2 + i;
				int pixel_j = m - kernel_size / 2 + j;
				unsigned char pixel_val = 0;
				if (pixel_i < 0 || pixel_i >= img_height || pixel_j < 0 || pixel_j >= img_width)
				{

				}
				else
				{
					pixel_val = *(img + pixel_i * img_width + pixel_j);
				}
				int temp = pixel_val * (*(kernel + k * kernel_size + m));
				conv_val += temp;
			}
		}
		output[x] = fabs(conv_val);
	}
}

__global__ void ReverseGPU(unsigned char* img, int img_width, int img_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x <= img_height * img_width - 1)
	{
		img[x] = 255 - img[x];
	}
}




void GpuArraySum()
{
	int arr[16];
	for (int i = 0; i < 16; i++)
	{
		arr[i] = i + 1;
	}

	int* cuda_arr;
	cudaMalloc(&cuda_arr, 16 * sizeof(int));
	cudaMemcpy(cuda_arr, arr, 16 * sizeof(int), cudaMemcpyHostToDevice);

	int* cuda_result;
	cudaMalloc(&cuda_result, 1 * sizeof(int));

	ArraySum<<<1, 16>>>(cuda_arr, cuda_result, 16);

	int result[1];
	cudaMemcpy(result, cuda_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);

	printf("%d\n", result[0]);

	cudaFree(cuda_arr);
	cudaFree(cuda_result);

	system("pause");
}

__global__ void ArraySum(int* arr, int* res, int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int data[16];
	data[x] = arr[x];
	__syncthreads();

	for (int i = size / 2; i > 0; i /= 2)
	{
		if (x < i)
		{
			data[x] += data[x + i];
		}
		__syncthreads();
	}
	res[0] = data[0];
}