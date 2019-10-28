#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void GenerateGaussKernel(int size, float sigma, float* kernel);
unsigned char GetPixelVal(unsigned char* img, int img_height, int img_width, int i, int j);
void Gauss(unsigned char* img, int img_width, int img_height, float* kernel, int kernel_size, unsigned char* output);
void Sobel(unsigned char* img, int img_width, int img_height, short* sobel_x, short* sobel_y, unsigned char* output);
void NoneMaxSuppress(unsigned char* sobel, int sobel_width, int sobel_height, short* sobel_x, short* sobel_y, unsigned char* output);
void DoubleThreshold(unsigned char* sobel, int sobel_width, int sobel_height, unsigned char* canny);

__device__ unsigned char CUDA_GetPixelVal(unsigned char* img, int img_height, int img_width, int i, int j);
__global__ void CUDA_Gauss(unsigned char* img, int img_width, int img_height, float* kernel, int kernel_size, unsigned char* output);
__global__ void CUDA_Sobel(unsigned char* img, int img_width, int img_height, short* sobel_x, short* sobel_y, unsigned char* output);

int main(int argc, char** argv)
{
	int cpu_gpu = 1;
	//cout<<(int)memcmp(argv[1], "gpu", 3)<<" "<<(int)memcmp(argv[1], "cpu", 3)<<endl;

	cout<<"---canny kasoku!---"<<endl;
	
	int width = 1280;
	int height = 960;
	int gauss_kernel_size = 3;
	
	int thread_size = 1024;
	int block_size  = (width * height + thread_size - 1) / thread_size;
	
	/*****cpu memory*****/
	unsigned char* gauss = new unsigned char[width * height];

	float* gauss_kernel = new float[gauss_kernel_size * gauss_kernel_size];
	GenerateGaussKernel(gauss_kernel_size, 1, gauss_kernel);

	short* sobel_x = new short[width * height];
	short* sobel_y = new short[width * height];
	unsigned char* sobel = new unsigned char[width * height];

	/*****gpu memory*****/
	unsigned char* cuda_gray;
	cudaMalloc(&cuda_gray, width * height * sizeof(unsigned char));

	unsigned char* cuda_gauss;
	cudaMalloc(&cuda_gauss, width * height * sizeof(unsigned char));

	float* cuda_gauss_kernel;
	cudaMalloc(&cuda_gauss_kernel, width * height * sizeof(float));
	cudaMemcpy(cuda_gauss_kernel, gauss_kernel, gauss_kernel_size * gauss_kernel_size * sizeof(float), cudaMemcpyHostToDevice);

	short* cuda_sobel_x;
	cudaMalloc(&cuda_sobel_x, width * height * sizeof(short));
	
	short* cuda_sobel_y;
	cudaMalloc(&cuda_sobel_y, width * height * sizeof(short));
	
	unsigned char* cuda_sobel;
	cudaMalloc(&cuda_sobel, width * height * sizeof(unsigned char));
	
	while(1)
	{
		if(cpu_gpu == 0)
		{
			Mat img_src   = imread("/home/katsuto/Pictures/Wallpapers/timg.jpeg");
			Mat img_gray, img_gauss, img_sobel, img_canny;
			cvtColor(img_src, img_gray, CV_BGR2GRAY);
			
			resize(img_gray, img_gray, Size(width, height), 0, 0);
			imshow("img_gray", img_gray);
	
			Gauss(img_gray.data, width, height, gauss_kernel, gauss_kernel_size, gauss);
			//img_gauss = Mat(Size(width, height), CV_8UC1, gauss);
			//imshow("img_gauss", img_gauss);
		
			Sobel(gauss, width, height, sobel_x, sobel_y, sobel);
			img_sobel = Mat(Size(width, height), CV_8UC1, sobel);
			imshow("img_sobel", img_sobel);
		}
		else
		{
			/*read image*/
			Mat img_src   = imread("/home/katsuto/Pictures/Wallpapers/timg.jpeg");
			Mat img_gray, img_gauss, img_sobel, img_canny;
			cvtColor(img_src, img_gray, CV_BGR2GRAY);
			resize(img_gray, img_gray, Size(width, height), 0, 0);
			imshow("img_gray", img_gray);
			
			/*load into gpu*/
			cudaMemcpy(cuda_gray, img_gray.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
			
			/*gauss filter*/
			CUDA_Gauss<<<block_size, thread_size>>>(cuda_gray, width, height, cuda_gauss_kernel, gauss_kernel_size, cuda_gauss);
			//cudaMemcpy(gauss, cuda_gauss, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			//img_gauss = Mat(Size(width, height), CV_8UC1, gauss);
			//imshow("img_gauss_cuda", img_gauss);
		
			/*sobel edge detection*/
			CUDA_Sobel<<<block_size, thread_size>>>(cuda_gauss, width, height, cuda_sobel_x, cuda_sobel_y, cuda_sobel);
			cudaMemcpy(sobel, cuda_sobel, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			img_sobel = Mat(Size(width, height), CV_8UC1, sobel);
			imshow("img_sobel_gpu", img_sobel);

		}
		waitKey(0);
		break;
	}

	cudaFree(cuda_gray);
	cudaFree(cuda_gauss);
	cudaFree(cuda_gauss_kernel);
	cudaFree(cuda_sobel_x);
	cudaFree(cuda_sobel_y);
	cudaFree(cuda_sobel);

	delete[] gauss;
	gauss = nullptr;
	delete[] gauss_kernel;
	gauss_kernel = nullptr;
	delete[] sobel_x;
	sobel_x = nullptr;

	delete[] sobel_y;
	sobel_y = nullptr;
	delete[] sobel;
	sobel = nullptr;

	return 0;
}

unsigned char GetPixelVal(unsigned char* img, int img_height, int img_width, int i, int j)
{
	if(i >= img_height || i < 0)
		return 0;
	if(j >= img_width  || j < 0)
		return 0;
	return *(img + i * img_width + j);	
}

__device__ unsigned char CUDA_GetPixelVal(unsigned char* img, int img_height, int img_width, int i, int j)
{	
	if(i >= img_height || i < 0)
		return 0;	
	else if(j >= img_width  || j < 0)
		return 0;
	return *(img + i * img_width + j);	
}

__global__ void CUDA_Gauss(unsigned char* img, int img_width, int img_height, float* kernel, int kernel_size, unsigned char* output)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i  = id / img_width;
	int j  = id % img_width;
	if(id < img_width * img_height)
	{
		int new_pixel_value  = 0;
		int half_kernel_size = kernel_size / 2;
		for(int k = 0; k < kernel_size; k++)
		{
			for(int m = 0; m < kernel_size; m++)
			{
				new_pixel_value += (*(kernel + k * kernel_size + m)) * CUDA_GetPixelVal(img, img_height, img_width, i + k - half_kernel_size, j + m - half_kernel_size);
				__syncthreads();
			}
		}	
		*(output + i * img_width + j) = new_pixel_value;
	}
}

__global__ void CUDA_Sobel(unsigned char* img, int img_width, int img_height, short* sobel_x, short* sobel_y, unsigned char* output)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i = id / img_width;
	int j = id % img_width;
	
	if(id >= img_width * img_height)
		return;

	*(sobel_x + i * img_width + j) = CUDA_GetPixelVal(img, img_height, img_width, i-1, j-1) * (1) +
			           		  		 CUDA_GetPixelVal(img, img_height, img_width, i-1, j  ) * (2) +
	                        	     CUDA_GetPixelVal(img, img_height, img_width, i-1, j+1) * (1) +
	                           	     CUDA_GetPixelVal(img, img_height, img_width, i  , j-1) * (0) +
	                             	 CUDA_GetPixelVal(img, img_height, img_width, i  , j  ) * (0) +
	                                 CUDA_GetPixelVal(img, img_height, img_width, i  , j+1) * (0) +
	                                 CUDA_GetPixelVal(img, img_height, img_width, i+1, j-1) * (-1) +
	                                 CUDA_GetPixelVal(img, img_height, img_width, i+1, j  ) * (-2) +
	                                 CUDA_GetPixelVal(img, img_height, img_width, i+1, j+1) * (-1);


	*(sobel_y + i * img_width + j) = CUDA_GetPixelVal(img, img_height, img_width, i-1, j-1) * (-1) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i-1, j  ) * (0) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i-1, j+1) * (1) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i  , j-1) * (-2) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i  , j  ) * (0) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i  , j+1) * (2) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i+1, j-1) * (-1) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i+1, j  ) * (0) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i+1, j+1) * (1);

	float val =sqrt(pow(*(sobel_x + i * img_width + j), 2) + pow(*(sobel_y + i * img_width + j), 2));
	if(val > 255)
		*(output + i * img_width + j) = 255;
	else
		*(output + i * img_width + j) = val;	
}

void GenerateGaussKernel(int size, float sigma, float* kernel)
{
	int center = size / 2;
	float sum = 0.0f;
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			*(kernel + i * size + j) = (float)1 / (2 * 3.1415926 * sigma * sigma) * exp(-(pow(i - center, 2) + pow(j - center, 2)) / (2 * pow(sigma, 2)));
			sum += *(kernel + i * size + j);
		}
	}	
	cout<<"gauss kenel : "<<endl;
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			*(kernel + i * size + j) /= sum;
			cout<<*(kernel + i * size + j)<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
}

void Gauss(unsigned char* img, int img_width, int img_height, float* kernel, int kernel_size, unsigned char* output)
{
	for(int i = 0; i < img_height; i++)
	{
		for(int j = 0; j < img_width; j++)
		{
			int new_pixel_value  = 0;
			int half_kernel_size = kernel_size / 2;
			for(int k = 0; k < kernel_size; k++)
			{
				for(int m = 0; m < kernel_size; m++)
				{
					new_pixel_value += GetPixelVal(img, img_height, img_width, i + k - half_kernel_size, j + m - half_kernel_size) * (*(kernel + k * kernel_size + m));
				}
			}
			*(output + i * img_width + j) = new_pixel_value;
		}
	}
}

void Sobel(unsigned char* img, int img_width, int img_height, short* sobel_x, short* sobel_y, unsigned char* output)
{
	float sobel_filter_x[9] = {1,2,1,0,0,0,-1,-2,-1};
	float sobel_filter_y[9] = {-1,0,1,-2,0,2,-1,0,1};
	
	for(int i = 0; i < img_height; i++)
	{
		for(int j = 0; j < img_width; j++)
		{
			*(sobel_x + i * img_width + j) = GetPixelVal(img, img_height, img_width, i-1, j-1) * sobel_filter_x[0] +
					           		  		 GetPixelVal(img, img_height, img_width, i-1, j  ) * sobel_filter_x[1] +
			                        	     GetPixelVal(img, img_height, img_width, i-1, j+1) * sobel_filter_x[2] +
			                           	     GetPixelVal(img, img_height, img_width, i  , j-1) * sobel_filter_x[3] +
			                             	 GetPixelVal(img, img_height, img_width, i  , j  ) * sobel_filter_x[4] +
			                                 GetPixelVal(img, img_height, img_width, i  , j+1) * sobel_filter_x[5] +
			                                 GetPixelVal(img, img_height, img_width, i+1, j-1) * sobel_filter_x[6] +
			                                 GetPixelVal(img, img_height, img_width, i+1, j  ) * sobel_filter_x[7] +
			                                 GetPixelVal(img, img_height, img_width, i+1, j+1) * sobel_filter_x[8];


			*(sobel_y + i * img_width + j) = GetPixelVal(img, img_height, img_width, i-1, j-1) * sobel_filter_y[0] +
				                             GetPixelVal(img, img_height, img_width, i-1, j  ) * sobel_filter_y[1] +
				                             GetPixelVal(img, img_height, img_width, i-1, j+1) * sobel_filter_y[2] +
				                             GetPixelVal(img, img_height, img_width, i  , j-1) * sobel_filter_y[3] +
				                             GetPixelVal(img, img_height, img_width, i  , j  ) * sobel_filter_y[4] +
				                             GetPixelVal(img, img_height, img_width, i  , j+1) * sobel_filter_y[5] +
				                             GetPixelVal(img, img_height, img_width, i+1, j-1) * sobel_filter_y[6] +
				                             GetPixelVal(img, img_height, img_width, i+1, j  ) * sobel_filter_y[7] +
				                             GetPixelVal(img, img_height, img_width, i+1, j+1) * sobel_filter_y[8];

			float val =sqrt(pow(*(sobel_x + i * img_width + j), 2) + pow(*(sobel_y + i * img_width + j), 2));
			if(val > 255)
				*(output + i * img_width + j) = 255;
			else	
				*(output + i * img_width + j) = val;
		}
	}
}

void NoneMaxSuppress(unsigned char* sobel, int sobel_width, int sobel_height, short* sobel_x, short* sobel_y, unsigned char* output)
{
	
}

void DoubleThreshold(unsigned char* sobel, int sobel_width, int sobel_height, unsigned char* canny)
{
	
}
