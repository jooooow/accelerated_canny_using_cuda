#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#define CUDA_ERR_HANDLE(err) (CudaErrorHandler(err, __FILE__, __LINE__))

int CudaErrorHandler(cudaError_t err, const char* file, int line)
{
	if(err != cudaSuccess)
	{
		cout<<file<<" no."<<line<<" error : "<<cudaGetErrorString(err)<<endl;
		return 1;
	}
	return 0;
}

void GenerateGaussKernel(int size, float sigma, float* kernel);
unsigned char GetPixelVal(unsigned char* img, int img_height, int img_width, int i, int j);
void Gauss(unsigned char* img, int img_width, int img_height, float* kernel, int kernel_size, unsigned char* output);
void Sobel(unsigned char* img, int img_width, int img_height, short* sobel_x, short* sobel_y, unsigned char* output);
void NoneMaxSuppress(unsigned char* sobel, int sobel_width, int sobel_height, short* sobel_x, short* sobel_y, unsigned char* output);
void DoubleThreshold(unsigned char* sobel, int sobel_width, int sobel_height, int min_val, int max_val, unsigned char* canny);

__device__ unsigned char CUDA_GetPixelVal(unsigned char* img, int img_height, int img_width, int i, int j);
__global__ void CUDA_Gauss(unsigned char* img, int img_width, int img_height, float* kernel, int kernel_size, unsigned char* output);
__global__ void CUDA_Sobel(unsigned char* img, int img_width, int img_height, short* sobel_x, short* sobel_y, unsigned char* output);
__global__ void CUDA_NoneMaxSuppress(unsigned char* sobel, int sobel_width, int sobel_height, short* sobel_x, short* sobel_y, unsigned char* output);
__global__ void CUDA_DoubleThreshold(unsigned char* sobel, int sobel_width, int sobel_height, int min_val, int max_val, unsigned char* canny);

int main(int argc, char** argv)
{
	int cpu_gpu = 0;
	if(argc >= 2 && strcmp(argv[1], "gpu") == 0)
	{
		cout<<"---canny acceleration[GPU]!---"<<endl;
		cpu_gpu = 1;
	}
	else
	{
		cout<<"---canny acceleration[CPU]!---"<<endl;
	}
	
	int width = 640;
	int height = 480;
	int gauss_kernel_size = 3;
	
	int thread_size = width;
	int block_size  = (width * height + thread_size - 1) / thread_size;

	/*****cpu memory*****/
	unsigned char* gauss = new unsigned char[width * height];

	float* gauss_kernel = new float[gauss_kernel_size * gauss_kernel_size];
	GenerateGaussKernel(gauss_kernel_size, 1, gauss_kernel);

	short* sobel_x = new short[width * height];
	short* sobel_y = new short[width * height];
	unsigned char* sobel = new unsigned char[width * height];

	unsigned char* canny = new unsigned char[width * height];

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
	
	unsigned char* cuda_canny;
	cudaMalloc(&cuda_canny, width * height * sizeof(unsigned char));


	VideoCapture camera(0);
	//VideoCapture camera;
	//camera.open("/home/katsuto/Pictures/Wallpapers/video.MP4");
	Mat img_src;
 	//img_src	= imread("/home/katsuto/Pictures/Wallpapers/nvidia.jpg");
	Mat img_gray, img_gauss, img_sobel, img_canny;
	
	while(1)
	{
		if(cpu_gpu == 0)
		{
			camera >> img_src;
 	        //img_src	= imread("/home/katsuto/Pictures/Wallpapers/nvidia.jpg");
			cvtColor(img_src, img_gray, CV_BGR2GRAY);
			
			resize(img_gray, img_gray, Size(width, height), 0, 0);
			imshow("img_gray", img_gray);
	
			Gauss(img_gray.data, width, height, gauss_kernel, gauss_kernel_size, gauss);
			//img_gauss = Mat(Size(width, height), CV_8UC1, gauss);
			//imshow("img_gauss", img_gauss);
		
			Sobel(gauss, width, height, sobel_x, sobel_y, sobel);
			img_sobel = Mat(Size(width, height), CV_8UC1, sobel);
			imshow("img_sobel", img_sobel);
		
			NoneMaxSuppress(sobel, width, height, sobel_x, sobel_y, sobel);
			//img_sobel = Mat(Size(width, height), CV_8UC1, sobel);
			//imshow("img_suppress", img_sobel);

			DoubleThreshold(sobel, width, height, 30, 130, canny);
			img_canny = Mat(Size(width, height), CV_8UC1, canny);
			imshow("img_canny", img_canny);
		}
		else
		{
			/*read image*/
			cout<<"gpu"<<endl;
			camera >> img_src;
			resize(img_src, img_src, Size(width, height), 0, 0);
			cvtColor(img_src, img_gray, CV_BGR2GRAY);
			
			/*load into gpu*/
			if(CUDA_ERR_HANDLE(cudaMemcpy(cuda_gray, img_gray.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice)))
			{
				cout<<"memcpy fail1"<<endl;
				continue;
			}
			
			/*gauss filter*/
			CUDA_Gauss<<<block_size, thread_size>>>(cuda_gray, width, height, cuda_gauss_kernel, gauss_kernel_size, cuda_gauss);
			
			/*sobel edge detection*/
			CUDA_Sobel<<<block_size, thread_size>>>(cuda_gauss, width, height, cuda_sobel_x, cuda_sobel_y, cuda_sobel);
			
			/*none max suppress*/
			CUDA_NoneMaxSuppress<<<block_size, thread_size>>>(cuda_sobel, width, height, cuda_sobel_x, cuda_sobel_y, cuda_sobel);
			cudaDeviceSynchronize();

			/*double threshold*/
			CUDA_DoubleThreshold<<<block_size, thread_size>>>(cuda_sobel, width, height, 30, 130, cuda_canny);
			
			if(CUDA_ERR_HANDLE(cudaMemcpy(canny, cuda_canny, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost)))
			{
				cout<<"memcpy fail2"<<endl;
				continue;
			}
			img_canny = Mat(Size(width, height), CV_8UC1, canny);
			imshow("img_canny_gpu", img_canny);	
		}
		if(waitKey(1) == 'q')
			break;
		//break;
	}

	cudaFree(cuda_gray);
	cudaFree(cuda_gauss);
	cudaFree(cuda_gauss_kernel);
	cudaFree(cuda_sobel_x);
	cudaFree(cuda_sobel_y);
	cudaFree(cuda_sobel);
	cudaFree(cuda_canny);

	delete[] gauss;
	gauss = nullptr;
	delete[] gauss_kernel;
	gauss_kernel = nullptr;
	delete[] sobel_x;
	sobel_x = nullptr;
	delete[] canny;
	canny = nullptr;
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
	                                 CUDA_GetPixelVal(img, img_height, img_width, i+1, j-1) * (-1) +
	                                 CUDA_GetPixelVal(img, img_height, img_width, i+1, j  ) * (-2) +
	                                 CUDA_GetPixelVal(img, img_height, img_width, i+1, j+1) * (-1);

	*(sobel_y + i * img_width + j) = CUDA_GetPixelVal(img, img_height, img_width, i-1, j-1) * (-1) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i-1, j+1) * (1) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i  , j-1) * (-2) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i  , j+1) * (2) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i+1, j-1) * (-1) +
		                             CUDA_GetPixelVal(img, img_height, img_width, i+1, j+1) * (1);
	
	float val =sqrt(pow(*(sobel_x + i * img_width + j), 2) + pow(*(sobel_y + i * img_width + j), 2));
	*(output + i * img_width + j) = val;
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

__global__ void CUDA_NoneMaxSuppress(unsigned char* sobel, int sobel_width, int sobel_height, short* sobel_x, short* sobel_y, unsigned char* output)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i = id / sobel_width;
	int j = id % sobel_width;
	
	if(id >= sobel_width * sobel_height)
		return;
	
	if(i == 0 || j == 0)
	{
		*(output + i * sobel_width + j) = 0;
	}
	else 
	{
		short gx = *(sobel_x + i * sobel_width + j);
		short gy = *(sobel_y + i * sobel_width + j);
		short g1,g2,g3,g4;
		float weight = 0.0f;
		if(gx == 0 || gy == 0)
		{
			*(output + i * sobel_width + j) = 0;
		}
		else
		{
			if(abs(gx) < abs(gy) && gx * gy >= 0)
			{
				weight = (float)abs(gx) / abs(gy);
				g1 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i    , j + 1);
				g2 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j + 1);
				g3 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i    , j - 1);
				g4 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j - 1);
			}
			else if(abs(gx) >= abs(gy) && gx * gy > 0)
			{
				weight = (float)abs(gy) / abs(gx);
				g1 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j    );
				g2 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j + 1);
				g3 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j    );
				g4 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j - 1);
			}
			else if(abs(gx) > abs(gy) && gx * gy <= 0)
			{
				weight = (float)abs(gy) / abs(gx);
				g1 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j    );
				g2 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j - 1);
				g3 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j    );
				g4 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j + 1);
			}
			else if(abs(gx) <= abs(gy) && gx * gy < 0)
			{
				weight = (float)abs(gx) / abs(gy);
				g1 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i    , j - 1);
				g2 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j - 1);
				g3 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i    , j + 1);
				g4 = CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j + 1);
			}
			else
			{
				printf("invalid gradient\n");
			}
			int dot1 = g1 * (1 - weight) + g2 * weight;
			int dot2 = g3 * (1 - weight) + g4 * weight;
			if(*(sobel + i * sobel_width + j) > dot1 && *(sobel + i * sobel_width + j) > dot2)
				*(output + i * sobel_width + j) = *(sobel + i * sobel_width + j);
			else
				*(output + i * sobel_width + j) = 0;
		}
	}
}

void NoneMaxSuppress(unsigned char* sobel, int sobel_width, int sobel_height, short* sobel_x, short* sobel_y, unsigned char* output)
{
	for(int i = 0; i < sobel_height; i++)
	{
		for(int j = 0; j < sobel_width; j++)
		{
			if(i == 0 || j == 0)
			{
				*(output + i * sobel_width + j) = 0;
			}
			else 
			{
				short gx = *(sobel_x + i * sobel_width + j);
				short gy = *(sobel_y + i * sobel_width + j);
				short g1,g2,g3,g4;
				float weight = 0.0f;
				if(gx == 0 || gy == 0)
				{
					*(output + i * sobel_width + j) = 0;
				}
				else
				{
					if(abs(gx) < abs(gy) && gx * gy >= 0)
					{
						weight = (float)abs(gx) / abs(gy);
						g1 = GetPixelVal(sobel, sobel_height, sobel_width, i    , j + 1);
						g2 = GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j + 1);
						g3 = GetPixelVal(sobel, sobel_height, sobel_width, i    , j - 1);
						g4 = GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j - 1);
					}
					else if(abs(gx) >= abs(gy) && gx * gy > 0)
					{
						weight = (float)abs(gy) / abs(gx);
						g1 = GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j    );
						g2 = GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j + 1);
						g3 = GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j    );
						g4 = GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j - 1);
					}
					else if(abs(gx) > abs(gy) && gx * gy <= 0)
					{
						weight = (float)abs(gy) / abs(gx);
						g1 = GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j    );
						g2 = GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j - 1);
						g3 = GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j    );
						g4 = GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j + 1);
					}
					else if(abs(gx) <= abs(gy) && gx * gy < 0)
					{
						weight = (float)abs(gx) / abs(gy);
						g1 = GetPixelVal(sobel, sobel_height, sobel_width, i    , j - 1);
						g2 = GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j - 1);
						g3 = GetPixelVal(sobel, sobel_height, sobel_width, i    , j + 1);
						g4 = GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j + 1);
					}
					else
					{
						cout<<"none"<<endl;
					}
					int dot1 = g1 * (1 - weight) + g2 * weight;
					int dot2 = g3 * (1 - weight) + g4 * weight;
					if(*(sobel + i * sobel_width + j) > dot1 && *(sobel + i * sobel_width + j) > dot2)
						*(output + i * sobel_width + j) = *(sobel + i * sobel_width + j);
					else
						*(output + i * sobel_width + j) = 0;
				}
			}
		}
	}	
}

__global__ void CUDA_DoubleThreshold(unsigned char* sobel, int sobel_width, int sobel_height, int min_val, int max_val, unsigned char* canny)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i = id / sobel_width;
	int j = id % sobel_width;
	if(id >= sobel_width * sobel_height)
		return;

	if(i == 0 || i == sobel_height - 1 || j == 0 || j == sobel_width - 1)
	{
		*(canny + i * sobel_width + j) = 255;
		return;
	}
	if(*(sobel + i * sobel_width + j) > max_val)
	{
		*(canny + i * sobel_width + j) = 255;
	}
	else if(*(sobel + i * sobel_width + j) > min_val && *(sobel + i * sobel_width + j) < max_val)
	{
		if(CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j - 1) > max_val ||
		   CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j    ) > max_val ||
		   CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j + 1) > max_val ||
		   CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i    , j - 1) > max_val ||
		   CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i    , j + 1) > max_val ||
		   CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j - 1) > max_val ||
		   CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j    ) > max_val ||
		   CUDA_GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j + 1) > max_val)
		{
			*(canny + i * sobel_width + j) = 255;
		}
		else
		{
			*(canny + i * sobel_width + j) = 0;
		}
	}
	else
	{
		*(canny + i * sobel_width + j) = 0;
	}
}

void DoubleThreshold(unsigned char* sobel, int sobel_width, int sobel_height, int min_val, int max_val, unsigned char* canny)
{
	for(int i = 1; i < sobel_height - 1; i++)
	{
		for(int j = 1; j < sobel_width - 1; j++)
		{
			if(*(sobel + i * sobel_width + j) > max_val)
			{
				*(canny + i * sobel_width + j) = 255;
			}
			else if(*(sobel + i * sobel_width + j) > min_val && *(sobel + i * sobel_width + j) < max_val)
			{
				if(GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j - 1) > max_val ||
				   GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j    ) > max_val ||
				   GetPixelVal(sobel, sobel_height, sobel_width, i - 1, j + 1) > max_val ||
				   GetPixelVal(sobel, sobel_height, sobel_width, i    , j - 1) > max_val ||
				   GetPixelVal(sobel, sobel_height, sobel_width, i    , j + 1) > max_val ||
				   GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j - 1) > max_val ||
				   GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j    ) > max_val ||
				   GetPixelVal(sobel, sobel_height, sobel_width, i + 1, j + 1) > max_val)
				{
					*(canny + i * sobel_width + j) = 255;
				}
				else
				{
					*(canny + i * sobel_width + j) = 0;
				}
			}
			else
			{
				*(canny + i * sobel_width + j) = 0;
			}
		}
	}
}

