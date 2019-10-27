#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

__global__ void test(int* a, int* b, int* c, int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int data[10];
	if(x < size)
	{
		data[x] = a[x];
		a[x] = b[x];
		b[x] = data[x];
		c[x] = a[x] + b[x];
	}
}

int main()
{
	printf("cuda_test\n");

	Mat img = imread("/home/katsuto/Pictures/Wallpapers/timg.jpeg");
	int height = img.size().height;
	int width  = img.size().width;
	resize(img, img, Size(width/2, height/2), 0, 0);
	imshow("img", img);
	waitKey(0);

	int N = 10;
	int* a = (int*)malloc(N * sizeof(int));
	int* b = (int*)malloc(N * sizeof(int));
	int* c = (int*)malloc(N * sizeof(int));

	for(int i = 0; i < N; i ++)
	{
		a[i] = i + 1;
		b[i] = i * i;
		printf("%d\t%d\n", a[i], b[i]);
	}

	int* cuda_a;
	cudaMalloc(&cuda_a, N * sizeof(int));
	cudaMemcpy(cuda_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_b;
	cudaMalloc(&cuda_b, N * sizeof(int));
	cudaMemcpy(cuda_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_c;
	cudaMalloc(&cuda_c, N * sizeof(int));

	test<<<1,N>>>(cuda_a, cuda_b, cuda_c, N);

	cudaMemcpy(a, cuda_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, cuda_b, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c, cuda_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++)
	{
		printf("%d\t%d\t%d\n", a[i], b[i], c[i]);
	}

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
	
	free(a);
	free(b);
	free(c);
	a = NULL;
	b = NULL;
	c = NULL;
	return 0;
}
