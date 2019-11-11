#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

#define PI 3.141592653
#define IS_NOT_EDGE(a) (a < min_val)
#define IS_STRONG_EDGE(a) (a >= max_val)
#define IS_WEAK_EDGE(a)   (a >= min_val && a < max_val)
unsigned char GetImgPixelValue(const unsigned char* img, Size img_size, int i, int j);
void Conv(const unsigned char* img, Size img_size, const float* kernel, unsigned char kernel_size, unsigned char* output);
void GenerateGaussKernel(unsigned char size, float sigma, float* output);

void Gauss(const unsigned char* img, Size img_size, float* kernel, unsigned char kernel_size, unsigned char* output);
void Sobel(const unsigned char* img, Size img_size, short* sobel_x, short* sobel_y, unsigned char* output);
void NoneMaxSuppress(const unsigned char* img, Size img_size, const short* sobel_x, const short* sobel_y, unsigned char* output);
void DoubleThreshold(const unsigned char* img, Size img_size, unsigned char min_val, unsigned char max_val, unsigned char* output);
void DoubleThreshold2(unsigned char* sobel, int width, int height, int min_val, int max_val, unsigned char* output);
void IsWeakEdge(unsigned char* sobel, int width, int height, int min_val, int max_val, int i, int j, unsigned int* stack, unsigned int* top, unsigned char* visited, unsigned char* output);
int main(int agrc, char** agrv)
{
	int width  = 640;
	int height = 480;
	int gauss_size = 3;
	Mat img_src, img_gray, img_gauss, img_sobel, img_suppress, img_canny;
    cout<<"CANNY KASOKU"<<endl;
	
	VideoCapture camera(0);
	float* gauss_kernel = new float[gauss_size * gauss_size];
	unsigned char* gauss = new unsigned char[width * height];
	short* sobel_x = new short[width * height];
	short* sobel_y = new short[width * height];
	unsigned char* sobel = new unsigned char[width * height];
	unsigned char* canny = new unsigned char[width * height];  
	clock_t start_time, end_time;

	GenerateGaussKernel(3, 1.5, gauss_kernel);

	cout<<endl;
	while(1)
	{
		camera >> img_src;
		resize(img_src,img_src, Size(width, height), 0, 0, CV_INTER_LINEAR);
		imshow("img_src", img_src);
		cvtColor(img_src, img_gray, CV_BGR2GRAY);
		imshow("img_gray", img_gray);
		
		start_time = clock();
		
		/*GAUSSIAN*/
		Gauss(img_gray.data, img_gray.size(), gauss_kernel, 3, gauss);
		//img_gauss = Mat(img_gray.size(), img_gray.type(), gauss);
		//imshow("guass", img_gauss);
		
		/*SOBEL*/
		Sobel(gauss, Size(width, height), sobel_x, sobel_y, sobel);
		//img_sobel = Mat(img_gray.size(), CV_8UC1, sobel);
		//resize(img_sobel, img_sobel, Size(640, 480), 0, 0, CV_INTER_LINEAR);
		//imshow("img_sobel", img_sobel);

		/*NONE_MAX_SUPPRESS*/
		NoneMaxSuppress(sobel, Size(width, height), sobel_x, sobel_y, sobel);
		//img_suppress = Mat(Size(width, height), CV_8UC1, sobel);
		//resize(img_suppress, img_suppress, Size(640, 480), 0, 0, CV_INTER_LINEAR);
		//imshow("img_suppress", img_suppress);

		/*DoubleThreashold*/
		DoubleThreshold2(sobel, width, height, 50, 90, canny);
		//DoubleThreshold(sobel, Size(width, height), 50, 90, canny);
		end_time = clock();
		img_canny = Mat(Size(width, height), CV_8UC1, canny);
		resize(img_canny, img_canny, Size(640, 480), 0, 0, CV_INTER_LINEAR);
		imshow("img_canny", img_canny);

		cout<<"time : "<<(double)(end_time - start_time) / CLOCKS_PER_SEC <<endl;
		if(waitKey(1) == 'q')
		{
			destroyAllWindows();
			break;
		}
	}
	delete[] gauss_kernel;
	gauss_kernel = nullptr;
	delete[] sobel_x;
	sobel_x = nullptr;
	delete[] sobel_y;
	sobel_y = nullptr;
	delete[] sobel;
	sobel = nullptr;
	delete[] canny;
	canny = nullptr;
    return 0;
}

unsigned char GetImgPixelValue(const unsigned char* img, Size img_size, int i, int j)
{
	if(i >= img_size.height || i < 0)
		return 0;
	if(j >= img_size.width  || j < 0)
		return 0;
	return *(img + i * img_size.width + j);
}

void GenerateGaussKernel(unsigned char size, float sigma, float* output)
{
	int center = size / 2;
	float sum = 0.0f;
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			*(output + i * size + j) = (float)1 / (2 * PI * sigma * sigma) * exp(-(pow(i - center, 2) + pow(j - center, 2)) / (2 * pow(sigma, 2)));
			sum += *(output + i * size + j);
		}
	}
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			*(output + i * size + j) /= sum;
		}
	}
}

void Gauss(const unsigned char* img, Size img_size, float* kernel, unsigned char kernel_size, unsigned char* output)
{
	Conv(img, img_size, kernel, kernel_size, output);
}

void Conv(const unsigned char* img, Size img_size, const float* kernel, unsigned char kernel_size, unsigned char* output)
{
    for(int i = 0; i < img_size.height; i++)
	{
		for(int j = 0; j < img_size.width; j++)
		{
			int new_pixel_value = 0;
			int mid_kernel_size = kernel_size / 2;
			for(int k = 0; k < kernel_size; k++)
			{
				for(int m = 0; m < kernel_size; m++)
				{
					new_pixel_value += GetImgPixelValue(img, img_size, i + k - mid_kernel_size, j + m - mid_kernel_size) * (*(kernel + k * kernel_size + m));
				}
			}
			*(output + i * img_size.width + j) = new_pixel_value;
		}
	}
}


void Sobel(const unsigned char* img, Size img_size, short* sobel_x, short* sobel_y, unsigned char* output)
{
    static float x[9] = {1,2,1,0,0,0,-1,-2,-1};
	static float y[9] = {-1,0,1,-2,0,2,-1,0,1};
    int width  = img_size.width;
	int height = img_size.height;
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			*(sobel_x + i * width + j) = GetImgPixelValue(img, img_size, i-1, j-1) * x[0] + 
							 			 GetImgPixelValue(img, img_size, i-1, j  ) * x[1] + 
			                             GetImgPixelValue(img, img_size, i-1, j+1) * x[2] + 
			                             GetImgPixelValue(img, img_size, i  , j-1) * x[3] + 
			                             GetImgPixelValue(img, img_size, i  , j  ) * x[4] + 
			                             GetImgPixelValue(img, img_size, i  , j+1) * x[5] + 
			                             GetImgPixelValue(img, img_size, i+1, j-1) * x[6] + 
			                             GetImgPixelValue(img, img_size, i+1, j  ) * x[7] + 
			                             GetImgPixelValue(img, img_size, i+1, j+1) * x[8]; 
			

			*(sobel_y + i * width + j) = GetImgPixelValue(img, img_size, i-1, j-1) * y[0] + 
				                         GetImgPixelValue(img, img_size, i-1, j  ) * y[1] + 
				                         GetImgPixelValue(img, img_size, i-1, j+1) * y[2] + 
				                         GetImgPixelValue(img, img_size, i  , j-1) * y[3] + 
				                         GetImgPixelValue(img, img_size, i  , j  ) * y[4] + 
				                         GetImgPixelValue(img, img_size, i  , j+1) * y[5] + 
				                         GetImgPixelValue(img, img_size, i+1, j-1) * y[6] + 
				                         GetImgPixelValue(img, img_size, i+1, j  ) * y[7] + 
				                         GetImgPixelValue(img, img_size, i+1, j+1) * y[8]; 
			
			*(output + i * width + j) = sqrt(pow(*(sobel_x + i * width + j), 2) + pow(*(sobel_y + i * width + j), 2));
		}
	}
}

void NoneMaxSuppress(const unsigned char* img, Size img_size, const short* sobel_x, const short* sobel_y, unsigned char* output)
{
	int height = img_size.height;
	int width  = img_size.width;
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			if(i == 0 || j == 0)
			{
				*(output + i * width + j) = 0;
			}
			else
			{
				short gv = *(sobel_x + i * width + j);
				short gh = *(sobel_y + i * width + j);
				short g1, g2, g3, g4;
				float weight = 1.0f;
				
				if(gv == 0 && gh == 0)
				{
					*(output + i * width + j) = 0;
				}
				else
				{
					if(abs(gv) < abs(gh) && gv * gh >= 0)
					{
						weight = (float)abs(gv) / abs(gh);
						g1 = GetImgPixelValue(img, img_size, i    , j + 1);
						g2 = GetImgPixelValue(img, img_size, i - 1, j + 1);
						g3 = GetImgPixelValue(img, img_size, i    , j - 1);
						g4 = GetImgPixelValue(img, img_size, i + 1, j - 1);
					}
					else if(abs(gv) >= abs(gh) && gv * gh > 0)
					{
						weight = (float)abs(gh) / abs(gv);
						g1 = GetImgPixelValue(img, img_size, i - 1, j    );
						g2 = GetImgPixelValue(img, img_size, i - 1, j + 1);
						g3 = GetImgPixelValue(img, img_size, i + 1, j    );
						g4 = GetImgPixelValue(img, img_size, i + 1, j - 1);
					}
					else if(abs(gv) > abs(gh) && gv * gh <= 0)
					{
						weight = (float)abs(gh) / abs(gv);
						g1 = GetImgPixelValue(img, img_size, i - 1, j    );
						g2 = GetImgPixelValue(img, img_size, i - 1, j - 1);
						g3 = GetImgPixelValue(img, img_size, i + 1, j    );
						g4 = GetImgPixelValue(img, img_size, i + 1, j + 1);
					}
					else if(abs(gv) <= abs(gh) && gv * gh < 0)
					{
						weight = (float)abs(gv) / abs(gh);
						g1 = GetImgPixelValue(img, img_size, i    , j - 1);
						g2 = GetImgPixelValue(img, img_size, i - 1, j - 1);
						g3 = GetImgPixelValue(img, img_size, i    , j + 1);
						g4 = GetImgPixelValue(img, img_size, i + 1, j + 1);
					}
					else
					{
						cout<<"none"<<endl;
					}

					unsigned char dot1 = g1 * (1 - weight) + g2 * weight;
					unsigned char dot2 = g3 * (1 - weight) + g4 * weight;
					if(*(img + i * width + j) > dot1 && *(img + i * width + j) > dot2)
						*(output + i * width + j) = *(img + i * width + j);
					else
						*(output + i * width + j) = 0;
				} 
			}
		}
	}
}

void DoubleThreshold(const unsigned char* img, Size img_size, unsigned char min_val, unsigned char max_val, unsigned char* output)
{
	int height = img_size.height;
	int width  = img_size.width;
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			if(*(img + i * width + j) > max_val)
			{
				*(output + i * width + j) = 255;
			}
			else if(*(img + i * width + j) < max_val && *(img + i * width + j) > min_val)
			{
				if(GetImgPixelValue(img, img_size, i - 1, j - 1) > max_val ||
				   GetImgPixelValue(img, img_size, i - 1, j    ) > max_val ||
				   GetImgPixelValue(img, img_size, i - 1, j + 1) > max_val ||
				   GetImgPixelValue(img, img_size, i    , j - 1) > max_val ||
				   GetImgPixelValue(img, img_size, i    , j + 1) > max_val ||
				   GetImgPixelValue(img, img_size, i + 1, j - 1) > max_val ||
				   GetImgPixelValue(img, img_size, i + 1, j    ) > max_val ||
				   GetImgPixelValue(img, img_size, i + 1, j + 1) > max_val)
				{
					*(output + i * width + j) = 255;
				}
				else
				{
					*(output + i * width + j) = 0;
				}
			}
			else
			{
				*(output + i * width + j) = 0;
			}				
		}
	}
}
void DoubleThreshold2(unsigned char* sobel, int width, int height, int min_val, int max_val, unsigned char* output)
{
	unsigned int* weak_stack = new unsigned int[width * height];
	unsigned char* visited = new unsigned char[width * height];
	unsigned int stack_top = 0;
	unsigned int center_index = 0;
	memset(visited, 0, width * height);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (IS_STRONG_EDGE(GetImgPixelValue(sobel, Size(width, height), i, j)) && visited[i * width + j] != 1)
			{
				IsWeakEdge(sobel, width, height, min_val, max_val, i, j, weak_stack, &stack_top, visited, output);
				while (stack_top > 0)
				{
					center_index = weak_stack[stack_top - 1];
					stack_top--;
					IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width - 1, center_index % width - 1, weak_stack, &stack_top, visited, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width - 1, center_index % width    , weak_stack, &stack_top, visited, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width - 1, center_index % width + 1, weak_stack, &stack_top, visited, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width    , center_index % width - 1, weak_stack, &stack_top, visited, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width    , center_index % width + 1, weak_stack, &stack_top, visited, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width + 1, center_index % width - 1, weak_stack, &stack_top, visited, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width + 1, center_index % width    , weak_stack, &stack_top, visited, output);
					IsWeakEdge(sobel, width, height, min_val, max_val, center_index / width + 1, center_index % width + 1, weak_stack, &stack_top, visited, output);
				}
			}
			else if (IS_NOT_EDGE(GetImgPixelValue(sobel, Size(width, height), i, j)))
			{
				output[i * width + j] = 0;
			}
		}
	}

	delete[] weak_stack;
	weak_stack = nullptr;
	delete[] visited;
	visited = nullptr;
}

void IsWeakEdge(unsigned char* sobel, int width, int height, int min_val, int max_val, int i, int j, unsigned int* stack, unsigned int* top, unsigned char* visited, unsigned char* output)
{
	if (i < 0 || i >= height)
		return;
	if (j < 0 || j >= width)
		return;
	if (visited[i * width + j] == 1)
		return;
	if (IS_STRONG_EDGE(GetImgPixelValue(sobel, Size(width, height), i, j)))
	{
		output[i * width + j] = 255;
		visited[i * width + j] = 1;
		stack[*top] = i * width + j;
		(*top)++;
	}
	else if(IS_WEAK_EDGE(GetImgPixelValue(sobel, Size(width, height), i, j)))
	{
		output[i * width + j] = 255;
		visited[i * width + j] = 1;
		stack[*top] = i * width + j;
		(*top)++;
	}
	else
	{
		visited[i * width + j] = 1;
		output[i * width + j] = 50;
	}
}
