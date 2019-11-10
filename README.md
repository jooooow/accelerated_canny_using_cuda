# accelerated_canny_using_cuda
Canny algorithm implemented on NVIDIA Jetson Nano using GPU to accelerate  
an implementation of  `Efficient Canny Edge Detection using a GPU, Kohei Ogawa, Yasuaki Ito, Koji Nakano`

## tools
CUDA10  
OpenCV3  
NVIDIA Jetson Nano  

## steps  
1.allocate image into SM  
2.extend subimage  
3.gauss and sobel  
4.cut edge  
5.none maximum suppress  
6.double threshold  
