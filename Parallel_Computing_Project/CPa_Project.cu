#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono> // For CPU benchmarking

using namespace std;
using namespace cv;

static int kernel = 1;  // Kernel size for CUDA kernel, change for different kernel sizes
static string PATH = ".\\Images\\XGA.jpg"; // Path to the image, change for other images

// CUDA kernel using global memory
__global__ void applyFilter(const unsigned char* input, unsigned char* output, int width, int height, int channels, const float* filter, int filterWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfFilterWidth = filterWidth / 2;
    float pixelSum[3] = { 0.0f, 0.0f, 0.0f };

    for (int ky = -halfFilterWidth; ky <= halfFilterWidth; ++ky) {
        for (int kx = -halfFilterWidth; kx <= halfFilterWidth; ++kx) {
            int imgX = min(max(x + kx, 0), width - 1);
            int imgY = min(max(y + ky, 0), height - 1);

            int imgIdx = (imgY * width + imgX) * channels;
            int filterIdx = (ky + halfFilterWidth) * filterWidth + (kx + halfFilterWidth);

            for (int c = 0; c < channels; ++c) {
                pixelSum[c] += input[imgIdx + c] * filter[filterIdx];
            }
        }
    }

    int outputIdx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c) {
        output[outputIdx + c] = min(max(int(pixelSum[c]), 0), 255);
    }
}


void applyCUDAFilter(const Mat& inputImage, Mat& outputImage, const float* filter, int filterWidth) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);

    unsigned char* d_input, * d_output;
    float* d_filter;

    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);
    cudaMalloc((void**)&d_filter, filterSize);

    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);

    dim3 blockSize(kernel, kernel);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    applyFilter << <gridSize, blockSize >> > (d_input, d_output, width, height, channels, d_filter, filterWidth);

    cudaDeviceSynchronize();
    cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}


// CUDA kernel using shared memory
__global__ void applyFilterShared(const unsigned char* input, unsigned char* output, int width, int height, int channels, const float* filter, int filterWidth) {
    extern __shared__ unsigned char sharedMem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int localX = threadIdx.x;
    int localY = threadIdx.y;

    int halfFilterWidth = filterWidth / 2;
    int sharedWidth = blockDim.x + 2 * halfFilterWidth;
    int sharedHeight = blockDim.y + 2 * halfFilterWidth;

    int sharedIdx = ((localY + halfFilterWidth) * sharedWidth + (localX + halfFilterWidth)) * channels;

    int globalIdx = (y * width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        if (x < width && y < height) {
            sharedMem[sharedIdx + c] = input[globalIdx + c];
        }
        else {
            sharedMem[sharedIdx + c] = 0; 
        }
    }

    for (int c = 0; c < channels; ++c) {
        // Left halo
        if (localX < halfFilterWidth) {
            int sharedHaloIdx = ((localY + halfFilterWidth) * sharedWidth + localX) * channels + c;
            int imgX = max(x - halfFilterWidth, 0);
            sharedMem[sharedHaloIdx] = input[(y * width + imgX) * channels + c];
        }

        // Right halo
        if (localX >= blockDim.x - halfFilterWidth) {
            int sharedHaloIdx = ((localY + halfFilterWidth) * sharedWidth + (localX + 2 * halfFilterWidth)) * channels + c;
            int imgX = min(x + halfFilterWidth, width - 1);
            sharedMem[sharedHaloIdx] = input[(y * width + imgX) * channels + c];
        }

        // Top halo
        if (localY < halfFilterWidth) {
            int sharedHaloIdx = (localY * sharedWidth + (localX + halfFilterWidth)) * channels + c;
            int imgY = max(y - halfFilterWidth, 0);
            sharedMem[sharedHaloIdx] = input[(imgY * width + x) * channels + c];
        }

        // Bottom halo
        if (localY >= blockDim.y - halfFilterWidth) {
            int sharedHaloIdx = ((localY + 2 * halfFilterWidth) * sharedWidth + (localX + halfFilterWidth)) * channels + c;
            int imgY = min(y + halfFilterWidth, height - 1);
            sharedMem[sharedHaloIdx] = input[(imgY * width + x) * channels + c];
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        float pixelSum[3] = { 0.0f, 0.0f, 0.0f };

        for (int ky = -halfFilterWidth; ky <= halfFilterWidth; ++ky) {
            for (int kx = -halfFilterWidth; kx <= halfFilterWidth; ++kx) {
                int sharedConvIdx = ((localY + halfFilterWidth + ky) * sharedWidth + (localX + halfFilterWidth + kx)) * channels;
                int filterIdx = (ky + halfFilterWidth) * filterWidth + (kx + halfFilterWidth);

                for (int c = 0; c < channels; ++c) {
                    pixelSum[c] += sharedMem[sharedConvIdx + c] * filter[filterIdx];
                }
            }
        }

        for (int c = 0; c < channels; ++c) {
            output[globalIdx + c] = min(max(int(pixelSum[c]), 0), 255);
        }
    }
}


void applyCUDAFilterShared(const Mat& inputImage, Mat& outputImage, const float* filter, int filterWidth) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);

    unsigned char* d_input, * d_output;
    float* d_filter;

    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);
    cudaMalloc((void**)&d_filter, filterSize);

    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);

    dim3 blockSize(kernel, kernel);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    int sharedMemSize = (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) * channels * sizeof(unsigned char);
    applyFilterShared << <gridSize, blockSize, sharedMemSize >> > (d_input, d_output, width, height, channels, d_filter, filterWidth);

    cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}


// CUDA kernel using texture memory
__global__ void applyFilterTexture(cudaTextureObject_t texObj, unsigned char* output, int width, int height, int channels, const float* filter, int filterWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfFilterWidth = filterWidth / 2;
    float pixelSum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int ky = -halfFilterWidth; ky <= halfFilterWidth; ++ky) {
        for (int kx = -halfFilterWidth; kx <= halfFilterWidth; ++kx) {
            int filterIdx = (ky + halfFilterWidth) * filterWidth + (kx + halfFilterWidth);

            float4 texValue = tex2D<float4>(texObj, x + kx + 0.5f, y + ky + 0.5f);

            pixelSum[0] += texValue.x * filter[filterIdx]; // R
            pixelSum[1] += texValue.y * filter[filterIdx]; // G
            pixelSum[2] += texValue.z * filter[filterIdx]; // B
            if (channels == 4) {
                pixelSum[3] += texValue.w * filter[filterIdx]; // A
            }
        }
    }

    int outputIdx = (y * width + x) * channels;
    output[outputIdx] = min(max(int(pixelSum[0] * 255.0f), 0), 255); // R
    output[outputIdx + 1] = min(max(int(pixelSum[1] * 255.0f), 0), 255); // G
    output[outputIdx + 2] = min(max(int(pixelSum[2] * 255.0f), 0), 255); // B
    if (channels == 4) {
        output[outputIdx + 3] = min(max(int(pixelSum[3] * 255.0f), 0), 255); // A
    }
}


void applyCUDAFilterTexture(const Mat& inputImage, Mat& outputImage, const float* filter, int filterWidth) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    Mat formattedInput;
    if (channels == 3) {
        cvtColor(inputImage, formattedInput, COLOR_BGR2BGRA); 
        channels = 4;
    }
    else {
        formattedInput = inputImage.clone();
    }

    size_t imageSize = width * height * channels * sizeof(unsigned char);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);

    unsigned char* d_output;
    float* d_filter;

    cudaMalloc((void**)&d_output, imageSize);
    cudaMalloc((void**)&d_filter, filterSize);

    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(cuArray, 0, 0, formattedInput.ptr(), formattedInput.step, formattedInput.step, height, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint; 
    texDesc.readMode = cudaReadModeNormalizedFloat; 
    texDesc.normalizedCoords = false;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    dim3 blockSize(kernel, kernel); 
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    applyFilterTexture << <gridSize, blockSize >> > (texObj, d_output, width, height, channels, d_filter, filterWidth);

    if (channels == 4) {
        Mat rgbaOutput(height, width, CV_8UC4);
        cudaMemcpy(rgbaOutput.data, d_output, imageSize, cudaMemcpyDeviceToHost);

        cvtColor(rgbaOutput, outputImage, COLOR_BGRA2BGR);
    }
    else {
        cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);
    }

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
    cudaFree(d_filter);
}


// CPU implementation
void benchmarkCPU(const Mat& inputImage, Mat& outputImage, const float* filter, int filterWidth) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    int halfFilterWidth = filterWidth / 2;


    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float pixelSum[3] = { 0.0f, 0.0f, 0.0f };
            for (int ky = -halfFilterWidth; ky <= halfFilterWidth; ++ky) {
                for (int kx = -halfFilterWidth; kx <= halfFilterWidth; ++kx) {
                    int imgX = min(max(x + kx, 0), width - 1);
                    int imgY = min(max(y + ky, 0), height - 1);

                    for (int c = 0; c < channels; ++c) {
                        pixelSum[c] += inputImage.at<Vec3b>(imgY, imgX)[c] * filter[(ky + halfFilterWidth) * filterWidth + (kx + halfFilterWidth)];
                    }
                }
            }
            for (int c = 0; c < channels; ++c) {
                outputImage.at<Vec3b>(y, x)[c] = min(max(int(pixelSum[c]), 0), 255);
            }
        }
    }

}


int main() {
    string imagePath = PATH;
    Mat inputImage = imread(imagePath, IMREAD_COLOR);

    if (inputImage.empty()) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }

    Mat outputImageCUDA_Global_soft = inputImage.clone();
    Mat outputImageCUDA_Global_sharp = inputImage.clone();
    Mat outputImageCUDA_Shared_soft = inputImage.clone();
    Mat outputImageCUDA_Shared_sharp = inputImage.clone();
    Mat outputImageCUDA_Texture_soft = inputImage.clone();
    Mat outputImageCUDA_Texture_sharp = inputImage.clone();
    Mat outputImageCPU_soft = inputImage.clone();
    Mat outputImageCPU_sharp = inputImage.clone();


    float softFilter[] = {
        1 / 16.0f, 2 / 16.0f, 1 / 16.0f,
        2 / 16.0f, 4 / 16.0f, 2 / 16.0f,
        1 / 16.0f, 2 / 16.0f, 1 / 16.0f
    };

    float sharpFilter[] = {
       0, -1, 0,
       -1, 5, -1,
       0, -1, 0
    };

    int filterWidth = 3;

	// Timing GPU execution Global Memory
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cout << "Applying CUDA filter with global memory..." << endl;
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
    {
        applyCUDAFilter(inputImage, outputImageCUDA_Global_soft, softFilter, filterWidth);
        applyCUDAFilter(inputImage, outputImageCUDA_Global_sharp, sharpFilter, filterWidth);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "CUDA Kernel Execution Time using Global memory: " << milliseconds/10 << " ms" << endl;


    // Timing GPU execution Shared Memory
    cout << "Applying CUDA filter with shared memory..." << endl;
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
    {
        applyCUDAFilterShared(inputImage, outputImageCUDA_Shared_soft, softFilter, filterWidth);
        applyCUDAFilterShared(inputImage, outputImageCUDA_Shared_sharp, sharpFilter, filterWidth);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
     milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "CUDA Kernel Execution Time using Shared memory: " << milliseconds/10 << " ms" << endl;

    // Timing GPU execution Texture Memory
    cout << "Applying CUDA filter with texture memory..." << endl;
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
    {
        applyCUDAFilterTexture(inputImage, outputImageCUDA_Texture_soft, softFilter, filterWidth);
        applyCUDAFilterTexture(inputImage, outputImageCUDA_Texture_sharp, sharpFilter, filterWidth);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
     milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "CUDA Kernel Execution Time using Texture memory: " << milliseconds/10 << " ms" << endl;


	//Timing CPU execution
    cout << "Benchmarking CPU..." << endl;
    auto start_CPU = chrono::high_resolution_clock::now();
	benchmarkCPU(inputImage, outputImageCPU_soft, softFilter, filterWidth);
	benchmarkCPU(inputImage, outputImageCPU_sharp, sharpFilter, filterWidth);
    auto stop_CPU = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = stop_CPU - start_CPU;
    cout << "CPU Execution Time: " << elapsed.count() << " ms" << endl;


    imshow("Input Image", inputImage);
    imshow("Filtered Image CUDA Soft Filter using Global memory", outputImageCUDA_Global_soft);
    imshow("Filtered Image CUDA Sharp Filter using Global memory", outputImageCUDA_Global_sharp);
    imshow("Filtered Image CUDA Soft Filter using Shared memory", outputImageCUDA_Shared_soft);
    imshow("Filtered Image CUDA Sharp Filter using Shared memory", outputImageCUDA_Shared_sharp);
    imshow("Filtered Image CUDA Soft Filter using Texture memory", outputImageCUDA_Texture_soft);
    imshow("Filtered Image CUDA Sharp Filter using Texture memory", outputImageCUDA_Texture_sharp);
    imshow("Filtered Image CPU Soft Filter", outputImageCPU_soft);
    imshow("Filtered Image CPU Sharp Filter", outputImageCPU_sharp);
    waitKey(0);

    return 0;
}
