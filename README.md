# Image Filtering Implementation in CUDA  

This project was developed during my **Erasmus exchange program** as part of a course on parallel and GPU computing.  
It explores different CUDA memory strategies for image filtering:  

- **Global memory**  
- **Shared memory**  
- **Texture memory**  

For comparison, a **CPU implementation** is also included.  

## Report  

The full project report (with explanations, methodology, and results) is available here:  
[Project Report (PDF)](/CPa_Project_Doc.pdf)  

## Requirements  

- **CUDA Toolkit** (11.x or newer recommended)  
- **NVIDIA GPU** with CUDA support  
- **Visual Studio** (project built with `.sln` / `.vcxproj`)  
- **OpenCV** (for image I/O and display)  

## How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/IMAGE-FILTERING-IMPLEMENTATION-IN-CUDA.git
   cd IMAGE-FILTERING-IMPLEMENTATION-IN-CUDA

2. Open Parallel_Computing_Project.sln in Visual Studio.

3. Make sure CUDA Toolkit and OpenCV are configured.

4. Build and run the project.


### The program will:

* Apply softening and sharpening filters using Global, Shared, and Texture memory in CUDA.

* Benchmark performance vs. CPU implementation.

* Show input and output images in display windows.

## Notes  

* The Images folder contains input images for testing.

* Results and performance comparisons are discussed in the [Project Report (PDF)](/CPa_Project_Doc.pdf)
