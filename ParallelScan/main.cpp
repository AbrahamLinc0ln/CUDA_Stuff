#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert>
#include <cstdio> 
#include <string> 
#include <opencv2/opencv.hpp> 
#include <time.h>
#include <sys/time.h>

#include "scan.h"

#define LENGTH 1024 

void checkApproxResults(int *ref, int *gpu){

    for(int i = 0; i < LENGTH; i++){
        if(ref[i] - gpu[i] > 1e-5){
            std::cerr << "Error at position " << i << "\n"; 

            std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] <<"\n";
            std::cerr << "GPU:: " << +gpu[i] << "\n";

            exit(1);
        } 
    }
    std::cout << "Got him \n";
}


/*
void checkResult(const std::string &reference_file, const std::string &output_file, float eps){
    cv::Mat ref_img, out_img; 

    ref_img = cv::imread(reference_file, -1);
    out_img = cv::imread(output_file, -1);


    unsigned char *refPtr = ref_img.ptr<unsigned char>(0);
    unsigned char *oPtr = out_img.ptr<unsigned char>(0);

    checkApproxResults(refPtr, oPtr, ref_img.rows*ref_img.cols*ref_img.channels());
    std::cout << "PASSED!";
}
*/

//standard scan
void cpuScan(int *vectorIn, int *vectorOut){
    vectorOut[0] = vectorIn[0];
    for(int i = 1; i < LENGTH; i++) vectorOut[i] = vectorIn[i] + vectorOut[i-1]; 
}


void vectorGen(int *vIn){
    for(int i = 0; i < LENGTH; i++) vIn[i] = rand() % 10;
}

void printResults(int *c_vIn, int *c_vOut, int *h_vOut){
    for(int i = 0; i < LENGTH; i++) std::cout << c_vIn[i] << " ";
    std::cout <<"\n";
    for(int i = 0; i < LENGTH; i++) std::cout << c_vOut[i] << " ";
    std::cout <<"\n";
    for(int i = 0; i < LENGTH; i++) std::cout << h_vOut[i] << " ";
    std::cout <<"\n";
}

int main(int argc, char const *argv[])
{
   int (*c_vIn) = new int[LENGTH];
   int (*c_vOut) = new int[LENGTH];
   int (*h_vIn) = new int[LENGTH];
   int (*h_vOutI) = new int[LENGTH];
   int (*h_vOutE) = new int[LENGTH];


   int *d_vInI, *d_vOutI, *d_vInE, *d_vOutE;

   struct timespec start, finish;

   // generate random vector 
   vectorGen(c_vIn);
   memcpy(h_vIn, c_vIn, sizeof(int)*LENGTH); 

   // Perform CPU based scan
   clock_gettime(CLOCK_REALTIME, &start);
   cpuScan(c_vIn, c_vOut);
   clock_gettime(CLOCK_REALTIME, &finish);
   
   long scan_time = finish.tv_nsec-start.tv_nsec;
   std::cout << "CPU scan time (ns): " << scan_time << "\n";
   

   checkCudaErrors(cudaMalloc((void**)&d_vInI, sizeof(int)*LENGTH));
   checkCudaErrors(cudaMalloc((void**)&d_vOutI, sizeof(int)*LENGTH)); 
   checkCudaErrors(cudaMalloc((void**)&d_vInE, sizeof(int)*LENGTH));
   checkCudaErrors(cudaMalloc((void**)&d_vOutE, sizeof(int)*LENGTH));

   checkCudaErrors(cudaMemcpy(d_vInI, h_vIn, sizeof(int)*LENGTH, cudaMemcpyHostToDevice)); 
   checkCudaErrors(cudaMemcpy(d_vInE, h_vIn, sizeof(int)*LENGTH, cudaMemcpyHostToDevice));
   //checkCudaErrors(cudaMemcpy(d_vOut, h_vIn, sizeof(int)*LENGTH, cudaMemcpyHostToDevice)); 
   //checkCudaErrors(cudaMemcpy(h_vOut, d_vOut, sizeof(int)*LENGTH, cudaMemcpyDeviceToHost));
   
   //printResults(c_vIn, c_vOut, h_vIn);

   // call the kernel 
   launch_scan(d_vInI, d_vInE, d_vOutI, d_vOutE, LENGTH);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());

   std::cout << "Finished kernel launch \n";

   checkCudaErrors(cudaMemcpy(h_vOutI, d_vOutI, sizeof(int)*LENGTH, cudaMemcpyDeviceToHost));
   checkCudaErrors(cudaMemcpy(h_vOutE, d_vOutE, sizeof(int)*LENGTH, cudaMemcpyDeviceToHost));

   printResults(c_vIn, c_vOut, h_vOutE);

   // check if the caclulation was correct to a degree of tolerance
   //checkApproxResults(c_vOut, h_vOutI);
   checkApproxResults(c_vOut, h_vOutE); 
   //checkResult(reference, outfile, 1e-5);

   cudaFree(d_vOutI);
   cudaFree(d_vInI);
   cudaFree(d_vOutE);
   cudaFree(d_vInE);

    return 0;
}



