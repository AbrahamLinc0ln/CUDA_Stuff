#include "im2Gray.h"

#define BLOCK 16



/*
 
  Given an input image d_in, perform the grayscale operation 
  using the luminance formula i.e. 
  o[i] = 0.224f*r + 0.587f*g + 0.111*b; 
  
  Your kernel needs to check for boundary conditions 
  and write the output pixels in gray scale format. 

  you may vary the BLOCK parameter.
 
 */
__global__ 
void im2Gray(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols){

 /*
   Your kernel here: Make sure to check for boundary conditions
  */
  int bX = blockDim.x;
  int bY = blockDim.y;
  int tX = threadIdx.x;
  int ty = threadIdx.y;

  unsigned int i = tX+bX*blockIdx.x;
  unsigned int j = ty+bY*blockIdx.y;

  if(i<numRows && j<numCols){
    unsigned int k=i+j*numCols;
    //d_grey[k] = (d_in[k].x + d_in[k].y + d_in[k].z)/3;
    d_grey[k] = (d_in[k].x*0.224 + d_in[k].y*0.587 + d_in[k].z*0.111); //Formula 2 
  }

}




void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    // configure launch params here 
    
    dim3 block(BLOCK,BLOCK,1);
    dim3 grid(ceil(numCols/BLOCK),ceil(numRows/BLOCK), 1);

    im2Gray<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
}





