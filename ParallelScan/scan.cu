#include "scan.h"

#define BLOCK 1024 

/*
__global__ 
void ineff_scan(int *d_vIn, int *d_vOut, int length, int stride){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
// Inefficient algorithm
	if(x<length){
		if((x-stride)>=0) d_vOut[x] = d_vOut[x] + d_vOut[x-stride];
	}

}
*/
__global__
void ineff_scan_shared(int *d_vIn, int *d_vOut, int length){
	__shared__ int results[BLOCK];
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x*blockDim.x + tx;
	
	
	if (x < length){

		results[tx] = d_vIn[x];

		for(unsigned int stride = 1; stride <= tx; stride *= 2){
			
			__syncthreads;
			int temp = results[tx - stride];
			__syncthreads;
			results[tx] = results[tx] + temp;
			
		}
		__syncthreads;
		d_vOut[x] = results[tx];
	}

	
}


__global__
void eff_scan(int *d_vIn, int *d_vOut, int length){
	__shared__ int results [2*BLOCK];
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x*blockDim.x+tx;

	if(x<length/2){
		results[2*tx] = d_vIn[2*x];
		results[2*tx+1] = d_vIn[2*x+1];

	}

	__syncthreads;

	for(int stride = 1; stride <= BLOCK; stride *=2){
		int i = (tx + 1)* 2* stride- 1;
		__syncthreads;
		if(i<2*BLOCK){
			
			results[i] = results[i]+ results[i- stride];

		}
	}	

	for(int fall = BLOCK/2; fall > 0; fall /= 2){
		__syncthreads;
		int j = (tx + 1) * 2 * fall - 1;

		if((j + fall) < 2*BLOCK){
			
			 results[j + fall] = results[j + fall] + results[j];
			 
		}
	}
	__syncthreads;

	if(x<2*length){
		 d_vOut[2*x] = results[2*tx];
		 d_vOut[2*x+1] = results[2*tx+1];
	}

}

__global__
void eff_scan2(int *d_vIn, int *d_vOut, int length){
	__shared__ int results [2*BLOCK];
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x*blockDim.x+tx;

	if(x<length) results[tx] = d_vIn[x];

	for(int stride = 1; stride < BLOCK; stride *=2){
		int i = (tx + 1)* 2* stride- 1;
		if(i<2*BLOCK){
			results[i] = results[i]+ results[i- stride];
		}
		__syncthreads;
	}	

	for(int fall = BLOCK/2; fall > 0; fall /= 2){
		int j = (tx + 1) * 2 * fall - 1;
		__syncthreads;
		if((j + fall) < 2*BLOCK) results[j + fall] = results[j + fall] + results[j];
	}
	__syncthreads;

	if(x<length) d_vOut[blockIdx.x] = results[tx];


}

__global__
void vec_sum(int *d_vIn1, int *d_vIn2, int *d_vOut, int length){
	__shared__ int results [2*BLOCK];
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x*blockDim.x+tx;

	if(x<length) d_vOut[x]=d_vIn1[x]+d_vIn2[x];
}


void launch_scan(int *d_vInI, int *d_vInE, int *d_vOutI, int *d_vOutE, int length){
    // configure launch params here 
    
    	dim3 block(BLOCK,1,1);
    	dim3 grid(BLOCK/length+1, 1, 1);

/*
    for(int i = 1; i < length; i *= 2){
    	ineff_scan<<<grid,block>>>(d_vIn, d_vOut, length, i);
    	cudaDeviceSynchronize();
    	checkCudaErrors(cudaGetLastError());
    	//std::cout << i << "\n";
    }
*/
/*    	ineff_scan_shared<<<grid,block>>>(d_vInI, d_vOutI, length);
    	cudaDeviceSynchronize();
    	checkCudaErrors(cudaGetLastError());
*/    
    	if(length<=2*BLOCK){
		eff_scan<<<grid,block>>>(d_vInE, d_vOutE, length);
		cudaDeviceSynchronize();
   		checkCudaErrors(cudaGetLastError()); 
	}
	else{
		int sum[length];
		eff_scan2<<<grid,block>>>(d_vInE, sum, length);
		cudaDeviceSynchronize();
   		checkCudaErrors(cudaGetLastError()); 
	
		eff_scan<<<grid,block>>>(sum, sum, length);
		cudaDeviceSynchronize();
   		checkCudaErrors(cudaGetLastError()); 
			
		vec_sum<<<grid,block>>>(d_vInE, sum, d_vOutE, length);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
}






