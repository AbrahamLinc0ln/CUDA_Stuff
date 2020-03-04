#include "./gaussian_kernel.h" 

#define FWIDTH 9
#define TILE_WIDTH 12
#define BLOCK (TILE_WIDTH+FWIDTH-1)

__constant__ float d_filterConstant[FWIDTH*FWIDTH];


__global__ 
void gaussianBlur(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter){
	
	unsigned int x =blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y =blockIdx.y * blockDim.y + threadIdx.y;

	if (x<rows && y<cols){
		int pVal = 0;

		int start_x = x - (FWIDTH/2);
		int start_y = y - (FWIDTH/2);

		for(int i = 0; i < FWIDTH; ++i){
			for(int j = 0; j < FWIDTH; ++j){

				int curX = start_x+j;
				int curY = start_y+i;

				if(curY > -1 && curY < rows && curX > -1 && curX < cols){
					pVal += d_in[curY*rows+curX]*d_filter[i*FWIDTH+j];
				}

			}
		}

		d_out[y*cols+x] = pVal;
	}
} 

__global__ 
void gaussianBlurShared(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter){
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	unsigned int x =blockIdx.x * blockIdx.x + tx;
	unsigned int y =blockIdx.y * blockIdx.y + ty;

	__shared__ float sData[BLOCK][BLOCK];

	int n = TILE_WIDTH/2;
	int index_i = x - n;
	int index_j = y - n;

	if((index_i >= 0 && index_i < rows) && (index_j >= 0 && index_j < cols)){
		sData[tx][ty] = d_in[y*cols + x];
	}
	else{
		sData[tx][ty] = 0;
	}

	__syncthreads();

	if (tx<=(TILE_WIDTH) && ty<=(TILE_WIDTH)){
		int pVal = 0;

//		int start_x = x - (FWIDTH/2);
//		int start_y = y - (FWIDTH/2);

		for(int i = 0; i < FWIDTH; i++){
			for(int j = 0; j < FWIDTH; j++){

				//int curX = start_x+j;
				//int curY = start_y+i;

				//if(curY > -1 && curY < rows && curX > -1 && curX < cols){
					pVal += sData[i+tx][i+ty]*d_filterConstant[i*FWIDTH+j];
				//}

			}
		}

		d_out[y*cols+x] = pVal;
	}

} 


__global__ 
void gaussianBlurConst(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter){
	
	float pVal;

	unsigned int x =blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y =blockIdx.y * blockDim.y + threadIdx.y;

	if (x<rows && y<cols){
		pVal = 0;

		int start_x = x - (FWIDTH/2);
		int start_y = y - (FWIDTH/2);

		for(int i = 0; i < FWIDTH; ++i){
			for(int j = 0; j < FWIDTH; ++j){

				int curX = start_x+j;
				int curY = start_y+i;

				if(curY > -1 && curY < rows && curX > -1 && curX < cols){
					pVal += d_in[curY*rows+curX]*d_filterConstant[i*FWIDTH+j];
					__syncthreads();
				}

			}
		}

		d_out[y*cols+x] = pVal;
	}
} 



/*
  Given an input RGBA image separate 
  that into appropriate rgba channels.
 */
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned int cols, unsigned int rows){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x<cols && y<rows){
		unsigned int k = x + y*cols;
		d_r[k] = d_imrgba[k].x;
		d_g[k] = d_imrgba[k].y;
		d_b[k] = d_imrgba[k].z;
	}

} 
 


__global__ 
void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba, unsigned int cols, unsigned int rows){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x<cols && y<rows){
		unsigned int k = x + cols*y;
		d_orgba[k] = make_uchar4(d_b[k], d_g[k], d_r[k], 255);
	}
} 


void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 
   checkCudaErrors(cudaMemcpyToSymbol(d_filterConstant, d_filter, sizeof(float)*(FWIDTH*FWIDTH), 0, cudaMemcpyDeviceToDevice));

   dim3 blockSize(BLOCK,BLOCK,1);
   dim3 gridSize(ceil((cols-1)/TILE_WIDTH),ceil((rows-1)/TILE_WIDTH),1);
   separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, cols, rows); 
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());  

//Normal
    gaussianBlur<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter); 
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());  

    gaussianBlur<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter);  
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());  

    gaussianBlur<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter); 
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
/*
//const
    gaussianBlurConst<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter); 
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());  

	gaussianBlurConst<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter);  
 	cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());  

    gaussianBlurConst<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter); 
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());   

//shared
   gaussianBlurShared<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter); 
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());  

   gaussianBlurShared<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter);  
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());  

   gaussianBlurShared<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter); 
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
*/

   recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, cols, rows); 

   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());   

}




