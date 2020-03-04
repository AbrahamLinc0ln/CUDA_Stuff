Gaussian blurs RGB image using CPU and then GPU and compares the processing time of each.

lena.png:  test file
lenaBlur.png: example of blurred lena.png
main.cpp: main cpp script that loads image and executes CPU blur
blur_kernels.cu: blurs rgb image using Gaussian kernel, with and without memory sharing for comparison.

To run use Makefile to compile then run executable
