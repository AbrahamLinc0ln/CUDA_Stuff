Script to convert RGB image to gray scale using CPU and GPU, then compares the times.

main.cpp:   main script that loads image and processes using the CPU
im2Gray.cu:   CUDA kernel that converts RGB pixels to grayscale and launcher
mandrill_gray.png:  Example output image

To compile run the Makefile, then execute.
