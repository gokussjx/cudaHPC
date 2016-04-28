
# This project is developed by Zhangwei Yang(zy5f9), Bidyut Mukherjee (bm346), and Singkhorn Sittirug (sswdd), for Parallel Programming CMP_SC 7080, (Spring 2016) at the University of Missouri, Columbia.

# Primary language and SDK used is C, CUDA, and cuda_Helper library.

# To compile from source, navigate to '$PROJECT_ROOT' directory:
$ make clean  # Optional, but just in case
$ make all

# Expected input is a PGM image file

# The output binary/executable can be found in '$PROJECT_ROOT' directory:
# To execute the program using following default arguments:
# WINDOWSIZE: 3  (Represents 3x3 window)
# INPUT FILE: lena.pgm
# OUTPUT FILE: lena_out.pgm
$ make run

# To execute the program using custom arguments:
$ ./driver 7 lena.pgm lena_out.pgm

# That's it! The output is a PGM file, blurred as per window size

#####
# Main source file is driver.cu
# Library files are inside inc, include, and src directories
# Dependent binaries are inside bin directory, after compilation
####

# PS: CUDA has been very interesting! It's quite fun (and weird, and fun) to work with.
