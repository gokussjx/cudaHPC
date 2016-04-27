#ifndef STANDARD_H
#define STANDARD_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h>
typedef unsigned char color;
#define DIM 512
#define P5

#ifdef P5
#define MagicNumber '5'
#endif

#ifdef P6
#define MagicNumber '6'
#endif

#endif
