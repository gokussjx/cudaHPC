#include "standard.h"
#define INPUT(I,x,y) input##I[((y)*(512*3))+(x)*3]
color* load(int fd){
    int ct0a=4; struct stat _fstat;
    if (fstat(fd, &_fstat) == -1) { perror("fstat()"); exit(1); }
    color *p=mmap(NULL, _fstat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    do{ p=memchr(p, 0x0a, 64)+1; } while(--ct0a);
    return p;
}

int main(int argc, char *argv[]){
    if(argc<3){ fprintf(stderr, "Usage: input1.ppm input2.ppm\n"); return 1; }
    color *input1=load(open(argv[1], O_RDONLY)), *input2=load(open(argv[2], O_RDONLY));
    int ct_same=0, ct_total=512*512;
    for(int y=0;y<512;++y)for(int x=0;x<512;++x)
        if(INPUT(1,x,y)==INPUT(2,x,y))++ct_same;
    printf("Match %d/%d %.3f%%\n", ct_same, ct_total, (float)ct_same*100.0f/ct_total);
    return 0;
}
