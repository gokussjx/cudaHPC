#include "standard.h"
int main(int argc, char *argv[]){
    if(argc<2){ fprintf(stderr, "Usage: input.ppm\n"); return 1; }
    int fd=open(argv[1], O_RDONLY); color *p, *a;
    int ct0a=4; struct stat _fstat;
    if (fstat(fd, &_fstat) == -1) { perror("fstat()"); exit(1); }
    a=p=mmap(NULL, _fstat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(!(a[0]=='P'&&a[1]=='6')) { fprintf(stderr, "Input not in P6 format\n"); return 1; }
    do{ p=memchr(p, 0x0a, 64)+1; } while(--ct0a);
    fwrite("P5",2,1,stdout);
    fwrite(a+2,p-a-2,1,stdout);
    for(int i=0;i<DIM*DIM;++i)
        fwrite(&p[i*3], 1, 1, stdout);
    return 0;
}
