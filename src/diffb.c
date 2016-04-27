#include "standard.h"
#ifdef P6
#define INPUT(I,x,y) input##I[((y)*(DIM*3))+(x)*3]
#endif

#ifdef P5
#define INPUT(I,x,y) input##I[((y)*(DIM))+(x)]
#endif

color tmp[3];
color* load(int fd){
    int ct0a=4; struct stat _fstat;
    if (fstat(fd, &_fstat) == -1) { perror("fstat()"); exit(1); }
    color *p=mmap(NULL, _fstat.st_size, PROT_READ, MAP_PRIVATE, fd, 0), *a=p;
    if(!(p[0]=='P'&&p[1]==MagicNumber)) { fprintf(stderr, "Unsupported format\n"); exit(1); }
    do{ p=memchr(p, 0x0a, 64)+1; } while(--ct0a);
    fwrite(a,p-a,1,stdout);
    return p;
}

int diff(color c1, color c2){ 
    int d=(int)c1-(int)c2;
    if(d<0)d=-d;
    return d; // i.e. error <= 4.7%
}

#define BAR 80
char bar[BAR+20];
int main(int argc, char *argv[]){
    if(argc<3){ fprintf(stderr, "Usage: input1.ppm input2.ppm\n"); return 1; }
    color *input1=load(open(argv[1], O_RDONLY)), *input2=load(open(argv[2], O_RDONLY));
    memset(bar, '=', BAR+20);
    for(int y=0;y<DIM;++y)for(int x=0;x<DIM;++x){
#ifdef P6
        memset(tmp, diff(INPUT(1,x,y),INPUT(2,x,y)), 3);
        fwrite(tmp, 3, 1, stdout);
#endif

#ifdef P5
        memset(tmp, diff(INPUT(1,x,y),INPUT(2,x,y)), 1);
        fwrite(tmp, 1, 1, stdout);
#endif
    }
    return 0;
}
