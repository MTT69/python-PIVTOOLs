int main(void){
    int a[64];
    for(int i=0;i<64;++i) a[i]=i*3;
    int s=0;
    for(int i=0;i<64;++i) s+=a[i];
    return s;
}
