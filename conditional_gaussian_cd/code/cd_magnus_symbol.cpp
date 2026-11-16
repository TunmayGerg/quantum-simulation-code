// Principal cubic Magnus symbol for a driven qubit and a single oscillator.
// X,P are commuting symbols here: their commutator contributes lower-degree
// oscillator terms, which must subsequently be checked by full propagation.
#include <complex>
#include <cmath>
#include <array>
using V=std::array<double,3>;
using M=std::array<V,3>;
V add(V a,V b){for(int i=0;i<3;++i)a[i]+=b[i];return a;}
V scale(V a,double s){for(double &v:a)v*=s;return a;}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
V apply(M a,V v){V r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)r[i]+=a[i][j]*v[j];return r;}
M mul(M a,M b){M r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)r[i][j]+=a[i][k]*b[k][j];return r;}
M rotation(double x,double y,double angle){
    double n=std::hypot(x,y);if(n==0)return {{{1,0,0},{0,1,0},{0,0,1}}};
    V v={x/n,y/n,0};double c=std::cos(angle),s=std::sin(angle);M r{};
    for(int j=0;j<3;++j){V e{};e[j]=1;V w=add(add(scale(e,c),scale(cross(v,e),s)),scale(v,(1-c)*v[j]));for(int i=0;i<3;++i)r[i][j]=w[i];}
    return r;
}
extern "C" void cd_magnus_symbol(int count,double dt,int steps,
    const std::complex<double>* q,const std::complex<double>* pump,double* out){
    V first[4]{},second[4]{},third[4]{};
    const double x[4]={1,0,1,1},p[4]={0,1,1,-1};
    M R={{{1,0,0},{0,1,0},{0,0,1}}};double h=dt/steps;
    for(int sample=0;sample<count;++sample){
        double qx=q[sample].real(),qy=q[sample].imag(),angle=-2*std::abs(q[sample])*h;
        M half=rotation(qx,qy,angle/2),full=rotation(qx,qy,angle);
        V u={pump[sample].real()/2,pump[sample].imag()/2,0};
        V v={-pump[sample].imag()/2,pump[sample].real()/2,0};
        for(int step=0;step<steps;++step){
            M midpoint=mul(R,half);V U=apply(midpoint,u),Vv=apply(midpoint,v);
            for(int k=0;k<4;++k){
                V a=scale(add(scale(U,x[k]),scale(Vv,p[k])),h);
                V aa=cross(a,first[k]);
                V cubic=add(cross(a,second[k]),scale(add(cross(a,aa),cross(first[k],cross(first[k],a))),1./3));
                third[k]=add(third[k],cubic);second[k]=add(second[k],aa);first[k]=add(first[k],a);
            }
            R=mul(R,full);
        }
    }
    for(int k=0;k<4;++k)for(int j=0;j<3;++j){out[k*9+j]=first[k][j];out[k*9+3+j]=second[k][j];out[k*9+6+j]=third[k][j];}
}
