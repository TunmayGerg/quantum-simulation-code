// Backward control gradient for the actual sample Hamiltonians.
// The forward solve is independent. Both state and adjoint are reconstructed
// backwards, avoiding a time-by-state history buffer. Endpoint quadrature of
// the local Frechet integral has O(dt^2) global gradient error; it is checked
// against parameter finite differences before use. Gate scores use the full
// forward Taylor solve, not this gradient approximation.
#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>
using C=std::complex<double>;
extern "C" void backward_waveform(int levels,int columns,int samples,int order,
 double dt,double trace_real,double trace_imag,const C* q,const C* s,
 const C* output,const C* adjoint,double* gradient) {
 int width=2*columns,total=2*levels*width;
 std::vector<double> re(total),im(total),tr(total),ti(total),nr(total),ni(total),sr(total),si(total),root(levels);
 for(int n=0;n<levels;++n)root[n]=std::sqrt(n+1.);
 for(int row=0;row<2*levels;++row)for(int c=0;c<columns;++c){
  int k=row*width+c,j=row*columns+c;
  re[k]=output[j].real();im[k]=output[j].imag();
  re[k+columns]=adjoint[j].real();im[k+columns]=adjoint[j].imag();
 }
 auto moments=[&](C* result){
  std::fill(result,result+4,C(0.,0.));
  for(int n=0;n<levels;++n)for(int c=0;c<columns;++c){
   int g=2*n*width+c,e=(2*n+1)*width+c;
   C left(re[e+columns]*re[g]+im[e+columns]*im[g],re[e+columns]*im[g]-im[e+columns]*re[g]);
   C right(re[g+columns]*re[e]+im[g+columns]*im[e],re[g+columns]*im[e]-im[g+columns]*re[e]);
   result[0]+=left+right;result[1]+=C(0.,1.)*(left-right);
   if(n+1<levels){
    int gp=(2*n+2)*width+c;
    left=C(re[e+columns]*re[gp]+im[e+columns]*im[gp],re[e+columns]*im[gp]-im[e+columns]*re[gp]);
    right=C(re[gp+columns]*re[e]+im[gp+columns]*im[e],re[gp+columns]*im[e]-im[gp+columns]*re[e]);
    result[2]+=root[n]*(left+right);result[3]+=C(0.,root[n])*(left-right);
   }
  }
 };
 C conjugate_trace=std::conj(C(trace_real,trace_imag));
 for(int sample=samples-1;sample>=0;--sample){
  C after[4],before[4];moments(after);
  tr=re;ti=im;sr=re;si=im;
  double qr=q[sample].real(),qi=q[sample].imag(),pr=s[sample].real(),pi=s[sample].imag();
  for(int power=1;power<=order;++power){
   std::fill(nr.begin(),nr.end(),0.);std::fill(ni.begin(),ni.end(),0.);
   for(int n=0;n<levels;++n){
    int g=2*n*width,e=(2*n+1)*width;
    for(int c=0;c<width;++c){
     nr[g+c]+=qr*tr[e+c]+qi*ti[e+c];ni[g+c]+=qr*ti[e+c]-qi*tr[e+c];
     nr[e+c]+=qr*tr[g+c]-qi*ti[g+c];ni[e+c]+=qr*ti[g+c]+qi*tr[g+c];
     if(n+1<levels){
      int gp=(2*n+2)*width;double r=pr*root[n],i=pi*root[n];
      nr[e+c]+=r*tr[gp+c]-i*ti[gp+c];ni[e+c]+=r*ti[gp+c]+i*tr[gp+c];
      nr[gp+c]+=r*tr[e+c]+i*ti[e+c];ni[gp+c]+=r*ti[e+c]-i*tr[e+c];
     }
    }
   }
   double factor=dt/power;
   for(int k=0;k<total;++k){double r=-factor*ni[k],i=factor*nr[k];tr[k]=r;ti[k]=i;sr[k]+=r;si[k]+=i;}
  }
  re.swap(sr);im.swap(si);moments(before);
  for(int control=0;control<4;++control)
   gradient[4*sample+control]=-2*std::real(conjugate_trace*C(0.,-dt/2)*(after[control]+before[control]))/(columns*columns);
 }
}
