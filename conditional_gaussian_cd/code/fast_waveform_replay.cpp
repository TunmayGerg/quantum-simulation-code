// Sample-exact Taylor action with separate real/imaginary work arrays.
// No fast-math flags; the independent generic complex solver is retained.
#include <complex>
#include <vector>
#include <cmath>
using C=std::complex<double>;
extern "C" void fast_waveform_replay(int levels,int columns,int samples,double dt,
    double chi,int order,const C* qubit,const C* sideband,const C* input,C* output) {
    const int size=2*levels*columns;
    std::vector<double> r(size),im(size),tr(size),ti(size),nr(size),ni(size),sr(size),si(size);
    std::vector<double> roots(levels+1);
    for(int n=0;n<=levels;++n) roots[n]=std::sqrt(double(n));
    for(int k=0;k<size;++k) {r[k]=input[k].real();im[k]=input[k].imag();}
    for(int sample=0;sample<samples;++sample) {
        tr=r;ti=im;sr=r;si=im;
        const double qr=qubit[sample].real(),qi=qubit[sample].imag();
        const double br=sideband[sample].real(),bi=sideband[sample].imag();
        for(int power=1;power<=order;++power) {
            const double scale=dt/power;
            for(int n=0;n<levels;++n) {
                const int g=2*n*columns,e=g+columns;
                const int prev_e=n>0 ? e-2*columns : e;
                const int next_g=n+1<levels ? g+2*columns : g;
                const double pr=br*roots[n],pi=bi*roots[n];
                const double ar=n+1<levels ? br*roots[n+1] : 0.;
                const double ai=n+1<levels ? bi*roots[n+1] : 0.;
                const double energy=chi*n;
                // Every column in this loop is independent and vectorises.
                for(int c=0;c<columns;++c) {
                    const double hg_r=qr*tr[e+c]+qi*ti[e+c]+pr*tr[prev_e+c]+pi*ti[prev_e+c];
                    const double hg_i=qr*ti[e+c]-qi*tr[e+c]+pr*ti[prev_e+c]-pi*tr[prev_e+c];
                    const double he_r=qr*tr[g+c]-qi*ti[g+c]+energy*tr[e+c]
                                      +ar*tr[next_g+c]-ai*ti[next_g+c];
                    const double he_i=qr*ti[g+c]+qi*tr[g+c]+energy*ti[e+c]
                                      +ar*ti[next_g+c]+ai*tr[next_g+c];
                    nr[g+c]=scale*hg_i;ni[g+c]=-scale*hg_r;
                    nr[e+c]=scale*he_i;ni[e+c]=-scale*he_r;
                    sr[g+c]+=nr[g+c];si[g+c]+=ni[g+c];
                    sr[e+c]+=nr[e+c];si[e+c]+=ni[e+c];
                }
            }
            tr.swap(nr);ti.swap(ni);
        }
        r.swap(sr);im.swap(si);
    }
    for(int k=0;k<size;++k) output[k]=C(r[k],im[k]);
}
