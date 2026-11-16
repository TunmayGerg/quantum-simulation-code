// Taylor coefficients in lambda for H(t)=H_qubit(t)+lambda H_JC(t).
// Every supplied fine sample is held exactly as in the waveform propagator.
// The three coefficient matrices evolve together; no fitted target enters.
#include <complex>
#include <vector>
#include <cmath>
using C=std::complex<double>;
extern "C" void jc_coupling_series(int levels,int samples,double dt,int order,
    const C* q,const C* s,C* output){
    int d=2*levels,block=d*d,total=3*block;
    std::vector<C> state(total),term(total),next(total),sum(total);
    std::vector<double> root(levels+1);for(int n=0;n<=levels;++n)root[n]=std::sqrt(double(n));
    for(int i=0;i<d;++i)state[i*d+i]=1.;
    for(int t=0;t<samples;++t){
        term=state;sum=state;
        for(int power=1;power<=order;++power){
            C factor(0.,-dt/power);
            for(int m=0;m<3;++m)for(int n=0;n<levels;++n){
                int g=m*block+2*n*d,e=g+d,prev=(m-1)*block+(2*n-1)*d,nxt=(m-1)*block+(2*n+2)*d;
                for(int c=0;c<d;++c){
                    C hg=std::conj(q[t])*term[e+c],he=q[t]*term[g+c];
                    if(m>0&&n>0)hg+=std::conj(s[t])*root[n]*term[prev+c];
                    if(m>0&&n+1<levels)he+=s[t]*root[n+1]*term[nxt+c];
                    next[g+c]=factor*hg;next[e+c]=factor*he;
                    sum[g+c]+=next[g+c];sum[e+c]+=next[e+c];
                }
            }
            term.swap(next);
        }
        state.swap(sum);
    }
    std::copy(state.begin(),state.end(),output);
}
