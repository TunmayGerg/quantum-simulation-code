// Independent sample-by-sample Schrodinger evolution. No cached pulse matrices.
// Units: microseconds and radians. Basis: g0,e0,g1,e1,...;18 input columns.
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
using Complex = std::complex<double>;

extern "C" void propagate_native_samples(
    int levels, int columns, int samples, double dt, double chi, int order,
    const Complex* qubit, const Complex* sideband,
    const Complex* input, Complex* output) {
    const int size = 2*levels*columns;
    std::vector<Complex> state(input,input+size), term(size), next(size), sum(size);
    for (int sample=0;sample<samples;++sample) {
        term=state;
        sum=state;
        for (int power=1;power<=order;++power) {
            std::fill(next.begin(),next.end(),Complex(0.,0.));
            for (int n=0;n<levels;++n) {
                const int g=2*n*columns;
                const int e=(2*n+1)*columns;
                for (int c=0;c<columns;++c) {
                    next[g+c] += std::conj(qubit[sample])*term[e+c];
                    next[e+c] += qubit[sample]*term[g+c] + chi*n*term[e+c];
                    if (n+1<levels) {
                        const int gp=(2*n+2)*columns;
                        const Complex coupling=sideband[sample]*std::sqrt(n+1.);
                        next[e+c] += coupling*term[gp+c];
                        next[gp+c] += std::conj(coupling)*term[e+c];
                    }
                }
            }
            const Complex factor(0.,-dt/power);
            for (int k=0;k<size;++k) {
                next[k] *= factor;
                sum[k] += next[k];
            }
            term.swap(next);
        }
        state.swap(sum);
    }
    std::copy(state.begin(),state.end(),output);
}
