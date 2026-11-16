// Sparse fourth-order commutator-free Magnus propagation of the driven LINC.
// Units ns and rad/ns. Full cosine matrix C is precomputed in Python.
// Two exponentials per step; each is evaluated by a bounded Taylor series.
// Waveform envelopes are linearly reconstructed from the known RFDC samples.
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
using C=std::complex<double>;
extern "C" void replay_linc(int size,int columns,int count,int steps,int order,
    double sample_dt,double dt,double pump_frequency,double qubit_frequency,
    double flux_per_dac,double drive_factor,double qubit_per_dac,
    const double* energies,const int* rowptr,const int* colidx,const double* values,
    const C* pump,const C* qubit,const C* input,C* output) {
    const int total=size*columns;
    std::vector<double> re(total),im(total),tr(total),ti(total),nr(total),ni(total),sr(total),si(total);
    for(int k=0;k<total;++k){re[k]=input[k].real();im[k]=input[k].imag();}
    auto envelope=[&](const C* data,double t){
        double x=t/sample_dt;
        int index=int(std::floor(x));
        if(index<0 || index>=count)return C(0.,0.);
        C next=index+1<count?data[index+1]:C(0.,0.);
        return data[index]+(x-index)*(next-data[index]);
    };
    const double root=std::sqrt(3.);
    const double a1=(3.-2.*root)/12.,a2=(3.+2.*root)/12.;
    const double c1=.5-root/6.,c2=.5+root/6.;
    for(int step=0;step<steps;++step){
        double t1=(step+c1)*dt,t2=(step+c2)*dt;
        auto controls=[&](double t,double& flux,double& q){
            // Positive pump carrier gives the sigma_+ a phase convention.
            double phi=-flux_per_dac*std::real(envelope(pump,t)*std::exp(C(0.,pump_frequency*t)));
            flux=drive_factor*std::sin(phi);
            // Actual qubit IQ multiplies sigma_-; the physical real carrier
            // has the conjugated raising coefficient in the resonant frame.
            q=2.*qubit_per_dac*std::real(std::conj(envelope(qubit,t))*std::exp(C(0.,-qubit_frequency*t)));
        };
        double f1,f2,q1,q2;controls(t1,f1,q1);controls(t2,f2,q2);
        for(int stage=0;stage<2;++stage){
            // Rightmost exponential acts first.
            double b1=stage==0?a2:a1,b2=stage==0?a1:a2;
            double f=b1*f1+b2*f2,q=b1*q1+b2*q2;
            tr=re;ti=im;sr=re;si=im;
            for(int power=1;power<=order;++power){
                std::fill(nr.begin(),nr.end(),0.);std::fill(ni.begin(),ni.end(),0.);
                for(int row=0;row<size;++row){
                    int offset=row*columns;
                    for(int col=0;col<columns;++col){
                        nr[offset+col]=.5*energies[row]*tr[offset+col]+q*tr[(row^1)*columns+col];
                        ni[offset+col]=.5*energies[row]*ti[offset+col]+q*ti[(row^1)*columns+col];
                    }
                    for(int j=rowptr[row];j<rowptr[row+1];++j){
                        int source=colidx[j]*columns;double value=f*values[j];
                        for(int col=0;col<columns;++col){
                            nr[offset+col]+=value*tr[source+col];
                            ni[offset+col]+=value*ti[source+col];
                        }
                    }
                }
                double scale=dt/power;
                for(int k=0;k<total;++k){
                    double r=scale*ni[k],i=-scale*nr[k];
                    tr[k]=r;ti[k]=i;sr[k]+=r;si[k]+=i;
                }
            }
            re.swap(sr);im.swap(si);
        }
    }
    for(int k=0;k<total;++k)output[k]=C(re[k],im[k]);
}

// The same CF4 Hamiltonians with a Chebyshev exponential. A fixed certified
// spectral radius permits one Bessel coefficient table for the whole waveform.
// This typically halves the sparse matrix products versus a Taylor expansion.
extern "C" void replay_linc_chebyshev(int size,int columns,int count,int steps,int order,
    double sample_dt,double dt,double pump_frequency,double qubit_frequency,
    double flux_per_dac,double drive_factor,double qubit_per_dac,
    const double* energies,const int* rowptr,const int* colidx,const double* values,
    const C* pump,const C* qubit,const C* input,C* output) {
    int total=size*columns;
    std::vector<double> re(total),im(total),p0r(total),p0i(total),p1r(total),p1i(total),
                        nr(total),ni(total),sr(total),si(total);
    double emax=0.,cmax=0.,pmax=0.,qmax=0.;
    for(int row=0;row<size;++row){
        emax=std::max(emax,std::abs(energies[row]));double sum=0.;
        for(int j=rowptr[row];j<rowptr[row+1];++j)sum+=std::abs(values[j]);
        cmax=std::max(cmax,sum);
    }
    for(int j=0;j<count;++j){pmax=std::max(pmax,std::abs(pump[j]));qmax=std::max(qmax,std::abs(qubit[j]));}
    double radius=dt*(.5*emax+drive_factor*std::min(1.,flux_per_dac*pmax)*cmax/std::sqrt(3.)+
                      2.*qubit_per_dac*qmax/std::sqrt(3.));
    std::vector<double> coefficients(order+1);
    for(int k=0;k<=order;++k)coefficients[k]=(k==0?1.:2.)*std::cyl_bessel_j(k,radius);
    for(int k=0;k<total;++k){re[k]=input[k].real();im[k]=input[k].imag();}
    auto env=[&](const C* data,double t){
        double x=t/sample_dt;int i=int(std::floor(x));
        if(i<0 || i>=count)return C(0.,0.);
        return data[i]+(x-i)*((i+1<count?data[i+1]:C(0.,0.))-data[i]);
    };
    auto apply=[&](const std::vector<double>& r,const std::vector<double>& i,
                   std::vector<double>& yr,std::vector<double>& yi,double f,double q){
        for(int row=0;row<size;++row){
            int offset=row*columns,other=(row^1)*columns;
            for(int c=0;c<columns;++c){
                yr[offset+c]=.5*energies[row]*r[offset+c]+q*r[other+c];
                yi[offset+c]=.5*energies[row]*i[offset+c]+q*i[other+c];
            }
            for(int j=rowptr[row];j<rowptr[row+1];++j){
                double value=f*values[j];int source=colidx[j]*columns;
                for(int c=0;c<columns;++c){yr[offset+c]+=value*r[source+c];yi[offset+c]+=value*i[source+c];}
            }
        }
    };
    auto add=[&](int k,const std::vector<double>& r,const std::vector<double>& i){
        double a=coefficients[k];
        for(int j=0;j<total;++j){
            if(k%4==0){sr[j]+=a*r[j];si[j]+=a*i[j];}
            if(k%4==1){sr[j]+=a*i[j];si[j]-=a*r[j];}
            if(k%4==2){sr[j]-=a*r[j];si[j]-=a*i[j];}
            if(k%4==3){sr[j]-=a*i[j];si[j]+=a*r[j];}
        }
    };
    double root=std::sqrt(3.),a1=(3.-2*root)/12.,a2=(3.+2*root)/12.;
    for(int step=0;step<steps;++step){
        double t1=(step+.5-root/6.)*dt,t2=(step+.5+root/6.)*dt;
        auto controls=[&](double t,double& f,double& q){
            double phi=-flux_per_dac*std::real(env(pump,t)*std::exp(C(0.,pump_frequency*t)));
            f=drive_factor*std::sin(phi);
            q=2*qubit_per_dac*std::real(std::conj(env(qubit,t))*std::exp(C(0.,-qubit_frequency*t)));
        };
        double f1,f2,q1,q2;controls(t1,f1,q1);controls(t2,f2,q2);
        for(int stage=0;stage<2;++stage){
            double b1=stage==0?a2:a1,b2=stage==0?a1:a2;
            double f=b1*f1+b2*f2,q=b1*q1+b2*q2;
            p0r=re;p0i=im;sr.assign(total,0.);si.assign(total,0.);add(0,p0r,p0i);
            apply(p0r,p0i,p1r,p1i,f,q);
            for(int j=0;j<total;++j){p1r[j]*=dt/radius;p1i[j]*=dt/radius;}
            add(1,p1r,p1i);
            for(int k=2;k<=order;++k){
                apply(p1r,p1i,nr,ni,f,q);
                for(int j=0;j<total;++j){nr[j]=2*dt/radius*nr[j]-p0r[j];ni[j]=2*dt/radius*ni[j]-p0i[j];}
                add(k,nr,ni);p0r.swap(p1r);p0i.swap(p1i);p1r.swap(nr);p1i.swap(ni);
            }
            re.swap(sr);im.swap(si);
        }
    }
    for(int j=0;j<total;++j)output[j]=C(re[j],im[j]);
}
