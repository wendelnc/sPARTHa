#ifndef WENO_RECON_HPP
#define WENO_RECON_HPP

#include <iostream> 
#include <memory>
#include <string>
#include <vector>
#include <cmath>

// Parthenon headers
#include <parthenon/package.hpp>

// WENO3 Function
KOKKOS_INLINE_FUNCTION
Real WENO3(const Real &q_im1, const Real &q_i, const Real &q_ip1) {

    const Real epsilon = 1E-10;

    Real beta[2]; // (2.62) 
    beta[0] = SQR(q_ip1 - q_i);
    beta[1] = SQR(q_i - q_im1);

    Real indicator[2]; // fraction part of (2.59) 
    indicator[0] = 1 / SQR(epsilon + beta[0]);
    indicator[1] = 1 / SQR(epsilon + beta[1]);

    // // compute qL_ip1
    Real f[2]; // polynomial based on constants c_{r,j} in Table 2.1 
    // Factor of 1/2 in coefficients of f[] array applied to alpha_sum to reduce divisions
    f[0] = q_i + q_ip1;
    f[1] = -q_im1 + 3.0 * q_i;

    Real alpha[2]; // (2.59)
    alpha[0] = indicator[0] * 2.0 / 3.0;
    alpha[1] = indicator[1] * 1.0 / 3.0;
    Real alpha_sum = 2.0 * (alpha[0] + alpha[1]);

    Real wn = (alpha[0] * f[0] + alpha[1] * f[1]) / alpha_sum; // (2.52) 

    return wn;

} // WENO3


// WENO5 Function
KOKKOS_INLINE_FUNCTION
Real WENO5(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1, const Real &q_ip2) {
    
    const Real epsilon = 1E-10;

    Real beta[3]; // (2.63) 
    beta[0] = (13.0/12.0)*SQR(q_i - 2*q_ip1 + q_ip2) + (1.0/4.0)*SQR(3*q_i - 4*q_ip1 + q_ip2);
    beta[1] = (13.0/12.0)*SQR(q_im1 - 2*q_i + q_ip1) + (1.0/4.0)*SQR(q_im1 - q_ip1);
    beta[2] = (13.0/12.0)*SQR(q_im2 - 2*q_im1 + q_i) + (1.0/4.0)*SQR(q_im2 - 4*q_im1 + 3*q_i);

    Real indicator[3]; // fraction part of (2.59) 
    indicator[0] = 1 / SQR(epsilon + beta[0]);
    indicator[1] = 1 / SQR(epsilon + beta[1]);
    indicator[2] = 1 / SQR(epsilon + beta[2]);

    // indicator[0] = 1.0 / std::pow(epsilon + beta[0], 3);
    // indicator[1] = 1.0 / std::pow(epsilon + beta[1], 3);
    // indicator[2] = 1.0 / std::pow(epsilon + beta[2], 3);

    // compute qL_ip1
    Real f[3]; // polynomial based on constants c_{r,j} in Table 2.1 
    // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
    f[0] =  2*q_i   + 5*q_ip1 - q_ip2;
    f[1] = -1*q_im1 + 5*q_i   + 2*q_ip1;
    f[2] =  2*q_im2 - 7*q_im1 + 11*q_i;

    Real alpha[3]; // (2.59) & below (2.54)
    alpha[0] = indicator[0] * 3.0 / 10.0;
    alpha[1] = indicator[1] * 6.0 / 10.0;
    alpha[2] = indicator[2] * 1.0 / 10.0;

    Real alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);

    Real wn = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2]) / alpha_sum; // (2.52) 

    return wn;

} // WENO5

// WENO7 Function
KOKKOS_INLINE_FUNCTION
Real WENO7(const Real &v_im3, const Real &v_im2, const Real &v_im1, const Real &v_i, const Real &v_ip1, const Real &v_ip2, const Real &v_ip3) {
    
    const Real epsilon = 1E-10;

    Real beta[4]; // flipped from MATLAB code where beta[0] is beta_3
    // beta[0] = (15943*v_i*v_i)/960 - (36709*v_i*v_ip1)/480 + (28429*v_i*v_ip2)/480 - (7663*v_i*v_ip3)/480 + (86447*v_ip1*v_ip1)/960 - (68407*v_ip1*v_ip2)/480 + (6223*v_ip1*v_ip3)/160 + (55247*v_ip2*v_ip2)/960 - (15269*v_ip2*v_ip3)/480 + (1421*v_ip3*v_ip3)/320;
    // beta[1] = (22847*v_i*v_i)/960 - (9389*v_i*v_im1)/480  - (18367*v_i*v_ip1)/480 + (4909*v_i*v_ip2)/480 + (1421*v_im1*v_im1)/320  + (2303*v_im1*v_ip1)/160  - (1783*v_im1*v_ip2)/480 + (15887*v_ip1*v_ip1)/960 - (4429*v_ip1*v_ip2)/480  + (1303*v_ip2*v_ip2)/960;
    // beta[2] = (10847*v_i*v_i)/960 - (10327*v_i*v_im1)/480 + (2909*v_i*v_im2)/480  - (1143*v_i*v_ip1)/160 + (10847*v_im1*v_im1)/960 - (1143*v_im1*v_im2)/160  + (2909*v_im1*v_ip1)/480 + (1303*v_im2*v_im2)/960  - (261*v_im2*v_ip1)/160   + (1303*v_ip1*v_ip1)/960;
    // beta[3] = (1421*v_i*v_i)/320  - (9389*v_i*v_im1)/480  + (2303*v_i*v_im2)/160  - (1783*v_i*v_im3)/480 + (22847*v_im1*v_im1)/960 - (18367*v_im1*v_im2)/480 + (4909*v_im1*v_im3)/480 + (15887*v_im2*v_im2)/960 - (4429*v_im2*v_im3)/480  + (1303*v_im3*v_im3)/960;

    // Balsara and Shu (III.a.)
    beta[0] =   v_i*(2107*v_i   - 9402*v_ip1 + 7042*v_ip2 - 1854*v_ip3 ) + v_ip1*(11003*v_ip1 - 17246*v_ip2 + 4642*v_ip3 ) + v_ip2*( 7043*v_ip2 - 3882*v_ip3 ) +  547*v_ip3*v_ip3;
    beta[1] = v_im1*( 547*v_im1 - 2522*v_i   + 1922*v_ip1 -  494*v_ip2 ) +   v_i*( 3443*v_i   -  5966*v_ip1 + 1602*v_ip2 ) + v_ip1*( 2843*v_ip1 - 1642*v_ip2 ) +  267*v_ip2*v_ip2;
    beta[2] = v_im2*( 267*v_im2 - 1642*v_im1 + 1602*v_i   -  494*v_ip1 ) + v_im1*( 2843*v_im1 -  5966*v_i   + 1922*v_ip1 ) +   v_i*( 3443*v_i   - 2522*v_ip1 ) +  547*v_ip1*v_ip1;
    beta[3] = v_im3*( 547*v_im3 - 3882*v_im2 + 4642*v_im1 - 1854*v_i   ) + v_im2*( 7043*v_im2 - 17246*v_im1 + 7042*v_i   ) + v_im1*(11003*v_im1 - 9402*v_i   ) + 2107*v_i*v_i;
   

    Real indicator[4]; // fraction part of (2.59) 
    indicator[0] = 1 / SQR(epsilon + beta[0]);
    indicator[1] = 1 / SQR(epsilon + beta[1]);
    indicator[2] = 1 / SQR(epsilon + beta[2]);
    indicator[3] = 1 / SQR(epsilon + beta[3]);

    // indicator[0] = 1.0 / std::pow(epsilon + beta[0], 4);
    // indicator[1] = 1.0 / std::pow(epsilon + beta[1], 4);
    // indicator[2] = 1.0 / std::pow(epsilon + beta[2], 4);
    // indicator[3] = 1.0 / std::pow(epsilon + beta[3], 4);

    // compute qL_ip1
    Real f[4]; // polynomial based on constants c_{r,j} in Table 2.1 
    // Factor of 1/12 in coefficients of f[] array applied to alpha_sum to reduce divisions
    f[0] =  3*v_i   + 13*v_ip1 -  5*v_ip2 +   v_ip3;
    f[1] =   -v_im1 +  7*v_i   +  7*v_ip1 -   v_ip2;
    f[2] =    v_im2 -  5*v_im1 + 13*v_i   + 3*v_ip1;
    f[3] = -3*v_im3 + 13*v_im2 - 23*v_im1 + 25*v_i;

    Real alpha[4]; // (2.59)
    alpha[0] = indicator[0] *  4.0 / 35.0;
    alpha[1] = indicator[1] * 18.0 / 35.0;
    alpha[2] = indicator[2] * 12.0 / 35.0;
    alpha[3] = indicator[3] *  1.0 / 35.0;

    Real alpha_sum = 12.0 * (alpha[0] + alpha[1] + alpha[2] + alpha[3]);

    Real wn = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2] + alpha[3] * f[3]) / alpha_sum; // (2.52) 

    return wn;

} // WENO7

// WENO9 Function
KOKKOS_INLINE_FUNCTION
Real WENO9(const Real &v_im4, const Real &v_im3, const Real &v_im2, const Real &v_im1, const Real &v_i, const Real &v_ip1, const Real &v_ip2, const Real &v_ip3, const Real &v_ip4) {
    
    const Real epsilon = 1E-10;

    Real beta[5]; // flipped from MATLAB code where beta[0] is beta_4
    // beta[0] = (3693653*v_i*v_i)/80640 - (2834627*v_i*v_ip1)/10080 + (2246389*v_i*v_ip2)/6720 - (1850819*v_i*v_ip3)/10080 + (1569797*v_i*v_ip4)/40320 + (8907527*v_ip1*v_ip1)/20160 - (3594209*v_ip1*v_ip2)/3360 + (5988821*v_ip1*v_ip3)/10080 - (639547*v_ip1*v_ip4)/5040  + (551713*v_ip2*v_ip2)/840   - (824853*v_ip2*v_ip3)/1120  + (1063739*v_ip2*v_ip4)/6720 + (4190927*v_ip3*v_ip3)/20160 - (226313*v_ip3*v_ip4)/2520  + (785153*v_ip4*v_ip4)/80640;  
    // beta[1] = (1906127*v_i*v_i)/20160 - (18406*v_i*v_im1)/315     - (776359*v_i*v_ip1)/3360  + (1281461*v_i*v_ip2)/10080 - (269519*v_i*v_ip3)/10080  + (785153*v_im1*v_im1)/80640  + (151953*v_im1*v_ip1)/2240  - (183637*v_im1*v_ip2)/5040   + (304757*v_im1*v_ip3)/40320 + (123103*v_ip1*v_ip1)/840   - (554809*v_ip1*v_ip2)/3360  + (78943*v_ip1*v_ip3)/2240   + (957767*v_ip2*v_ip2)/20160  - (207527*v_ip2*v_ip3)/10080 + (182453*v_ip3*v_ip3)/80640;
    // beta[2] = (22439*v_i*v_i)/420     - (84463*v_i*v_im1)/1120    + (124409*v_i*v_im2)/6720  - (214259*v_i*v_ip1)/3360   + (92839*v_i*v_ip2)/6720    + (574727*v_im1*v_im1)/20160  - (151877*v_im1*v_im2)/10080 + (426341*v_im1*v_ip1)/10080  - (2782*v_im1*v_ip2)/315     + (182453*v_im2*v_im2)/80640 - (100889*v_im2*v_ip1)/10080 + (82157*v_im2*v_ip2)/40320  + (410927*v_ip1*v_ip1)/20160  - (46801*v_ip1*v_ip2)/5040   + (91313*v_ip2*v_ip2)/80640;
    // beta[3] = (574727*v_i*v_i)/20160  - (84463*v_i*v_im1)/1120    + (426341*v_i*v_im2)/10080 - (2782*v_i*v_im3)/315      - (151877*v_i*v_ip1)/10080  + (22439*v_im1*v_im1)/420     - (214259*v_im1*v_im2)/3360  + (92839*v_im1*v_im3)/6720    + (124409*v_im1*v_ip1)/6720  + (410927*v_im2*v_im2)/20160 - (46801*v_im2*v_im3)/5040   - (100889*v_im2*v_ip1)/10080 + (91313*v_im3*v_im3)/80640   + (82157*v_im3*v_ip1)/40320  + (182453*v_ip1*v_ip1)/80640;
    // beta[4] = (785153*v_i*v_i)/80640  - (18406*v_i*v_im1)/315     + (151953*v_i*v_im2)/2240  - (183637*v_i*v_im3)/5040   + (304757*v_i*v_im4)/40320  + (1906127*v_im1*v_im1)/20160 - (776359*v_im1*v_im2)/3360  + (1281461*v_im1*v_im3)/10080 - (269519*v_im1*v_im4)/10080 + (123103*v_im2*v_im2)/840   - (554809*v_im2*v_im3)/3360  + (78943*v_im2*v_im4)/2240   + (957767*v_im3*v_im3)/20160  - (207527*v_im3*v_im4)/10080 + (182453*v_im4*v_im4)/80640;

    beta[0] =   v_i*(107918*v_i   - 649501*v_ip1 + 758823*v_ip2 - 411487*v_ip3 + 86329*v_ip4) + v_ip1*(1020563*v_ip1 - 2462076*v_ip2 + 1358458*v_ip3 - 288007*v_ip4) + v_ip2*(1521393*v_ip2 - 1704396*v_ip3 + 364863*v_ip4) + v_ip3*( 482963*v_ip3 - 208501*v_ip4) +  22658*v_ip4*v_ip4;
    beta[1] = v_im1*( 22658*v_im1 - 140251*v_i   + 165153*v_ip1 -  88297*v_ip2 + 18079*v_ip3) +   v_i*( 242723*v_i   -  611976*v_ip1 +  337018*v_ip2 -  70237*v_ip3) + v_ip1*( 406293*v_ip1 -  464976*v_ip2 +  99213*v_ip3) + v_ip2*( 138563*v_ip2 -  60871*v_ip3) +   6908*v_ip3*v_ip3;
    beta[2] = v_im2*(  6908*v_im2 -  51001*v_im1 +  67923*v_i   -  38947*v_ip1 +  8209*v_ip2) + v_im1*( 104963*v_im1 -  299076*v_i   +  179098*v_ip1 -  38947*v_ip2) +   v_i*( 231153*v_i   -  299076*v_ip1 +  67923*v_ip2) + v_ip1*( 104963*v_ip1 -  51001*v_ip2) +   6908*v_ip2*v_ip2;
    beta[3] = v_im3*(  6908*v_im3 -  60871*v_im2 +  99213*v_im1 -  70237*v_i   + 18079*v_ip1) + v_im2*( 138563*v_im2 -  464976*v_im1 +  337018*v_i   -  88297*v_ip1) + v_im1*( 406293*v_im1 -  611976*v_i   + 165153*v_ip1) +   v_i*( 242723*v_i   - 140251*v_ip1) +  22658*v_ip1*v_ip1;
    beta[4] = v_im4*( 22658*v_im4 - 208501*v_im3 + 364863*v_im2 - 288007*v_im1 + 86329*v_i)   + v_im3*( 482963*v_im3 - 1704396*v_im2 + 1358458*v_im1 - 411487*v_i)   + v_im2*(1521393*v_im2 - 2462076*v_im1 + 758823*v_i)   + v_im1*(1020563*v_im1 - 649501*v_i)   + 107918*v_i*v_i;
    
    Real indicator[5]; // fraction part of (2.59) 
    indicator[0] = 1 / SQR(epsilon + beta[0]);
    indicator[1] = 1 / SQR(epsilon + beta[1]);
    indicator[2] = 1 / SQR(epsilon + beta[2]);
    indicator[3] = 1 / SQR(epsilon + beta[3]);
    indicator[4] = 1 / SQR(epsilon + beta[4]);

    // indicator[0] = 1.0 / std::pow(epsilon + beta[0], 4);
    // indicator[1] = 1.0 / std::pow(epsilon + beta[1], 4);
    // indicator[2] = 1.0 / std::pow(epsilon + beta[2], 4);
    // indicator[3] = 1.0 / std::pow(epsilon + beta[3], 4);
    // indicator[4] = 1.0 / std::pow(epsilon + beta[4], 4);


    // compute qL_ip1
    Real f[5]; // polynomial based on constants c_{r,j} in Table 2.1 
    // Factor of 1/60 in coefficients of f[] array applied to alpha_sum to reduce divisions
    f[0] = 12*v_i   + 77*v_ip1 -  43*v_ip2 +  17*v_ip3 -   3*v_ip4;
    f[1] = -3*v_im1 + 27*v_i   +  47*v_ip1 -  13*v_ip2 +   2*v_ip3;
    f[2] =  2*v_im2 - 13*v_im1 +  47*v_i   +  27*v_ip1 -   3*v_ip2;
    f[3] = -3*v_im3 + 17*v_im2 -  43*v_im1 +  77*v_i   +  12*v_ip1;
    f[4] = 12*v_im4 - 63*v_im3 + 137*v_im2 - 163*v_im1 + 137*v_i;

    Real alpha[5]; // (2.59)
    alpha[0] = indicator[0] *  5.0 / 126.0;
    alpha[1] = indicator[1] * 40.0 / 126.0;
    alpha[2] = indicator[2] * 60.0 / 126.0;
    alpha[3] = indicator[3] * 20.0 / 126.0;
    alpha[4] = indicator[4] *  1.0 / 126.0;

    Real alpha_sum = 60.0 * (alpha[0] + alpha[1] + alpha[2] + alpha[3] + alpha[4]);

    Real wn = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2] + alpha[3] * f[3] + alpha[4] * f[4]) / alpha_sum; // (2.52) 

    return wn;

} // WENO9

#endif // WENO_RECON_HPP
