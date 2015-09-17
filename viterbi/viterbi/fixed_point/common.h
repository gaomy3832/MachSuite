#ifndef COMMON_H_
#define COMMON_H_


/*************************************************
 * Data format and precision.
 *************************************************/
#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>

#define W_DATA    48
#define IW_DATA   32
typedef ap_fixed<W_DATA, IW_DATA> fixed_t;

inline ap_int<IW_DATA> to_int(const fixed_t& a) {
    return a.range(W_DATA-1, W_DATA-IW_DATA);
}


/*************************************************
 * Interface.
 *************************************************/
#include <hls_stream.h>

// Output queue format
#define NQWIDTH 6
typedef ap_uint<NQWIDTH> qid_t;

template<typename out_t>
struct qid_out_t {
    qid_t qid;
    out_t out;

    qid_out_t() : qid(0), out() {};
    qid_out_t(qid_t q, out_t o) : qid(q), out(o) {};
};

// Forced register
// see http://forums.xilinx.com/t5/High-Level-Synthesis-HLS/Forcing-a-register-stage/td-p/632339
// and <Vivado_HLS>/include/hls/utils/x_hls_utils.h
template<class T>
T forced_reg(T in) {
#pragma HLS INLINE self off
#pragma HLS INTERFACE ap_none port=return register
    return in;
}


/*************************************************
 * Operators.
 *************************************************/
// 48 x 48 adder/subtractor, avoid overflow.
inline fixed_t operator+(const fixed_t& a, const fixed_t& b) {
    ap_int<W_DATA> aa = a(W_DATA-1, 0);
    ap_int<W_DATA> bb = b(W_DATA-1, 0);

    ap_int<W_DATA> s = aa + bb;

    fixed_t sum;
    sum.range() = s(W_DATA-1, 0);

    return sum;
}

inline fixed_t operator-(const fixed_t& a, const fixed_t& b) {
    ap_int<W_DATA> aa = a(W_DATA-1, 0);
    ap_int<W_DATA> bb = b(W_DATA-1, 0);

    ap_int<W_DATA> d = aa - bb;

    fixed_t diff;
    diff.range() = d(W_DATA-1, 0);

    return diff;
}

// 48 x 48 multiplier.
inline fixed_t operator*(const fixed_t& a, const fixed_t& b) {
    ap_int<W_DATA> aa = a(W_DATA-1, 0);
    ap_int<W_DATA> bb = b(W_DATA-1, 0);

    ap_int<2*W_DATA> p = aa * bb;
#pragma HLS RESOURCE variable=p core=Mul

    fixed_t prod;
    prod.range() = p(2*W_DATA-IW_DATA, W_DATA-IW_DATA);

    return prod;
}

// Pipelined units.
template<class T, class U>
inline T pipe_add_input_reg(const T& a, const U& b) {
    return forced_reg(a) + forced_reg(b);
}

template<class T, class U>
inline T pipe_sub_input_reg(const T& a, const U& b) {
    return forced_reg(a) - forced_reg(b);
}

template<class T, class U>
inline T pipe_add_output_reg(const T& a, const U& b) {
    // separate op and register
    // define register input type to avoid overflow bit.
    T res = a + b;
    return forced_reg(res);
}

template<class T, class U>
inline T pipe_sub_output_reg(const T& a, const U& b) {
    // separate op and register
    // define register input type to avoid overflow bit.
    T res = a - b;
    return forced_reg(res);
}

template<class T, class U>
inline T pipe_mult(const T& a, const U& b) {
    return forced_reg(forced_reg(a) * forced_reg(b));
}


/*************************************************
 * Helper functions.
 *************************************************/
inline bool check_eq_fixed(double vtrue, fixed_t v, double tol) {
    return fabs((double)v - vtrue) < tol;
}

#endif // COMMON_H_
