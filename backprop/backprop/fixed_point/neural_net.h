#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include "common.h"

#define PARTWIDTH 8
#define PART_SIZE (1<<PARTWIDTH)

struct nid_t {
    ap_uint<48> id;

    nid_t(uint32_t t = 0, uint32_t p = 0, uint32_t i = 0) {
        ap_uint<16> tid = t & ((1<<16) - 1);
        ap_uint<32-PARTWIDTH> pid = p & ((1<<(32-PARTWIDTH)) - 1);
        ap_uint<PARTWIDTH> inp = i & ((1<<(PARTWIDTH)) - 1);

        id = (tid, pid, inp);
    }

    ap_uint<16> tid() { return id(47, 32); }
    ap_uint<32-PARTWIDTH> pid() { return id(31, PARTWIDTH); }
    ap_uint<PARTWIDTH> inp() { return id(PARTWIDTH-1, 0); }

    bool operator==(const nid_t& o) { return id == o.id; }
    bool operator<(const nid_t& o) { return id < o.id; }
    bool operator!=(const nid_t& o) { return !(*this == o); }
    bool operator>(const nid_t& o) { return !(*this == o) && !(*this < o); }
    bool operator<=(const nid_t& o) { return !(*this > o); }
    bool operator>=(const nid_t& o) { return !(*this < o); }
};

struct conn_base_t {
    nid_t src;
    nid_t dst;

    conn_base_t() {}
    conn_base_t(const nid_t& s, const nid_t& d) : src(s), dst(d) {}
};

struct neuron_base_t {
    neuron_base_t() {}
};

struct update_base_t {
    nid_t dst;

    update_base_t() {}
    update_base_t(const nid_t& d) : dst(d) {}
};


inline ap_int<16> lut_tanh_coef(ap_int<5> idx) {
    switch (idx) {
        case -16: return 163;
        case -15: return 207;
        case -14: return 262;
        case -13: return 330;
        case -12: return 414;
        case -11: return 516;
        case -10: return 638;
        case -9: return 781;
        case -8: return 944;
        case -7: return 1126;
        case -6: return 1320;
        case -5: return 1516;
        case -4: return 1701;
        case -3: return 1860;
        case -2: return 1978;
        case -1: return 2040;
        case 0: return 2040;
        case 1: return 1978;
        case 2: return 1860;
        case 3: return 1701;
        case 4: return 1516;
        case 5: return 1320;
        case 6: return 1126;
        case 7: return 944;
        case 8: return 781;
        case 9: return 638;
        case 10: return 516;
        case 11: return 414;
        case 12: return 330;
        case 13: return 262;
        case 14: return 207;
        case 15:
        default: return 163;
    }
}

inline ap_int<16> lut_tanh_incp(ap_int<5> idx) {
    switch (idx) {
        case -16: return -1649;
        case -15: return -1566;
        case -14: return -1470;
        case -13: return -1360;
        case -12: return -1234;
        case -11: return -1094;
        case -10: return -942;
        case -9: return -781;
        case -8: return -618;
        case -7: return -459;
        case -6: return -314;
        case -5: return -192;
        case -4: return -99;
        case -3: return -39;
        case -2: return -9;
        case -1: return -0;
        case 0: return 0;
        case 1: return 9;
        case 2: return 39;
        case 3: return 99;
        case 4: return 192;
        case 5: return 314;
        case 6: return 459;
        case 7: return 618;
        case 8: return 781;
        case 9: return 942;
        case 10: return 1094;
        case 11: return 1234;
        case 12: return 1360;
        case 13: return 1470;
        case 14: return 1566;
        case 15:
        default: return 1649;
    }
}

inline fixed_t lut_tanh(fixed_t x) {
    // no inline to avoid sharing registers with outside logic
#pragma HLS INLINE self off
    ap_int<5> rng;
    // sign bit
    rng[4] = x[W_DATA-1];
    // interesting bits
    rng(3,0) = x(W_DATA-IW_DATA+1-1, W_DATA-IW_DATA-3);

    // all higher bits should be equal to sign bit if no overflow
    bool overflow = x(W_DATA-1, W_DATA-IW_DATA+1) !=
        (rng[4] ? ap_uint<IW_DATA-1>(-1) : ap_uint<IW_DATA-1>(0));


    if (overflow) {
        return rng[4] ? fixed_t(-0.96403) : fixed_t(0.96403);
    } else {
        ap_fixed<16, 5> a, b;
        a.range() = lut_tanh_coef(rng);
        b.range() = lut_tanh_incp(rng);
        fixed_t coef = 0;
        coef.range(20, 5) = a.range();
        for (int ii = 21; ii < 48; ii++) {
#pragma HLS UNROLL
            coef[ii] = a[15];
        }
        // the output register breaks the critical path, which includes
        // the adder, and several MUX's after it.
        return pipe_add_output_reg(pipe_mult(coef, x), fixed_t(b));
    }
}

inline ap_int<16> lut_sigmoid_coef(ap_int<5> idx) {
    switch (idx) {
        case -16: return 41;
        case -15: return 52;
        case -14: return 66;
        case -13: return 83;
        case -12: return 104;
        case -11: return 129;
        case -10: return 159;
        case -9: return 195;
        case -8: return 236;
        case -7: return 281;
        case -6: return 330;
        case -5: return 379;
        case -4: return 425;
        case -3: return 465;
        case -2: return 494;
        case -1: return 510;
        case 0: return 510;
        case 1: return 494;
        case 2: return 465;
        case 3: return 425;
        case 4: return 379;
        case 5: return 330;
        case 6: return 281;
        case 7: return 236;
        case 8: return 195;
        case 9: return 159;
        case 10: return 129;
        case 11: return 104;
        case 12: return 83;
        case 13: return 66;
        case 14: return 52;
        case 15: return 41;
        default: return 0;
    }
}

inline ap_int<16> lut_sigmoid_incp(ap_int<5> idx) {
    switch (idx) {
        case -16: return 200;
        case -15: return 241;
        case -14: return 289;
        case -13: return 344;
        case -12: return 407;
        case -11: return 477;
        case -10: return 553;
        case -9: return 633;
        case -8: return 715;
        case -7: return 794;
        case -6: return 867;
        case -5: return 928;
        case -4: return 975;
        case -3: return 1005;
        case -2: return 1020;
        case -1: return 1024;
        case 0: return 1024;
        case 1: return 1028;
        case 2: return 1043;
        case 3: return 1073;
        case 4: return 1120;
        case 5: return 1181;
        case 6: return 1254;
        case 7: return 1333;
        case 8: return 1415;
        case 9: return 1495;
        case 10: return 1571;
        case 11: return 1641;
        case 12: return 1704;
        case 13: return 1759;
        case 14: return 1807;
        case 15: return 1848;
        default: return 0;
    }
}

inline fixed_t lut_sigmoid(fixed_t x) {
    // no inline to avoid sharing registers with outside logic
#pragma HLS INLINE self off
    ap_int<5> rng;
    // sign bit
    rng[4] = x[W_DATA-1];
    // interesting bits
    rng(3,0) = x(W_DATA-IW_DATA+2-1, W_DATA-IW_DATA-2);

    // all higher bits should be equal to sign bit if no overflow
    bool overflow = x(W_DATA-1, W_DATA-IW_DATA+1) !=
        (rng[4] ? ap_uint<IW_DATA-1>(-1) : ap_uint<IW_DATA-1>(0));


    if (overflow) {
        return rng[4] ? fixed_t(0.0179862) : fixed_t(0.9820138);
    } else {
        ap_fixed<16, 5> a, b;
        a.range() = lut_sigmoid_coef(rng);
        b.range() = lut_sigmoid_incp(rng);
        fixed_t coef = 0;
        coef.range(20, 5) = a.range();
        for (int ii = 21; ii < 48; ii++) {
#pragma HLS UNROLL
            coef[ii] = a[15];
        }
        // the output register breaks the critical path, which includes
        // the adder, and several MUX's after it.
        return pipe_add_output_reg(pipe_mult(coef, x), fixed_t(b));
    }
}

#endif // NEURAL_NET_H_
