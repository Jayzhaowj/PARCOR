// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// cp_sd
Rcpp::List cp_sd(arma::cube phi, arma::mat SIGMA, arma::vec w);
RcppExport SEXP _PARCOR_cp_sd(SEXP phiSEXP, SEXP SIGMASEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type SIGMA(SIGMASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(cp_sd(phi, SIGMA, w));
    return rcpp_result_gen;
END_RCPP
}
// get_sd
arma::mat get_sd(Rcpp::List sd, int ts1, int ts2, int type);
RcppExport SEXP _PARCOR_get_sd(SEXP sdSEXP, SEXP ts1SEXP, SEXP ts2SEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type sd(sdSEXP);
    Rcpp::traits::input_parameter< int >::type ts1(ts1SEXP);
    Rcpp::traits::input_parameter< int >::type ts2(ts2SEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(get_sd(sd, ts1, ts2, type));
    return rcpp_result_gen;
END_RCPP
}
// cp_sd_uni
arma::cube cp_sd_uni(arma::cube phi, arma::vec sigma2, arma::vec w);
RcppExport SEXP _PARCOR_cp_sd_uni(SEXP phiSEXP, SEXP sigma2SEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sigma2(sigma2SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(cp_sd_uni(phi, sigma2, w));
    return rcpp_result_gen;
END_RCPP
}
// compute_DIC_TVVAR
Rcpp::List compute_DIC_TVVAR(Rcpp::List temp_filter, int sample_size, arma::cube St, int P_max);
RcppExport SEXP _PARCOR_compute_DIC_TVVAR(SEXP temp_filterSEXP, SEXP sample_sizeSEXP, SEXP StSEXP, SEXP P_maxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type temp_filter(temp_filterSEXP);
    Rcpp::traits::input_parameter< int >::type sample_size(sample_sizeSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type St(StSEXP);
    Rcpp::traits::input_parameter< int >::type P_max(P_maxSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_DIC_TVVAR(temp_filter, sample_size, St, P_max));
    return rcpp_result_gen;
END_RCPP
}
// compute_spec
Rcpp::List compute_spec(arma::cube phi, arma::mat SIGMA, arma::vec w, int P_max, int ch1, int ch2, bool time_depend);
RcppExport SEXP _PARCOR_compute_spec(SEXP phiSEXP, SEXP SIGMASEXP, SEXP wSEXP, SEXP P_maxSEXP, SEXP ch1SEXP, SEXP ch2SEXP, SEXP time_dependSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type SIGMA(SIGMASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type P_max(P_maxSEXP);
    Rcpp::traits::input_parameter< int >::type ch1(ch1SEXP);
    Rcpp::traits::input_parameter< int >::type ch2(ch2SEXP);
    Rcpp::traits::input_parameter< bool >::type time_depend(time_dependSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_spec(phi, SIGMA, w, P_max, ch1, ch2, time_depend));
    return rcpp_result_gen;
END_RCPP
}
// forward_filter_backward_smooth
Rcpp::List forward_filter_backward_smooth(arma::mat yt, arma::mat F1, arma::mat F2, double n_0, double S_0, int n_t, int n_I, int m, int type, int P, double delta1, double delta2, int sample_size);
RcppExport SEXP _PARCOR_forward_filter_backward_smooth(SEXP ytSEXP, SEXP F1SEXP, SEXP F2SEXP, SEXP n_0SEXP, SEXP S_0SEXP, SEXP n_tSEXP, SEXP n_ISEXP, SEXP mSEXP, SEXP typeSEXP, SEXP PSEXP, SEXP delta1SEXP, SEXP delta2SEXP, SEXP sample_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type yt(ytSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type F1(F1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type F2(F2SEXP);
    Rcpp::traits::input_parameter< double >::type n_0(n_0SEXP);
    Rcpp::traits::input_parameter< double >::type S_0(S_0SEXP);
    Rcpp::traits::input_parameter< int >::type n_t(n_tSEXP);
    Rcpp::traits::input_parameter< int >::type n_I(n_ISEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    Rcpp::traits::input_parameter< int >::type P(PSEXP);
    Rcpp::traits::input_parameter< double >::type delta1(delta1SEXP);
    Rcpp::traits::input_parameter< double >::type delta2(delta2SEXP);
    Rcpp::traits::input_parameter< int >::type sample_size(sample_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(forward_filter_backward_smooth(yt, F1, F2, n_0, S_0, n_t, n_I, m, type, P, delta1, delta2, sample_size));
    return rcpp_result_gen;
END_RCPP
}
// ffbs_DIC
Rcpp::List ffbs_DIC(arma::mat yt, arma::mat F1, arma::mat F2, double n_0, double S_0, int n_t, int n_I, int m, int type, int P, arma::mat delta, bool DIC, int sample_size, int chains, bool uncertainty);
RcppExport SEXP _PARCOR_ffbs_DIC(SEXP ytSEXP, SEXP F1SEXP, SEXP F2SEXP, SEXP n_0SEXP, SEXP S_0SEXP, SEXP n_tSEXP, SEXP n_ISEXP, SEXP mSEXP, SEXP typeSEXP, SEXP PSEXP, SEXP deltaSEXP, SEXP DICSEXP, SEXP sample_sizeSEXP, SEXP chainsSEXP, SEXP uncertaintySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type yt(ytSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type F1(F1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type F2(F2SEXP);
    Rcpp::traits::input_parameter< double >::type n_0(n_0SEXP);
    Rcpp::traits::input_parameter< double >::type S_0(S_0SEXP);
    Rcpp::traits::input_parameter< int >::type n_t(n_tSEXP);
    Rcpp::traits::input_parameter< int >::type n_I(n_ISEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    Rcpp::traits::input_parameter< int >::type P(PSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< bool >::type DIC(DICSEXP);
    Rcpp::traits::input_parameter< int >::type sample_size(sample_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type chains(chainsSEXP);
    Rcpp::traits::input_parameter< bool >::type uncertainty(uncertaintySEXP);
    rcpp_result_gen = Rcpp::wrap(ffbs_DIC(yt, F1, F2, n_0, S_0, n_t, n_I, m, type, P, delta, DIC, sample_size, chains, uncertainty));
    return rcpp_result_gen;
END_RCPP
}
// filter_smooth_TVVAR
Rcpp::List filter_smooth_TVVAR(arma::mat F1, arma::mat G, arma::mat mk_0, arma::mat Ck_0, double n_0, arma::mat S_0, int m, arma::mat delta, int pp);
RcppExport SEXP _PARCOR_filter_smooth_TVVAR(SEXP F1SEXP, SEXP GSEXP, SEXP mk_0SEXP, SEXP Ck_0SEXP, SEXP n_0SEXP, SEXP S_0SEXP, SEXP mSEXP, SEXP deltaSEXP, SEXP ppSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type F1(F1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type G(GSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mk_0(mk_0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Ck_0(Ck_0SEXP);
    Rcpp::traits::input_parameter< double >::type n_0(n_0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type S_0(S_0SEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< int >::type pp(ppSEXP);
    rcpp_result_gen = Rcpp::wrap(filter_smooth_TVVAR(F1, G, mk_0, Ck_0, n_0, S_0, m, delta, pp));
    return rcpp_result_gen;
END_RCPP
}
// filter
Rcpp::List filter(arma::mat F1_fwd, arma::mat F1_bwd, arma::mat S_0, int m, arma::rowvec delta, int type_num, int P, int n_t, int n_I, int n_I2);
RcppExport SEXP _PARCOR_filter(SEXP F1_fwdSEXP, SEXP F1_bwdSEXP, SEXP S_0SEXP, SEXP mSEXP, SEXP deltaSEXP, SEXP type_numSEXP, SEXP PSEXP, SEXP n_tSEXP, SEXP n_ISEXP, SEXP n_I2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type F1_fwd(F1_fwdSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type F1_bwd(F1_bwdSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type S_0(S_0SEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::rowvec >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< int >::type type_num(type_numSEXP);
    Rcpp::traits::input_parameter< int >::type P(PSEXP);
    Rcpp::traits::input_parameter< int >::type n_t(n_tSEXP);
    Rcpp::traits::input_parameter< int >::type n_I(n_ISEXP);
    Rcpp::traits::input_parameter< int >::type n_I2(n_I2SEXP);
    rcpp_result_gen = Rcpp::wrap(filter(F1_fwd, F1_bwd, S_0, m, delta, type_num, P, n_t, n_I, n_I2));
    return rcpp_result_gen;
END_RCPP
}
// filter_smooth
Rcpp::List filter_smooth(arma::mat F1_fwd, arma::mat F1_bwd, arma::mat S_0, int m, arma::mat delta, int type_num, int P, bool DIC, int sample_size, int chains, bool uncertainty);
RcppExport SEXP _PARCOR_filter_smooth(SEXP F1_fwdSEXP, SEXP F1_bwdSEXP, SEXP S_0SEXP, SEXP mSEXP, SEXP deltaSEXP, SEXP type_numSEXP, SEXP PSEXP, SEXP DICSEXP, SEXP sample_sizeSEXP, SEXP chainsSEXP, SEXP uncertaintySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type F1_fwd(F1_fwdSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type F1_bwd(F1_bwdSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type S_0(S_0SEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< int >::type type_num(type_numSEXP);
    Rcpp::traits::input_parameter< int >::type P(PSEXP);
    Rcpp::traits::input_parameter< bool >::type DIC(DICSEXP);
    Rcpp::traits::input_parameter< int >::type sample_size(sample_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type chains(chainsSEXP);
    Rcpp::traits::input_parameter< bool >::type uncertainty(uncertaintySEXP);
    rcpp_result_gen = Rcpp::wrap(filter_smooth(F1_fwd, F1_bwd, S_0, m, delta, type_num, P, DIC, sample_size, chains, uncertainty));
    return rcpp_result_gen;
END_RCPP
}
// run_whittle
Rcpp::List run_whittle(arma::cube phi_fwd, arma::cube phi_bwd, int n_I);
RcppExport SEXP _PARCOR_run_whittle(SEXP phi_fwdSEXP, SEXP phi_bwdSEXP, SEXP n_ISEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type phi_fwd(phi_fwdSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type phi_bwd(phi_bwdSEXP);
    Rcpp::traits::input_parameter< int >::type n_I(n_ISEXP);
    rcpp_result_gen = Rcpp::wrap(run_whittle(phi_fwd, phi_bwd, n_I));
    return rcpp_result_gen;
END_RCPP
}
// sample_tvar_coef
Rcpp::List sample_tvar_coef(arma::cube phi_fwd, arma::cube phi_bwd, Rcpp::List Cnt_fwd, Rcpp::List Cnt_bwd, int n_I, int P_opt, int P_max, int h);
RcppExport SEXP _PARCOR_sample_tvar_coef(SEXP phi_fwdSEXP, SEXP phi_bwdSEXP, SEXP Cnt_fwdSEXP, SEXP Cnt_bwdSEXP, SEXP n_ISEXP, SEXP P_optSEXP, SEXP P_maxSEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type phi_fwd(phi_fwdSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type phi_bwd(phi_bwdSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type Cnt_fwd(Cnt_fwdSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type Cnt_bwd(Cnt_bwdSEXP);
    Rcpp::traits::input_parameter< int >::type n_I(n_ISEXP);
    Rcpp::traits::input_parameter< int >::type P_opt(P_optSEXP);
    Rcpp::traits::input_parameter< int >::type P_max(P_maxSEXP);
    Rcpp::traits::input_parameter< int >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_tvar_coef(phi_fwd, phi_bwd, Cnt_fwd, Cnt_bwd, n_I, P_opt, P_max, h));
    return rcpp_result_gen;
END_RCPP
}
// run_dl
Rcpp::List run_dl(arma::cube phi_fwd, arma::cube phi_bwd);
RcppExport SEXP _PARCOR_run_dl(SEXP phi_fwdSEXP, SEXP phi_bwdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type phi_fwd(phi_fwdSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type phi_bwd(phi_bwdSEXP);
    rcpp_result_gen = Rcpp::wrap(run_dl(phi_fwd, phi_bwd));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_PARCOR_cp_sd", (DL_FUNC) &_PARCOR_cp_sd, 3},
    {"_PARCOR_get_sd", (DL_FUNC) &_PARCOR_get_sd, 4},
    {"_PARCOR_cp_sd_uni", (DL_FUNC) &_PARCOR_cp_sd_uni, 3},
    {"_PARCOR_compute_DIC_TVVAR", (DL_FUNC) &_PARCOR_compute_DIC_TVVAR, 4},
    {"_PARCOR_compute_spec", (DL_FUNC) &_PARCOR_compute_spec, 7},
    {"_PARCOR_forward_filter_backward_smooth", (DL_FUNC) &_PARCOR_forward_filter_backward_smooth, 13},
    {"_PARCOR_ffbs_DIC", (DL_FUNC) &_PARCOR_ffbs_DIC, 15},
    {"_PARCOR_filter_smooth_TVVAR", (DL_FUNC) &_PARCOR_filter_smooth_TVVAR, 9},
    {"_PARCOR_filter", (DL_FUNC) &_PARCOR_filter, 10},
    {"_PARCOR_filter_smooth", (DL_FUNC) &_PARCOR_filter_smooth, 11},
    {"_PARCOR_run_whittle", (DL_FUNC) &_PARCOR_run_whittle, 3},
    {"_PARCOR_sample_tvar_coef", (DL_FUNC) &_PARCOR_sample_tvar_coef, 8},
    {"_PARCOR_run_dl", (DL_FUNC) &_PARCOR_run_dl, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_PARCOR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
