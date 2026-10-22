#include "TFile.h"
#include "TLorentzVector.h"
#include "TParameter.h"
#include "TTree.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace C {
constexpr double PI = 3.14159265358979323846;
constexpr double E_CHARGE = 0.303;
constexpr double M_P = 2.435e18;  // reduced Planck mass, GeV
constexpr double KM3_TO_INVGEV3 = 1.3005e56;

// Exact integrated two-body Lorentz-invariant phase space for
// two massless final-state particles.
constexpr double PHI2_MASSLESS = 1.0/(8.0*PI);
}

double fermiMinus(double E, double T, double mu)
{
    const double x = (E - mu)/T;
    if (x > 60.0) return std::exp(-x);
    if (x < -60.0) return 1.0;
    return 1.0/(std::exp(x) + 1.0);
}

double fermiPlus(double E, double T, double mu)
{
    const double x = (E + mu)/T;
    if (x > 60.0) return std::exp(-x);
    if (x < -60.0) return 1.0;
    return 1.0/(std::exp(x) + 1.0);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " pool.root output.csv\n";
        return 1;
    }

    const std::string poolPath = argv[1];
    const std::string csvPath = argv[2];

    TFile fin(poolPath.c_str(), "READ");
    if (fin.IsZombie()) {
        std::cerr << "ERROR: cannot open " << poolPath << "\n";
        return 2;
    }

    auto* tree = dynamic_cast<TTree*>(fin.Get("events"));
    auto* pNProposal =
        dynamic_cast<TParameter<Long64_t>*>(fin.Get("N_PROPOSAL"));
    auto* pR =
        dynamic_cast<TParameter<double>*>(fin.Get("R_KM"));

    if (!tree || !pNProposal || !pR) {
        std::cerr << "ERROR: required tree/metadata missing\n";
        return 3;
    }

    const Long64_t nProposal = pNProposal->GetVal();
    const Long64_t nValid = tree->GetEntries();
    const double R_km = pR->GetVal();

    const double volumeWeight =
        (4.0*C::PI/3.0)
        *std::pow(R_km, 3)
        *C::KM3_TO_INVGEV3;

    // Same logarithmic range used by the other ROOT/PHSP channels.
    constexpr int NBIN = 180;
    const double omegaMin = 8.0e-5;
    const double omegaMax = 3.3;
    const double logMin = std::log(omegaMin);
    const double logMax = std::log(omegaMax);
    const double dLog = (logMax - logMin)/NBIN;

    std::vector<double> centers(NBIN, 0.0);
    for (int j = 0; j < NBIN; ++j) {
        const double lo = std::exp(logMin + j*dLog);
        const double hi = std::exp(logMin + (j + 1)*dLog);
        centers[j] = std::sqrt(lo*hi);
    }

    std::vector<double> sumZ(NBIN, 0.0);
    std::vector<double> sumZ2(NBIN, 0.0);
    std::vector<long long> counts(NBIN, 0);

    double r_km = 0.0;
    double T = 0.0;
    double np = 0.0;
    double nplus = 0.0;
    double nminus_fd = 0.0;
    double mu = 0.0;

    TLorentzVector* em_in = nullptr;
    TLorentzVector* ep_in = nullptr;
    TLorentzVector* h_out = nullptr;
    TLorentzVector* gamma_out = nullptr;

    double gminus_radial = 0.0;
    double gplus_radial = 0.0;
    int em_component = -1;

    tree->SetBranchAddress("r_km", &r_km);
    tree->SetBranchAddress("T", &T);
    tree->SetBranchAddress("np", &np);
    tree->SetBranchAddress("nplus", &nplus);
    tree->SetBranchAddress("nminus_fd", &nminus_fd);
    tree->SetBranchAddress("mu", &mu);

    tree->SetBranchAddress("em_in", &em_in);
    tree->SetBranchAddress("ep_in", &ep_in);
    tree->SetBranchAddress("h_out", &h_out);
    tree->SetBranchAddress("gamma_out", &gamma_out);

    tree->SetBranchAddress("gminus_radial", &gminus_radial);
    tree->SetBranchAddress("gplus_radial", &gplus_radial);
    tree->SetBranchAddress("em_component", &em_component);

    Long64_t nFiniteWeight = 0;
    Long64_t nSpectrumRange = 0;
    Long64_t nRejected = 0;
    long double totalRateSum = 0.0L;

    for (Long64_t i = 0; i < nValid; ++i) {
        tree->GetEntry(i);

        if (!em_in || !ep_in || !h_out || !gamma_out) {
            ++nRejected;
            continue;
        }

        if (!(T > 0.0) || !(mu > 0.0)
            || !(np > 0.0) || !(nplus > 0.0) || !(nminus_fd > 0.0)
            || !(gminus_radial > 0.0) || !(gplus_radial > 0.0)) {
            ++nRejected;
            continue;
        }

        // Exact FD occupation factors in the plasma rest frame.
        const double fminus = fermiMinus(em_in->E(), T, mu);
        const double fplus = fermiPlus(ep_in->E(), T, mu);

        // D = integral d^3p f(p).  With spin degeneracy g=2,
        // n = [2/(2pi)^3] D, hence D = 4 pi^3 n.
        // The finite-T profile satisfies n_- = n_p + n_+.
        const double Dminus = 4.0*std::pow(C::PI, 3)*nminus_fd;
        const double Dplus = 4.0*std::pow(C::PI, 3)*nplus;

        const double pminus = em_in->P();
        const double pplus = ep_in->P();

        const double Rminus =
            4.0*C::PI*pminus*pminus*fminus
            /(Dminus*gminus_radial);

        const double Rplus =
            4.0*C::PI*pplus*pplus*fplus
            /(Dplus*gplus_radial);

        if (!std::isfinite(Rminus) || !std::isfinite(Rplus)
            || Rminus < 0.0 || Rplus < 0.0) {
            ++nRejected;
            continue;
        }

        // Mandelstam invariants from exact event four-vectors.
        // The analytical reference uses
        // |M|^2 = e^2/M_P^2 * (t^2 + u^2)/s.
        const double s = (*em_in + *ep_in).M2();
        const double t = (*em_in - *h_out).M2();
        const double u = (*em_in - *gamma_out).M2();

        if (!(s > 0.0)) {
            ++nRejected;
            continue;
        }

        const double M2 =
            (C::E_CHARGE*C::E_CHARGE)
            /(C::M_P*C::M_P)
            *(t*t + u*u)/s;

        if (!std::isfinite(M2) || M2 < 0.0) {
            ++nRejected;
            continue;
        }

        const double fluxMeasure =
            1.0/(4.0*em_in->E()*ep_in->E());

        // The source note adopts n_- ~= n_p for the density prefactor,
        // while n_+ is evaluated with the refined finite-T chemical potential.
        // The exact FD n_- = n_p + n_+ is retained only in Dminus above,
        // so the initial electron momentum distribution is correctly normalized.
        const double wRate =
            volumeWeight
            *np*nplus
            *Rminus*Rplus
            *fluxMeasure
            *C::PHI2_MASSLESS
            *M2;

        if (!std::isfinite(wRate) || wRate < 0.0) {
            ++nRejected;
            continue;
        }

        ++nFiniteWeight;
        totalRateSum += static_cast<long double>(wRate);

        const double omega = h_out->E();
        if (!(omega >= omegaMin && omega < omegaMax)) {
            continue;
        }

        const int ibin = static_cast<int>(
            (std::log(omega) - logMin)/dLog
        );

        if (ibin < 0 || ibin >= NBIN) continue;

        const double z = omega*wRate;
        sumZ[ibin] += z;
        sumZ2[ibin] += z*z;
        counts[ibin] += 1;
        ++nSpectrumRange;

        if ((i + 1) % 500000 == 0) {
            std::cout
                << "\rWeighted events: "
                << (i + 1) << " / " << nValid
                << std::flush;
        }
    }

    std::cout << "\n";

    std::vector<double> spectrum(NBIN, 0.0);
    std::vector<double> specErr(NBIN, 0.0);
    std::vector<double> neff(NBIN, 0.0);

    for (int j = 0; j < NBIN; ++j) {
        spectrum[j] =
            sumZ[j]/static_cast<double>(nProposal)/dLog;

        if (nProposal > 1) {
            const double numerator =
                sumZ2[j]
                -sumZ[j]*sumZ[j]/static_cast<double>(nProposal);

            const double varMean =
                std::max(0.0, numerator)
                /(
                    static_cast<double>(nProposal)
                    *static_cast<double>(nProposal - 1)
                 );

            specErr[j] = std::sqrt(varMean)/dLog;
        }

        if (sumZ2[j] > 0.0) {
            neff[j] = sumZ[j]*sumZ[j]/sumZ2[j];
        }
    }

    std::ofstream out(csvPath);
    if (!out) {
        std::cerr << "ERROR: cannot create " << csvPath << "\n";
        return 4;
    }

    out
        << "omega_GeV,"
        << "omega_dP_domega_GeV2,"
        << "mc_error_GeV2,"
        << "raw_event_count,"
        << "effective_weighted_events\n";

    out << std::setprecision(12);

    for (int j = 0; j < NBIN; ++j) {
        out
            << centers[j] << ","
            << spectrum[j] << ","
            << specErr[j] << ","
            << counts[j] << ","
            << neff[j] << "\n";
    }

    out.close();

    std::cout
        << "=== e- e+ ANNIHILATION WEIGHT SUMMARY ===\n"
        << "N_PROPOSAL=" << nProposal << "\n"
        << "N_VALID_TREE=" << nValid << "\n"
        << "finite_weight_events=" << nFiniteWeight << "\n"
        << "spectrum_range_events=" << nSpectrumRange << "\n"
        << "rejected_events=" << nRejected << "\n"
        << "mean_total_rate_estimator_GeV="
        << static_cast<double>(totalRateSum/static_cast<long double>(nProposal))
        << "\n"
        << "output=" << csvPath << "\n";

    return 0;
}
