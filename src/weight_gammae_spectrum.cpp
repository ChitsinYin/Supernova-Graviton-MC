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
constexpr double M_PLANCK = 2.435e18;        // GeV
constexpr double KM3_TO_INVGEV3 = 1.3005e56;
constexpr double A_MAP = 0.999999;
constexpr double PHOTON_TS_FACTOR = 10.0;
constexpr double PHI2_MASSLESS = 1.0/(8.0*PI);
}

double fFD(double E, double T, double mu)
{
    const double x = (E - mu)/T;
    if (x > 60.0) return std::exp(-x);
    if (x < -60.0) return 1.0;
    return 1.0/(std::exp(x) + 1.0);
}

double electronNormUR(double T, double mu)
{
    return (4.0*C::PI/3.0)*std::pow(mu, 3)
         * (1.0 + std::pow(C::PI*T/mu, 2));
}

double photonNormBoltzmann(double T)
{
    return 8.0*C::PI*std::pow(T, 3);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " input_pool.root output.csv\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];

    constexpr int NBIN = 120;
    const double omegaMin = 8.0e-5;
    const double omegaMax = 3.6;
    const double logMin = std::log(omegaMin);
    const double logMax = std::log(omegaMax);
    const double dLogOmega = (logMax - logMin)/NBIN;

    std::vector<double> centers(NBIN), sumZ(NBIN, 0.0), sumZ2(NBIN, 0.0);
    std::vector<Long64_t> counts(NBIN, 0);
    for (int i = 0; i < NBIN; ++i) {
        const double lo = std::exp(logMin + dLogOmega*i);
        const double hi = std::exp(logMin + dLogOmega*(i + 1));
        centers[i] = std::sqrt(lo*hi);
    }

    TFile fin(inputPath.c_str(), "READ");
    if (fin.IsZombie()) {
        std::cerr << "ERROR: cannot open " << inputPath << "\n";
        return 2;
    }

    auto* tree = dynamic_cast<TTree*>(fin.Get("events"));
    auto* pNProposal = dynamic_cast<TParameter<Long64_t>*>(fin.Get("N_PROPOSAL"));
    auto* pR = dynamic_cast<TParameter<double>*>(fin.Get("R_KM"));
    if (!tree || !pNProposal || !pR) {
        std::cerr << "ERROR: required tree/metadata missing\n";
        return 3;
    }

    const Long64_t nProposal = pNProposal->GetVal();
    const Long64_t nValid = tree->GetEntries();
    const double R_km = pR->GetVal();
    const double volumeWeight = (4.0*C::PI/3.0)*std::pow(R_km, 3)*C::KM3_TO_INVGEV3;

    float r_km = 0.0f, T = 0.0f, ne = 0.0f, ngamma = 0.0f;
    float mu = 0.0f, kappa = 0.0f, pmax_e = 0.0f;
    TLorentzVector *pe_in = nullptr, *pg_in = nullptr, *pe_out = nullptr, *h_out = nullptr;

    tree->SetBranchAddress("r_km", &r_km);
    tree->SetBranchAddress("T", &T);
    tree->SetBranchAddress("ne", &ne);
    tree->SetBranchAddress("ngamma", &ngamma);
    tree->SetBranchAddress("mu", &mu);
    tree->SetBranchAddress("kappa", &kappa);
    tree->SetBranchAddress("pmax_e", &pmax_e);
    tree->SetBranchAddress("pe_in", &pe_in);
    tree->SetBranchAddress("pg_in", &pg_in);
    tree->SetBranchAddress("pe_out", &pe_out);
    tree->SetBranchAddress("h_out", &h_out);

    Long64_t nFinite = 0;

    for (Long64_t i = 0; i < nValid; ++i) {
        tree->GetEntry(i);

        const double Tloc = T;
        const double muloc = mu;
        const double peMag = pe_in->P();
        const double pgMag = pg_in->P();
        if (!(Tloc > 0.0) || !(peMag > 0.0) || !(pgMag > 0.0)) continue;

        const double De = electronNormUR(Tloc, muloc);
        const double Re = 4.0*C::PI*static_cast<double>(pmax_e)*peMag*peMag
                        * fFD(peMag, Tloc, muloc)/De;

        const double Ts = C::PHOTON_TS_FACTOR*Tloc;
        const double gGamma = std::exp(-pgMag/Ts)/(C::A_MAP*Ts);
        const double fGamma = std::exp(-pgMag/Tloc);
        const double Rg = 4.0*C::PI*pgMag*pgMag*fGamma
                        /(photonNormBoltzmann(Tloc)*gGamma);

        const double s = (*pe_in + *pg_in).M2();
        const double t = (*pg_in - *h_out).M2();
        const double u = (*pe_in - *h_out).M2();
        const double denom = static_cast<double>(kappa)*kappa - t;
        if (!(s > 0.0) || !(denom > 0.0)) continue;

        const double M2 = (C::E_CHARGE*C::E_CHARGE)/(C::M_PLANCK*C::M_PLANCK)
                        * (s*s + u*u)/denom;
        const double blocking = 1.0 - fFD(pe_out->E(), Tloc, muloc);
        const double flux = 1.0/(4.0*peMag*pgMag);

        const double wRate = volumeWeight
                           * static_cast<double>(ne)*static_cast<double>(ngamma)
                           * Re*Rg*flux*C::PHI2_MASSLESS*blocking*M2;
        if (!std::isfinite(wRate) || wRate < 0.0) continue;
        ++nFinite;

        const double omega = h_out->E();
        if (!(omega >= omegaMin && omega < omegaMax)) continue;
        const int ibin = static_cast<int>((std::log(omega) - logMin)/dLogOmega);
        if (ibin < 0 || ibin >= NBIN) continue;

        const double z = omega*wRate;
        sumZ[ibin] += z;
        sumZ2[ibin] += z*z;
        counts[ibin] += 1;

        if ((i + 1) % 500000 == 0)
            std::cout << "\rWeighted " << (i + 1) << " / " << nValid << std::flush;
    }
    std::cout << "\n";

    std::ofstream out(outputPath);
    if (!out) {
        std::cerr << "ERROR: cannot create " << outputPath << "\n";
        return 4;
    }
    out << "omega_GeV,omega_dP_domega_GeV2,mc_error_GeV2,raw_event_count,effective_weighted_events\n";
    out << std::setprecision(12);

    for (int j = 0; j < NBIN; ++j) {
        const double spec = sumZ[j]/static_cast<double>(nProposal)/dLogOmega;
        double err = 0.0;
        if (nProposal > 1) {
            const double numerator = sumZ2[j] - sumZ[j]*sumZ[j]/static_cast<double>(nProposal);
            const double varMean = std::max(0.0, numerator)
                                 /(static_cast<double>(nProposal)*static_cast<double>(nProposal - 1));
            err = std::sqrt(varMean)/dLogOmega;
        }
        const double neff = (sumZ2[j] > 0.0) ? sumZ[j]*sumZ[j]/sumZ2[j] : 0.0;
        out << centers[j] << ',' << spec << ',' << err << ',' << counts[j] << ',' << neff << '\n';
    }

    std::cout << "N_PROPOSAL=" << nProposal << "\n"
              << "N_VALID_TREE=" << nValid << "\n"
              << "finite_weight_events=" << nFinite << "\n"
              << "output=" << outputPath << "\n";
    return 0;
}
