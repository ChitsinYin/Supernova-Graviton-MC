// weight_gammap_spectrum.cpp
//
// Physics reweighting and spectrum integration for
//     gamma + p -> h + p
//
// This version includes the exact importance correction for the broadened
// photon thermal-tail proposal used by generate_gammap_phsp.cpp.
//
// Input:
//   ROOT PHSP pool produced by generate_gammap_phsp.cpp
//
// Output:
//   CSV with
//     omega_GeV,
//     omega_dP_domega_GeV2,
//     mc_error_GeV2,
//     raw_event_count,
//     effective_weighted_events
//
// The exact massive-proton squared amplitude is evaluated in the incoming
// proton rest frame using the Mathematica/FeynGrav result:
//
// |M|^2_vac =
//   ((c+1) e^2 kappa_g^2 m_p
//    [-(c-1)(c^2+1) E m_p
//      +(c^2+1)m_p^2
//      +(c-1)^2 E^2])
//   /(4(c-1)[(c-1)E-m_p])).
//
// Here E is the incoming photon energy and c=cos(theta) between the incoming
// photon and outgoing graviton in the proton rest frame.
//
// Plasma screening is applied as
//
//   S = (-t)/(kappa_D^2 - t),
//
// so that in the strict NR limit
//
//   -t -> 4 omega^2 sin^2(theta/2),
//
// reproducing the standard screened proton-photoproduction factor.
//
// To avoid an artificial inf*0 cancellation in the forward direction, the
// code evaluates the algebraically combined screened expression directly.

#include "TFile.h"
#include "TNamed.h"
#include "TParameter.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TVector3.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace C {

constexpr double PI =
    3.14159265358979323846;

constexpr double ZETA3 =
    1.2020569031595942854;

constexpr double ALPHA =
    0.00729735;

constexpr double E2 =
    4.0 * PI * ALPHA;

constexpr double M_PROTON =
    0.9395;  // GeV

// Reduced Planck mass.
constexpr double M_PLANCK =
    2.435e18;  // GeV

// FeynGrav convention:
//   kappa_g^2 = 32*pi*G = 4/Mbar_P^2.
constexpr double KAPPA_GRAV_SQ =
    4.0
    / (M_PLANCK * M_PLANCK);

constexpr double KM3_TO_INVGEV3 =
    1.3005e56;

}  // namespace C

struct Spectrum {
    std::vector<double> sum;
    std::vector<double> sum2;
    std::vector<long long> count;

    explicit Spectrum(int n)
        : sum(n, 0.0),
          sum2(n, 0.0),
          count(n, 0)
    {}
};

TLorentzVector unpack4(
    const float a[4]
)
{
    TLorentzVector v;

    v.SetPxPyPzE(
        a[0],
        a[1],
        a[2],
        a[3]
    );

    return v;
}

double cos3(
    const TLorentzVector& a,
    const TLorentzVector& b
)
{
    const double pa = a.P();
    const double pb = b.P();

    if (!(pa > 0.0)
        || !(pb > 0.0)) {
        return 1.0;
    }

    double c =
        a.Vect().Dot(
            b.Vect()
        )
        / (pa * pb);

    if (c > 1.0) {
        c = 1.0;
    }

    if (c < -1.0) {
        c = -1.0;
    }

    return c;
}

// Same density convention as the existing gamma-e photoproduction code:
//
// thermal SHAPE: Maxwell-Boltzmann exp(-p/T),
// photon number density: equilibrium Bose-Einstein value.
double photonNumberDensity(double T)
{
    return
        2.0
        * C::ZETA3
        / (C::PI * C::PI)
        * T * T * T;
}


// Normalized physical MB photon radial density used by the calculation:
//
// q_phys(p;T) = p^2 exp(-p/T)/(2 T^3).
double physicalPhotonRadialPdf(
    double p,
    double T
)
{
    if (!(p > 0.0)
        || !(T > 0.0)) {
        return 0.0;
    }

    return
        p * p
        * std::exp(-p / T)
        /
        (
            2.0
            * T
            * T
            * T
        );
}

// Exact screened massive-proton matrix element.
//
// Inputs:
//   Ein       incoming photon energy in the initial-proton rest frame
//   omegaOut  outgoing graviton energy in the same frame
//   c         cos(theta_gamma,h) in that frame
//   kappaD    Debye screening scale
//
// Define a = 1-c >=0 and
//
//   Q^2 = -t = 2 Ein omegaOut a.
//
// Vacuum M^2 has a 1/a forward pole.  Multiplying by
//
//   Q^2/(kappaD^2 + Q^2)
//
// cancels that explicit a.  The expression below is the already-combined
// finite form and is numerically stable as c -> 1.
double screenedGammaPM2(
    double Ein,
    double omegaOut,
    double c,
    double kappaD
)
{
    if (!(Ein > 0.0)
        || !(omegaOut > 0.0)
        || !(kappaD >= 0.0)) {
        return 0.0;
    }

    c =
        std::max(
            -1.0,
            std::min(
                1.0,
                c
            )
        );

    const double a =
        1.0 - c;

    const double m =
        C::M_PROTON;

    const double c2 =
        c * c;

    // Positive form of the exact Mathematica result after using
    // the exact 2->2 recoil relation.
    const double B =
        (1.0 + c2)
        * m * m
        +
        a
        * (1.0 + c2)
        * Ein
        * m
        +
        a * a
        * Ein * Ein;

    const double q2 =
        2.0
        * Ein
        * omegaOut
        * a;  // = -t

    const double screenDen =
        kappaD * kappaD
        + q2;

    const double recoilDen =
        m
        + a * Ein;

    if (!(screenDen > 0.0)
        || !(recoilDen > 0.0)) {
        return 0.0;
    }

    // Algebraically:
    //
    // M2_screened
    // = M2_vac * q2/(kappaD^2+q2)
    //
    // with the forward a factor cancelled analytically.
    const double M2 =
        C::E2
        * C::KAPPA_GRAV_SQ
        * m
        * (1.0 + c)
        * B
        * Ein
        * omegaOut
        /
        (
            2.0
            * recoilDen
            * screenDen
        );

    if (!std::isfinite(M2)) {
        return 0.0;
    }

    return M2;
}

void fillSpectrum(
    Spectrum& s,
    double omega,
    double wRate,
    double logMin,
    double dLog
)
{
    if (!(omega > 0.0)
        || !std::isfinite(wRate)) {
        return;
    }

    const int n =
        static_cast<int>(
            s.sum.size()
        );

    const int ibin =
        static_cast<int>(
            (
                std::log(omega)
                - logMin
            )
            / dLog
        );

    if (ibin < 0
        || ibin >= n) {
        return;
    }

    // For logarithmic bins:
    //
    // dP/dln(omega) = omega dP/domega.
    //
    // wRate estimates the event contribution to the total production
    // rate Gamma, so one extra factor of omega converts rate -> power.
    const double z =
        omega * wRate;

    s.sum[ibin] += z;
    s.sum2[ibin] += z * z;
    s.count[ibin] += 1;
}

void finalizeSpectrum(
    const Spectrum& in,
    long long nProposal,
    double dLog,
    std::vector<double>& spec,
    std::vector<double>& err,
    std::vector<double>& neff
)
{
    const int n =
        static_cast<int>(
            in.sum.size()
        );

    spec.assign(
        n,
        0.0
    );

    err.assign(
        n,
        0.0
    );

    neff.assign(
        n,
        0.0
    );

    for (
        int j = 0;
        j < n;
        ++j
    ) {

        spec[j] =
            in.sum[j]
            /
            static_cast<double>(
                nProposal
            )
            /
            dLog;

        if (nProposal > 1) {

            const double numerator =
                in.sum2[j]
                -
                in.sum[j]
                * in.sum[j]
                /
                static_cast<double>(
                    nProposal
                );

            const double varianceMean =
                std::max(
                    0.0,
                    numerator
                )
                /
                (
                    static_cast<double>(
                        nProposal
                    )
                    *
                    static_cast<double>(
                        nProposal - 1
                    )
                );

            err[j] =
                std::sqrt(
                    varianceMean
                )
                /
                dLog;
        }

        if (in.sum2[j] > 0.0) {

            neff[j] =
                in.sum[j]
                * in.sum[j]
                /
                in.sum2[j];
        }
    }
}

int main(
    int argc,
    char** argv
)
{
    if (argc < 3) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " pool.root output.csv\n";

        return 1;
    }

    const std::string poolPath =
        argv[1];

    const std::string csvOut =
        argv[2];

    const std::filesystem::path csvPath(csvOut);

    if (csvPath.has_parent_path()) {
        std::filesystem::create_directories(
            csvPath.parent_path()
        );
    }

    TFile fin(
        poolPath.c_str(),
        "READ"
    );

    if (fin.IsZombie()) {

        std::cerr
            << "ERROR: cannot open "
            << poolPath
            << "\n";

        return 2;
    }

    auto* tree =
        dynamic_cast<TTree*>(
            fin.Get("events")
        );

    auto* pNProposal =
        dynamic_cast<TParameter<Long64_t>*>(
            fin.Get("N_PROPOSAL")
        );

    auto* pR =
        dynamic_cast<TParameter<double>*>(
            fin.Get("R_KM")
        );

    auto* pChannel =
        dynamic_cast<TNamed*>(
            fin.Get("CHANNEL")
        );

    if (!tree
        || !pNProposal
        || !pR
        || !pChannel) {

        std::cerr
            << "ERROR: required tree/metadata missing\n";

        return 3;
    }

    if (
        std::string(
            pChannel->GetTitle()
        )
        != "gammap"
    ) {

        std::cerr
            << "ERROR: input ROOT file is not a gammap pool\n";

        return 4;
    }

    const Long64_t nProposal =
        pNProposal->GetVal();

    const Long64_t nValid =
        tree->GetEntries();

    const double R_km =
        pR->GetVal();

    const double volumeWeight =
        (4.0 * C::PI / 3.0)
        * std::pow(
            R_km,
            3
        )
        * C::KM3_TO_INVGEV3;

    // Same logarithmic spectrum range used by the other public channels.
    constexpr int NBIN = 180;

    const double omegaMin =
        8.0e-5;

    const double omegaMax =
        3.3;

    const double logMin =
        std::log(
            omegaMin
        );

    const double logMax =
        std::log(
            omegaMax
        );

    const double dLog =
        (logMax - logMin)
        / NBIN;

    std::vector<double> edges(
        NBIN + 1
    );

    std::vector<double> centers(
        NBIN
    );

    for (
        int i = 0;
        i <= NBIN;
        ++i
    ) {

        edges[i] =
            std::exp(
                logMin
                +
                (logMax - logMin)
                * static_cast<double>(i)
                / NBIN
            );
    }

    for (
        int i = 0;
        i < NBIN;
        ++i
    ) {

        centers[i] =
            std::sqrt(
                edges[i]
                * edges[i + 1]
            );
    }

    Spectrum spectrum(NBIN);

    float r_km = 0.0f;
    float T = 0.0f;
    float np = 0.0f;
    float kappa = 0.0f;

    float pGammaA[4] = {0, 0, 0, 0};
    float pProtonA[4] = {0, 0, 0, 0};
    float khA[4] = {0, 0, 0, 0};
    float kProtonA[4] = {0, 0, 0, 0};

    double g_gamma_radial = 0.0;
    int gamma_component = -1;

    double w_tgen = 0.0;
    double w_phi2 = 0.0;

    tree->SetBranchAddress(
        "r_km",
        &r_km
    );

    tree->SetBranchAddress(
        "T",
        &T
    );

    tree->SetBranchAddress(
        "np",
        &np
    );

    tree->SetBranchAddress(
        "kappa",
        &kappa
    );

    tree->SetBranchAddress(
        "pGamma",
        pGammaA
    );

    tree->SetBranchAddress(
        "pProton",
        pProtonA
    );

    tree->SetBranchAddress(
        "kh",
        khA
    );

    tree->SetBranchAddress(
        "kProton",
        kProtonA
    );

    tree->SetBranchAddress(
        "g_gamma_radial",
        &g_gamma_radial
    );

    tree->SetBranchAddress(
        "gamma_component",
        &gamma_component
    );

    tree->SetBranchAddress(
        "w_tgen",
        &w_tgen
    );

    tree->SetBranchAddress(
        "w_phi2",
        &w_phi2
    );

    long long nFinite = 0;
    long long nNegative = 0;
    long long nBroad = 0;

    long double signedSum = 0.0L;
    long double absSum = 0.0L;
    long double broadSignedSum = 0.0L;

    for (
        Long64_t i = 0;
        i < nValid;
        ++i
    ) {

        tree->GetEntry(i);

        const double Tloc =
            static_cast<double>(T);

        const double nploc =
            static_cast<double>(np);

        const double kappaloc =
            static_cast<double>(kappa);

        if (!(Tloc > 0.0)
            || !(nploc > 0.0)
            || !(g_gamma_radial > 0.0)
            || !(w_phi2 > 0.0)) {
            continue;
        }

        const TLorentzVector pGamma =
            unpack4(pGammaA);

        const TLorentzVector pProton =
            unpack4(pProtonA);

        const TLorentzVector kh =
            unpack4(khA);

        const TLorentzVector kProton =
            unpack4(kProtonA);

        const double omega =
            kh.E();  // plasma-frame graviton energy

        if (!(omega > 0.0)
            || !(pGamma.E() > 0.0)
            || !(pProton.E() > 0.0)) {
            continue;
        }

        const double nGamma =
            photonNumberDensity(
                Tloc
            );

        // The proton proposal still exactly matches the normalized NR
        // Maxwell distribution, so R_p = 1.
        //
        // The photon now uses a nominal+broad mixture proposal, hence the
        // exact importance correction
        //
        //   R_gamma = q_phys(p;T) / g_gamma(p)
        //
        // is required event by event.
        const double qGammaPhysical =
            physicalPhotonRadialPdf(
                pGamma.P(),
                Tloc
            );

        const double Rgamma =
            qGammaPhysical
            / g_gamma_radial;

        if (!std::isfinite(Rgamma)
            || !(Rgamma >= 0.0)) {
            continue;
        }

        const double fluxMeasure =
            1.0
            /
            (
                4.0
                * pGamma.E()
                * pProton.E()
            );

        // Evaluate the exact massive-proton amplitude in the incoming
        // proton rest frame.
        TLorentzVector pGammaStar =
            pGamma;

        TLorentzVector khStar =
            kh;

        TLorentzVector pProtonStar =
            pProton;

        const TVector3 toProtonRest =
            -pProton.BoostVector();

        pGammaStar.Boost(
            toProtonRest
        );

        khStar.Boost(
            toProtonRest
        );

        pProtonStar.Boost(
            toProtonRest
        );

        const double Ein =
            pGammaStar.E();

        const double omegaStar =
            khStar.E();

        const double c =
            cos3(
                pGammaStar,
                khStar
            );

        const double M2 =
            screenedGammaPM2(
                Ein,
                omegaStar,
                c,
                kappaloc
            );

        const double wRate =
            volumeWeight
            * nGamma
            * nploc
            * Rgamma
            * fluxMeasure
            * w_phi2
            * M2;

        if (!std::isfinite(wRate)) {
            continue;
        }

        ++nFinite;

        if (wRate < 0.0) {
            ++nNegative;
        }

        signedSum +=
            static_cast<long double>(
                wRate
            );

        if (gamma_component == 1) {
            ++nBroad;

            broadSignedSum +=
                static_cast<long double>(
                    wRate
                );
        }

        absSum +=
            std::fabs(
                static_cast<long double>(
                    wRate
                )
            );

        if (
            omega >= omegaMin
            &&
            omega < omegaMax
        ) {

            fillSpectrum(
                spectrum,
                omega,
                wRate,
                logMin,
                dLog
            );
        }

        if ((i + 1) % 1000000 == 0) {

            std::cout
                << "\rWeighted "
                << (i + 1)
                << " / "
                << nValid
                << std::flush;
        }
    }

    std::cout << "\n";

    std::vector<double> spec;
    std::vector<double> err;
    std::vector<double> neff;

    finalizeSpectrum(
        spectrum,
        nProposal,
        dLog,
        spec,
        err,
        neff
    );

    std::ofstream csv(
        csvOut
    );

    if (!csv) {

        std::cerr
            << "ERROR: cannot create "
            << csvOut
            << "\n";

        return 5;
    }

    csv
        << "omega_GeV,"
        << "omega_dP_domega_GeV2,"
        << "mc_error_GeV2,"
        << "raw_event_count,"
        << "effective_weighted_events\n";

    csv
        << std::setprecision(12);

    for (
        int j = 0;
        j < NBIN;
        ++j
    ) {

        csv
            << centers[j]
            << ","
            << spec[j]
            << ","
            << err[j]
            << ","
            << spectrum.count[j]
            << ","
            << neff[j]
            << "\n";
    }

    csv.close();

    std::cout
        << "=== GAMMA-P SPECTRUM SUMMARY ===\n"
        << "N_PROPOSAL="
        << nProposal
        << "\n"
        << "N_VALID_TREE="
        << nValid
        << "\n"
        << "finite_weight_events="
        << nFinite
        << "\n"
        << "negative_weight_events="
        << nNegative
        << "\n"
        << "broad_component_events="
        << nBroad
        << "\n";

    if (absSum > 0.0L) {

        std::cout
            << "signed_over_absolute="
            << static_cast<double>(
                signedSum / absSum
            )
            << "\n";
    }

    if (signedSum != 0.0L) {

        std::cout
            << "weighted_fraction_from_broad_component="
            << static_cast<double>(
                broadSignedSum / signedSum
            )
            << "\n";
    }

    std::cout
        << "output="
        << csvOut
        << "\n";

    return 0;
}
