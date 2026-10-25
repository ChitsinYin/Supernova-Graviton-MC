// weight_brem_spectrum_v1.cpp
//
// Updated screened bremsstrahlung reweighter for:
//
//   ee : e-(p1) + e-(p2) -> e-(k1) + e-(k2) + h(kh)
//   ez : e-(p1) + Z (p2) -> e-(k1) + Z (k2) + h(kh)
//
// Physics input is frozen to the 2026-10-13 handoff:
//
//   * photon propagators are BARE in the amplitude derivation;
//   * no M_eff / omega_pl is used anywhere in the matrix element;
//   * screening is applied once, at squared-amplitude level;
//   * ee: M2_scr = M2_Born * S_g * S_t * S_u;
//   * eZ: M2_scr = bare static-source eikonal kernel * S_Q^2,
//         algebraically implemented as a (Q^2+kappa^2)^(-2) denominator.
//
// Thermal distributions, Pauli blocking, phase-space measure, flux,
// number densities and identical-particle factors are applied outside M2.
//
// Output:
//   CSV columns:
//     omega_GeV,
//     omega_dP_domega_GeV2,
//     mc_error_GeV2,
//     raw_event_count,
//     effective_weighted_events
//
// Usage:
//   ./weight_brem_spectrum_v1 <ee|ez> pool.root output.csv
//
// If ROOT split the tree into pool.root, pool_1.root, ...,
// pass only the base pool.root; the numbered siblings are chained automatically.

#include "TChain.h"
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
#include <limits>
#include <string>
#include <vector>

namespace C {

constexpr long double PI =
    3.141592653589793238462643383279502884L;

constexpr long double ALPHA =
    0.00729735L;

constexpr long double E2 =
    4.0L * PI * ALPHA;

constexpr long double E4 =
    E2 * E2;

constexpr long double M_PLANCK =
    2.435e18L;  // reduced Planck mass, GeV

constexpr long double M_TARGET =
    0.9395L;  // GeV, proton target

constexpr long double TARGET_Z =
    1.0L;

constexpr long double KM3_TO_INVGEV3 =
    1.3005e56L;

// Frozen identical-particle convention from the handoff:
//
// The full labeled final-state ee phase space requires a final-state 1/2!.
// The ordered (p1,p2) thermal integration also requires an initial-state 1/2!.
// Hence the total identical-particle rate factor is 1/4.
constexpr long double EE_FINAL_IDENTICAL_FACTOR =
    0.5L;

constexpr long double EE_INITIAL_PAIR_FACTOR =
    0.5L;

// Numerical tolerance used only to distinguish machine-level negative
// cancellation residue in S_g from a genuinely inconsistent event.
constexpr long double SG_NEGATIVE_REL_TOL =
    1.0e-10L;

}  // namespace C

struct Spectrum {
    std::vector<long double> sum;
    std::vector<long double> sum2;
    std::vector<long long> count;

    explicit Spectrum(int n)
        : sum(n, 0.0L),
          sum2(n, 0.0L),
          count(n, 0)
    {}
};

TLorentzVector unpack4(
    const double a[4]
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

long double dot4(
    const TLorentzVector& a,
    const TLorentzVector& b
)
{
    return
        static_cast<long double>(a.E())
        * static_cast<long double>(b.E())
        -
        static_cast<long double>(a.Px())
        * static_cast<long double>(b.Px())
        -
        static_cast<long double>(a.Py())
        * static_cast<long double>(b.Py())
        -
        static_cast<long double>(a.Pz())
        * static_cast<long double>(b.Pz());
}

double clampCos(double x)
{
    return
        std::max(
            -1.0,
            std::min(
                1.0,
                x
            )
        );
}

double cos3(
    const TVector3& a,
    const TVector3& b
)
{
    const double aa = a.Mag();
    const double bb = b.Mag();

    if (!(aa > 0.0)
        || !(bb > 0.0)) {
        return 1.0;
    }

    return
        clampCos(
            a.Dot(b)
            / (aa * bb)
        );
}

long double fFD(
    long double E,
    long double T,
    long double mu
)
{
    const long double x =
        (E - mu)
        / T;

    if (x > 60.0L) {
        return std::exp(-x);
    }

    if (x < -60.0L) {
        return 1.0L;
    }

    return
        1.0L
        /
        (
            std::exp(x)
            + 1.0L
        );
}

long double electronNormUR(
    long double T,
    long double mu
)
{
    return
        (4.0L * C::PI / 3.0L)
        * std::pow(mu, 3)
        * (
            1.0L
            + std::pow(
                C::PI * T / mu,
                2
            )
        );
}

long double targetThermal(
    long double p,
    long double T
)
{
    const long double E =
        std::sqrt(
            p * p
            + C::M_TARGET
              * C::M_TARGET
        );

    return
        std::exp(
            -(E - C::M_TARGET)
            / T
        );
}

long double targetNormHeavy(
    long double T
)
{
    return
        std::pow(
            2.0L
            * C::PI
            * C::M_TARGET
            * T,
            1.5L
        );
}

// ---------------------------------------------------------------------------
// ee screened leading-soft kernel
// ---------------------------------------------------------------------------
//
// Handoff definitions:
//
//   s = (p1+p2)^2
//   t = (p1-k1)^2
//   u = -s-t              [strict leading-soft hard projection]
//
//   X = 2 p1.k
//   Y = 2 p2.k
//   A = 2 k1.k
//   B = 2 k2.k
//
//   M2Born = 4 e^4 (s^2+s t+t^2)^2 / (t^2 u^2)
//
//   Sg = (2/MP^2)[
//          s^2/(XY) + s^2/(AB)
//        - t^2/(XA) - u^2/(XB)
//        - u^2/(YA) - t^2/(YB)]
//
//   St = (-t)/((-t)+kappa^2)
//   Su = (-u)/((-u)+kappa^2)
//
//   M2 = M2Born*Sg*St*Su.
//
// The soft products are computed from the exact 2->3 event.  The hard u
// is NOT replaced by exact (p1-k2)^2.
struct EEM2Result {
    long double value = 0.0L;
    bool valid = false;
    bool clampedTinyNegativeSg = false;
    bool badNegativeSg = false;
};

EEM2Result M2EEScreened(
    const TLorentzVector& p1,
    const TLorentzVector& p2,
    const TLorentzVector& k1,
    const TLorentzVector& k2,
    const TLorentzVector& kh,
    long double kappaDebye
)
{
    EEM2Result out{};

    const TLorentzVector Pin =
        p1 + p2;

    const TLorentzVector qT =
        p1 - k1;

    const long double s =
        dot4(Pin, Pin);

    const long double t =
        dot4(qT, qT);

    // Frozen leading-soft hard mapping.
    const long double u =
        -s - t;

    const long double X =
        2.0L * dot4(p1, kh);

    const long double Y =
        2.0L * dot4(p2, kh);

    const long double A =
        2.0L * dot4(k1, kh);

    const long double B =
        2.0L * dot4(k2, kh);

    if (!(s > 0.0L)
        || !(t < 0.0L)
        || !(u < 0.0L)
        || !(X > 0.0L)
        || !(Y > 0.0L)
        || !(A > 0.0L)
        || !(B > 0.0L)
        || !(kappaDebye >= 0.0L)) {
        return out;
    }

    const long double poly =
        s * s
        + s * t
        + t * t;

    const long double M2Born =
        4.0L
        * C::E4
        * poly * poly
        /
        (
            t * t
            * u * u
        );

    const long double term1 =
        s * s
        / (X * Y);

    const long double term2 =
        s * s
        / (A * B);

    const long double term3 =
        -t * t
        / (X * A);

    const long double term4 =
        -u * u
        / (X * B);

    const long double term5 =
        -u * u
        / (Y * A);

    const long double term6 =
        -t * t
        / (Y * B);

    long double sgBracket =
        term1
        + term2
        + term3
        + term4
        + term5
        + term6;

    const long double sgScale =
        std::fabs(term1)
        + std::fabs(term2)
        + std::fabs(term3)
        + std::fabs(term4)
        + std::fabs(term5)
        + std::fabs(term6);

    if (sgBracket < 0.0L) {

        if (
            sgScale > 0.0L
            &&
            std::fabs(sgBracket)
            <= C::SG_NEGATIVE_REL_TOL
               * sgScale
        ) {

            sgBracket = 0.0L;
            out.clampedTinyNegativeSg = true;
        }
        else {

            out.badNegativeSg = true;
            return out;
        }
    }

    const long double Sg =
        2.0L
        / (
            C::M_PLANCK
            * C::M_PLANCK
        )
        * sgBracket;

    const long double Qt2 =
        -t;

    const long double Qu2 =
        -u;

    const long double kappa2 =
        kappaDebye
        * kappaDebye;

    const long double St =
        Qt2
        / (Qt2 + kappa2);

    const long double Su =
        Qu2
        / (Qu2 + kappa2);

    const long double M2 =
        M2Born
        * Sg
        * St
        * Su;

    if (!(M2 >= 0.0L)
        || !std::isfinite(
            static_cast<double>(M2)
        )) {
        return out;
    }

    out.value = M2;
    out.valid = true;

    return out;
}

// ---------------------------------------------------------------------------
// eZ screened static-source eikonal kernel
// ---------------------------------------------------------------------------
//
// This kernel MUST be evaluated in the initial-target rest frame.
//
// Frozen mapping:
//
//   E     = incoming electron energy
//   omega = graviton energy
//   Ef    = E - omega
//
// with exact event directions ui, uf, n.
//
//   Q^2 = E^2 + Ef^2 + omega^2
//         -2 E Ef ce -2 E omega ci +2 Ef omega cf
//
//   R_TT = [E(1+ci)-Ef(1+cf)]^2
//          +4 E Ef(1+ci)(1+cf) sin^2(phi_eZ)
//
// Final screened kernel:
//
//   M2 = 4 Z^2 e^4 M^2 E Ef (1+ce) R_TT
//        / [MP^2 omega^2 (Q^2+kappa^2)^2].
//
// No M_eff and no additional angular screening are allowed.
struct EZM2Result {
    long double value = 0.0L;
    bool valid = false;
    bool collinearAzimuthDegeneracy = false;
    bool tinyNegativeQ2Clamped = false;
};

EZM2Result M2EZScreenedStatic(
    const TLorentzVector& pElectronIn,
    const TLorentzVector& pTargetIn,
    const TLorentzVector& pElectronOut,
    const TLorentzVector& kh,
    long double kappaDebye
)
{
    EZM2Result out{};

    TLorentzVector p =
        pElectronIn;

    TLorentzVector pf =
        pElectronOut;

    TLorentzVector h =
        kh;

    // Boost all required four-vectors into the initial-target rest frame.
    const TVector3 toTargetRest =
        -pTargetIn.BoostVector();

    p.Boost(toTargetRest);
    pf.Boost(toTargetRest);
    h.Boost(toTargetRest);

    const long double E =
        static_cast<long double>(
            p.E()
        );

    const long double omega =
        static_cast<long double>(
            h.E()
        );

    const long double Ef =
        E - omega;

    if (!(E > 0.0L)
        || !(omega > 0.0L)
        || !(Ef > 0.0L)
        || !(omega < E)
        || !(kappaDebye >= 0.0L)) {
        return out;
    }

    const TVector3 ui =
        p.Vect().Unit();

    const TVector3 uf =
        pf.Vect().Unit();

    const TVector3 n =
        h.Vect().Unit();

    const long double ci =
        static_cast<long double>(
            cos3(ui, n)
        );

    const long double cf =
        static_cast<long double>(
            cos3(uf, n)
        );

    const long double ce =
        static_cast<long double>(
            cos3(ui, uf)
        );

    // sin^2(phi_eZ) from the oriented transverse geometry:
    //
    // n . (vi x vf) = n . (ui x uf),
    // |vi|^2 = 1-ci^2, |vf|^2 = 1-cf^2.
    const long double vi2 =
        std::max(
            0.0L,
            1.0L - ci * ci
        );

    const long double vf2 =
        std::max(
            0.0L,
            1.0L - cf * cf
        );

    long double sinPhi2 = 0.0L;

    const long double transverseDen =
        vi2 * vf2;

    if (transverseDen > 1.0e-28L) {

        const long double triple =
            static_cast<long double>(
                n.Dot(
                    ui.Cross(uf)
                )
            );

        sinPhi2 =
            triple * triple
            / transverseDen;

        sinPhi2 =
            std::max(
                0.0L,
                std::min(
                    1.0L,
                    sinPhi2
                )
            );
    }
    else {

        // Exact collinear azimuth is a measure-zero coordinate degeneracy.
        // Use the neighboring-limit convention sin^2(phi)=0 and count it.
        out.collinearAzimuthDegeneracy = true;
        sinPhi2 = 0.0L;
    }

    long double Q2 =
        E * E
        + Ef * Ef
        + omega * omega
        - 2.0L * E * Ef * ce
        - 2.0L * E * omega * ci
        + 2.0L * Ef * omega * cf;

    // Q2 is a positive spatial momentum-transfer squared.
    if (Q2 < 0.0L) {

        const long double qScale =
            E * E
            + Ef * Ef
            + omega * omega;

        if (
            qScale > 0.0L
            &&
            std::fabs(Q2)
            <= 1.0e-12L
               * qScale
        ) {

            Q2 = 0.0L;
            out.tinyNegativeQ2Clamped = true;
        }
        else {

            return out;
        }
    }

    const long double radiationTT =
        std::pow(
            E * (1.0L + ci)
            - Ef * (1.0L + cf),
            2
        )
        +
        4.0L
        * E
        * Ef
        * (1.0L + ci)
        * (1.0L + cf)
        * sinPhi2;

    if (!(radiationTT >= 0.0L)) {
        return out;
    }

    const long double denom =
        C::M_PLANCK
        * C::M_PLANCK
        * omega
        * omega
        * std::pow(
            Q2
            + kappaDebye
              * kappaDebye,
            2
        );

    if (!(denom > 0.0L)) {
        return out;
    }

    const long double M2 =
        4.0L
        * C::TARGET_Z
        * C::TARGET_Z
        * C::E4
        * C::M_TARGET
        * C::M_TARGET
        * E
        * Ef
        * (1.0L + ce)
        * radiationTT
        / denom;

    if (!(M2 >= 0.0L)
        || !std::isfinite(
            static_cast<double>(M2)
        )) {
        return out;
    }

    out.value = M2;
    out.valid = true;

    return out;
}

void fillSpectrum(
    Spectrum& s,
    long double omega,
    long double wRate,
    long double logMin,
    long double dLog
)
{
    if (!(omega > 0.0L)
        || !std::isfinite(
            static_cast<double>(wRate)
        )) {
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

    // wRate estimates dGamma.  Multiplying by omega converts rate to power.
    // With logarithmic bins, this directly estimates
    //     omega dP/domega = dP/dln(omega).
    const long double z =
        omega
        * wRate;

    s.sum[ibin] += z;
    s.sum2[ibin] += z * z;
    s.count[ibin] += 1;
}

void finalizeSpectrum(
    const Spectrum& in,
    long long nProposal,
    long double dLog,
    std::vector<long double>& spec,
    std::vector<long double>& err,
    std::vector<long double>& neff
)
{
    const int n =
        static_cast<int>(
            in.sum.size()
        );

    spec.assign(n, 0.0L);
    err.assign(n, 0.0L);
    neff.assign(n, 0.0L);

    for (
        int j = 0;
        j < n;
        ++j
    ) {

        spec[j] =
            in.sum[j]
            /
            static_cast<long double>(
                nProposal
            )
            /
            dLog;

        if (nProposal > 1) {

            const long double numerator =
                in.sum2[j]
                -
                in.sum[j]
                * in.sum[j]
                /
                static_cast<long double>(
                    nProposal
                );

            const long double varianceMean =
                std::max(
                    0.0L,
                    numerator
                )
                /
                (
                    static_cast<long double>(
                        nProposal
                    )
                    *
                    static_cast<long double>(
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

        if (in.sum2[j] > 0.0L) {

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
    if (argc < 4) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <ee|ez> pool.root output.csv\n";

        return 1;
    }

    const std::string channel =
        argv[1];

    const bool isEE =
        channel == "ee";

    const bool isEZ =
        channel == "ez";

    if (!isEE && !isEZ) {

        std::cerr
            << "ERROR: channel must be ee or ez\n";

        return 2;
    }

    const std::string poolPath =
        argv[2];

    const std::string csvOut =
        argv[3];

    const std::filesystem::path csvPath(csvOut);

    if (csvPath.has_parent_path()) {
        std::filesystem::create_directories(
            csvPath.parent_path()
        );
    }

    // TTree automatically starts a new file after reaching ROOT's
    // maximum tree size.  A 500M pool can therefore appear as
    //
    //   sample.root
    //   sample_1.root
    //   sample_2.root
    //   ...
    //
    // Treat all such files as one logical event pool using TChain.
    const std::filesystem::path basePoolPath(
        poolPath
    );

    std::vector<std::string> poolFiles;
    poolFiles.push_back(
        basePoolPath.string()
    );

    const std::filesystem::path parentDir =
        basePoolPath.parent_path();

    const std::string stem =
        basePoolPath.stem().string();

    const std::string extension =
        basePoolPath.extension().string();

    for (
        int iPart = 1;
        ;
        ++iPart
    ) {

        const std::filesystem::path partPath =
            parentDir
            /
            (
                stem
                + "_"
                + std::to_string(iPart)
                + extension
            );

        if (!std::filesystem::exists(
                partPath
            )) {
            break;
        }

        poolFiles.push_back(
            partPath.string()
        );
    }

    TChain chain("events");

    for (const auto& file : poolFiles) {

        if (chain.Add(
                file.c_str()
            ) == 0) {

            std::cerr
                << "ERROR: cannot add ROOT tree file "
                << file
                << "\n";

            return 3;
        }
    }

    TTree* tree =
        &chain;

    // Metadata can end up in the last split file because TTree::ChangeFile
    // changes the current ROOT directory.  Search the split set from the
    // last file backwards and use the first complete metadata block found.
    Long64_t nProposal =
        -1;

    double R_km_value =
        -1.0;

    std::string metadataChannel;
    std::string kernelVersion;
    std::string metadataFile;

    bool metadataFound =
        false;

    for (
        auto it = poolFiles.rbegin();
        it != poolFiles.rend();
        ++it
    ) {

        TFile metaFile(
            it->c_str(),
            "READ"
        );

        if (metaFile.IsZombie()) {
            continue;
        }

        auto* pNProposal =
            dynamic_cast<TParameter<Long64_t>*>(
                metaFile.Get(
                    "N_PROPOSAL"
                )
            );

        auto* pR =
            dynamic_cast<TParameter<double>*>(
                metaFile.Get(
                    "R_KM"
                )
            );

        auto* pChannel =
            dynamic_cast<TNamed*>(
                metaFile.Get(
                    "CHANNEL"
                )
            );

        auto* pKernelVersion =
            dynamic_cast<TNamed*>(
                metaFile.Get(
                    "KERNEL_VERSION"
                )
            );

        if (
            pNProposal
            && pR
            && pChannel
        ) {

            nProposal =
                pNProposal->GetVal();

            R_km_value =
                pR->GetVal();

            metadataChannel =
                pChannel->GetTitle();

            if (pKernelVersion) {
                kernelVersion =
                    pKernelVersion->GetTitle();
            }

            metadataFile =
                *it;

            metadataFound =
                true;

            break;
        }
    }

    if (!metadataFound) {

        std::cerr
            << "ERROR: required metadata not found in any split ROOT file\n";

        return 4;
    }

    if (metadataChannel != channel) {

        std::cerr
            << "ERROR: channel mismatch between command and ROOT pool\n";

        return 5;
    }

    const Long64_t nValid =
        tree->GetEntries();

    if (nValid != nProposal) {

        std::cout
            << "WARNING: chained_entries="
            << nValid
            << " differs from N_PROPOSAL="
            << nProposal
            << "\n";
    }

    std::cout
        << "pool_files="
        << poolFiles.size()
        << "\n";

    for (
        std::size_t iFile = 0;
        iFile < poolFiles.size();
        ++iFile
    ) {

        std::cout
            << "pool_file["
            << iFile
            << "]="
            << poolFiles[iFile]
            << "\n";
    }

    std::cout
        << "metadata_file="
        << metadataFile
        << "\n";

    if (!kernelVersion.empty()) {

        std::cout
            << "pool_kernel_version="
            << kernelVersion
            << "\n";
    }

    const long double R_km =
        static_cast<long double>(
            R_km_value
        );

    const long double volumeWeight =
        (4.0L * C::PI / 3.0L)
        * std::pow(
            R_km,
            3
        )
        * C::KM3_TO_INVGEV3;

    constexpr int NBIN =
        180;

    const long double omegaMin =
        8.0e-5L;

    const long double omegaMax =
        3.3L;

    const long double logMin =
        std::log(
            omegaMin
        );

    const long double logMax =
        std::log(
            omegaMax
        );

    const long double dLog =
        (logMax - logMin)
        / static_cast<long double>(
            NBIN
        );

    std::vector<long double> centers(
        NBIN
    );

    for (
        int i = 0;
        i < NBIN;
        ++i
    ) {

        const long double lo =
            std::exp(
                logMin
                + dLog
                  * static_cast<long double>(i)
            );

        const long double hi =
            std::exp(
                logMin
                + dLog
                  * static_cast<long double>(i + 1)
            );

        centers[i] =
            std::sqrt(
                lo * hi
            );
    }

    Spectrum spectrum(NBIN);

    double r_km = 0.0;
    double T = 0.0;
    double ne = 0.0;
    double mu = 0.0;
    double kappa = 0.0;

    double p1a[4] = {0, 0, 0, 0};
    double p2a[4] = {0, 0, 0, 0};

    double k1a[4] = {0, 0, 0, 0};
    double k2a[4] = {0, 0, 0, 0};
    double kha[4] = {0, 0, 0, 0};

    double g1_radial = 0.0;
    double g2_radial = 0.0;

    int component1 = -1;
    int component2 = -1;

    double w_tgen = 0.0;
    double w_phi3 = 0.0;

    tree->SetBranchAddress(
        "r_km",
        &r_km
    );

    tree->SetBranchAddress(
        "T",
        &T
    );

    tree->SetBranchAddress(
        "ne",
        &ne
    );

    tree->SetBranchAddress(
        "mu",
        &mu
    );

    tree->SetBranchAddress(
        "kappa",
        &kappa
    );

    tree->SetBranchAddress(
        "p1",
        p1a
    );

    tree->SetBranchAddress(
        "p2",
        p2a
    );

    tree->SetBranchAddress(
        "k1",
        k1a
    );

    tree->SetBranchAddress(
        "k2",
        k2a
    );

    tree->SetBranchAddress(
        "kh",
        kha
    );

    tree->SetBranchAddress(
        "g1_radial",
        &g1_radial
    );

    tree->SetBranchAddress(
        "g2_radial",
        &g2_radial
    );

    tree->SetBranchAddress(
        "component1",
        &component1
    );

    tree->SetBranchAddress(
        "component2",
        &component2
    );

    tree->SetBranchAddress(
        "w_tgen",
        &w_tgen
    );

    tree->SetBranchAddress(
        "w_phi3",
        &w_phi3
    );

    long long nFinite = 0;
    long long nNegative = 0;
    long long nInvalidM2 = 0;

    long long nEETinyNegativeSgClamped = 0;
    long long nEEBadNegativeSg = 0;

    long long nEZCollinearAzimuth = 0;
    long long nEZTinyNegativeQ2Clamped = 0;

    long long nAnyBroad = 0;

    long double signedSum = 0.0L;
    long double absSum = 0.0L;

    long double weightedAnyBroad = 0.0L;
    long double weightedTotal = 0.0L;

    for (
        Long64_t i = 0;
        i < nValid;
        ++i
    ) {

        tree->GetEntry(i);

        const long double Tloc =
            static_cast<long double>(T);

        const long double muloc =
            static_cast<long double>(mu);

        const long double neloc =
            static_cast<long double>(ne);

        const long double kappaloc =
            static_cast<long double>(
                kappa
            );

        if (!(Tloc > 0.0L)
            || !(muloc > 0.0L)
            || !(neloc > 0.0L)
            || !(kappaloc >= 0.0L)
            || !(g1_radial > 0.0)
            || !(g2_radial > 0.0)
            || !(w_phi3 >= 0.0)) {
            continue;
        }

        const TLorentzVector p1 =
            unpack4(p1a);

        const TLorentzVector p2 =
            unpack4(p2a);

        const TLorentzVector k1 =
            unpack4(k1a);

        const TLorentzVector k2 =
            unpack4(k2a);

        const TLorentzVector kh =
            unpack4(kha);

        const long double omega =
            static_cast<long double>(
                kh.E()
            );

        if (!(omega > 0.0L)
            || !(p1.E() > 0.0)
            || !(p2.E() > 0.0)) {
            continue;
        }

        const long double De =
            electronNormUR(
                Tloc,
                muloc
            );

        const long double p1mag =
            static_cast<long double>(
                p1.P()
            );

        const long double f1 =
            fFD(
                static_cast<long double>(
                    p1.E()
                ),
                Tloc,
                muloc
            );

        const long double R1 =
            4.0L
            * C::PI
            * p1mag
            * p1mag
            * f1
            /
            (
                De
                * static_cast<long double>(
                    g1_radial
                )
            );

        long double R2 = 0.0L;

        if (isEE) {

            const long double p2mag =
                static_cast<long double>(
                    p2.P()
                );

            const long double f2 =
                fFD(
                    static_cast<long double>(
                        p2.E()
                    ),
                    Tloc,
                    muloc
                );

            R2 =
                4.0L
                * C::PI
                * p2mag
                * p2mag
                * f2
                /
                (
                    De
                    * static_cast<long double>(
                        g2_radial
                    )
                );
        }
        else {

            const long double p2mag =
                static_cast<long double>(
                    p2.P()
                );

            const long double fTarget =
                targetThermal(
                    p2mag,
                    Tloc
                );

            const long double DTarget =
                targetNormHeavy(
                    Tloc
                );

            R2 =
                4.0L
                * C::PI
                * p2mag
                * p2mag
                * fTarget
                /
                (
                    DTarget
                    * static_cast<long double>(
                        g2_radial
                    )
                );
        }

        if (!std::isfinite(
                static_cast<double>(R1)
            )
            || !std::isfinite(
                static_cast<double>(R2)
            )
            || R1 < 0.0L
            || R2 < 0.0L) {
            continue;
        }

        const long double fluxMeasure =
            1.0L
            /
            (
                4.0L
                * static_cast<long double>(
                    p1.E()
                )
                * static_cast<long double>(
                    p2.E()
                )
            );

        if (!(fluxMeasure > 0.0L)) {
            continue;
        }

        const long double common =
            volumeWeight
            * neloc
            * neloc
            * R1
            * R2
            * fluxMeasure
            * static_cast<long double>(
                w_phi3
            );

        long double M2 = 0.0L;
        long double blocking = 0.0L;
        long double identicalFactor = 1.0L;

        if (isEE) {

            const EEM2Result m2res =
                M2EEScreened(
                    p1,
                    p2,
                    k1,
                    k2,
                    kh,
                    kappaloc
                );

            if (m2res.clampedTinyNegativeSg) {
                ++nEETinyNegativeSgClamped;
            }

            if (m2res.badNegativeSg) {
                ++nEEBadNegativeSg;
            }

            if (!m2res.valid) {
                ++nInvalidM2;
                continue;
            }

            M2 =
                m2res.value;

            const long double b1 =
                1.0L
                - fFD(
                    static_cast<long double>(
                        k1.E()
                    ),
                    Tloc,
                    muloc
                );

            const long double b2 =
                1.0L
                - fFD(
                    static_cast<long double>(
                        k2.E()
                    ),
                    Tloc,
                    muloc
                );

            blocking =
                b1 * b2;

            identicalFactor =
                C::EE_FINAL_IDENTICAL_FACTOR
                * C::EE_INITIAL_PAIR_FACTOR;
        }
        else {

            const EZM2Result m2res =
                M2EZScreenedStatic(
                    p1,
                    p2,
                    k1,
                    kh,
                    kappaloc
                );

            if (m2res.collinearAzimuthDegeneracy) {
                ++nEZCollinearAzimuth;
            }

            if (m2res.tinyNegativeQ2Clamped) {
                ++nEZTinyNegativeQ2Clamped;
            }

            if (!m2res.valid) {
                ++nInvalidM2;
                continue;
            }

            M2 =
                m2res.value;

            blocking =
                1.0L
                - fFD(
                    static_cast<long double>(
                        k1.E()
                    ),
                    Tloc,
                    muloc
                );
        }

        const long double wRate =
            common
            * blocking
            * identicalFactor
            * M2;

        if (!std::isfinite(
                static_cast<double>(wRate)
            )) {
            continue;
        }

        ++nFinite;

        if (wRate < 0.0L) {
            ++nNegative;
        }

        signedSum +=
            wRate;

        absSum +=
            std::fabs(
                wRate
            );

        const bool anyBroad =
            component1 == 1
            || component2 == 1;

        if (anyBroad) {

            ++nAnyBroad;

            weightedAnyBroad +=
                wRate;
        }

        weightedTotal +=
            wRate;

        if (
            omega >= omegaMin
            && omega < omegaMax
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
                << "\r"
                << channel
                << " weighted: "
                << (i + 1)
                << " / "
                << nValid
                << std::flush;
        }
    }

    std::cout << "\n";

    std::vector<long double> spec;
    std::vector<long double> err;
    std::vector<long double> neff;

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

        return 6;
    }

    csv
        << "omega_GeV,"
        << "omega_dP_domega_GeV2,"
        << "mc_error_GeV2,"
        << "raw_event_count,"
        << "effective_weighted_events\n";

    csv
        << std::setprecision(14);

    for (
        int j = 0;
        j < NBIN;
        ++j
    ) {

        csv
            << static_cast<double>(
                centers[j]
            )
            << ","
            << static_cast<double>(
                spec[j]
            )
            << ","
            << static_cast<double>(
                err[j]
            )
            << ","
            << spectrum.count[j]
            << ","
            << static_cast<double>(
                neff[j]
            )
            << "\n";
    }

    csv.close();

    std::cout
        << "=== "
        << channel
        << " SCREENED-BREM-V1 SPECTRUM SUMMARY ===\n"
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
        << "invalid_M2_events="
        << nInvalidM2
        << "\n"
        << "any_broad_component_events="
        << nAnyBroad
        << "\n";

    if (isEE) {

        std::cout
            << "ee_final_identical_factor="
            << static_cast<double>(
                C::EE_FINAL_IDENTICAL_FACTOR
            )
            << "\n"
            << "ee_initial_pair_factor="
            << static_cast<double>(
                C::EE_INITIAL_PAIR_FACTOR
            )
            << "\n"
            << "ee_tiny_negative_Sg_clamped="
            << nEETinyNegativeSgClamped
            << "\n"
            << "ee_bad_negative_Sg_rejected="
            << nEEBadNegativeSg
            << "\n";
    }
    else {

        std::cout
            << "ez_collinear_azimuth_degeneracy="
            << nEZCollinearAzimuth
            << "\n"
            << "ez_tiny_negative_Q2_clamped="
            << nEZTinyNegativeQ2Clamped
            << "\n";
    }

    if (absSum > 0.0L) {

        std::cout
            << "signed_over_absolute="
            << static_cast<double>(
                signedSum / absSum
            )
            << "\n";
    }

    if (weightedTotal != 0.0L) {

        std::cout
            << "weighted_fraction_from_any_broad_component="
            << static_cast<double>(
                weightedAnyBroad
                / weightedTotal
            )
            << "\n";
    }

    std::cout
        << "output="
        << csvOut
        << "\n";

    return 0;
}
