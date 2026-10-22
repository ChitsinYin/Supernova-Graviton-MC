// generate_brem_phsp_v1.cpp
//
// ROOT/PHSP generator for the updated screened bremsstrahlung calculation:
//
//   ee : e-(p1) + e-(p2) -> e-(k1) + e-(k2) + h(kh)
//   ez : e-(p1) + Z (p2) -> e-(k1) + Z (k2) + h(kh)
//
// This file generates ONLY thermal initial states + exact three-body
// relativistic kinematics.  The updated bare squared amplitudes and the
// phenomenological Debye-screening weights are applied later by
// weight_brem_spectrum_v1.cpp.
//
// Frozen conventions for this v1 implementation:
//   * all external electrons are massless;
//   * graviton is massless;
//   * eZ uses a proton target: Z=1, M=0.9395 GeV;
//   * initial thermal sampling is performed in the local SN plasma frame;
//   * full-support importance proposals are retained from the previous
//     production framework.
//
// Usage:
//   ./generate_brem_phsp_v1 <ee|ez> profile.csv output.root
//                           [Nproposal] [seed] [R_km]

#include "TFile.h"
#include "TGenPhaseSpace.h"
#include "TLorentzVector.h"
#include "TNamed.h"
#include "TParameter.h"
#include "TRandom.h"
#include "TTree.h"
#include "TVector3.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace C {

constexpr double PI =
    3.141592653589793238462643383279502884;

constexpr double ALPHA =
    0.00729735;

constexpr double M_TARGET =
    0.9395;  // GeV, proton target

constexpr double TARGET_Z =
    1.0;

constexpr double RHO_TO_GEV4 =
    4.31e-18;

constexpr double ELECTRON_BROAD_FRACTION =
    0.15;

constexpr double TARGET_BROAD_FRACTION =
    0.10;

}  // namespace C

struct ProfileRow {
    double r_km;
    double rho_gcm3;
    double yp;
    double T_MeV;
};

class SNProfile {
public:
    bool load(const std::string& path)
    {
        std::ifstream in(path);

        if (!in) {
            return false;
        }

        std::string line;

        while (std::getline(in, line)) {

            if (line.empty()) {
                continue;
            }

            std::replace(
                line.begin(),
                line.end(),
                ',',
                ' '
            );

            std::stringstream ss(line);

            ProfileRow row{};

            if (!(ss
                  >> row.r_km
                  >> row.rho_gcm3
                  >> row.yp
                  >> row.T_MeV)) {
                // Allows a header or other non-numeric lines.
                continue;
            }

            rows_.push_back(row);
        }

        std::sort(
            rows_.begin(),
            rows_.end(),
            [](
                const ProfileRow& a,
                const ProfileRow& b
            ) {
                return a.r_km < b.r_km;
            }
        );

        return rows_.size() >= 2;
    }

    ProfileRow eval(double r) const
    {
        if (r <= rows_.front().r_km) {
            return rows_.front();
        }

        if (r >= rows_.back().r_km) {
            return rows_.back();
        }

        auto it =
            std::lower_bound(
                rows_.begin(),
                rows_.end(),
                r,
                [](
                    const ProfileRow& a,
                    double x
                ) {
                    return a.r_km < x;
                }
            );

        const auto& hi = *it;
        const auto& lo = *(it - 1);

        const double f =
            (r - lo.r_km)
            /
            (hi.r_km - lo.r_km);

        ProfileRow out{};

        out.r_km = r;

        out.rho_gcm3 =
            lo.rho_gcm3
            + f
              * (hi.rho_gcm3 - lo.rho_gcm3);

        out.yp =
            lo.yp
            + f
              * (hi.yp - lo.yp);

        out.T_MeV =
            lo.T_MeV
            + f
              * (hi.T_MeV - lo.T_MeV);

        return out;
    }

    size_t size() const
    {
        return rows_.size();
    }

private:
    std::vector<ProfileRow> rows_;
};

double safeUniform()
{
    double u = gRandom->Rndm();

    while (!(u > 0.0) || !(u < 1.0)) {
        u = gRandom->Rndm();
    }

    return u;
}

TVector3 isotropicVector(double p)
{
    const double c =
        2.0 * gRandom->Rndm() - 1.0;

    const double phi =
        2.0 * C::PI * gRandom->Rndm();

    const double s =
        std::sqrt(
            std::max(
                0.0,
                1.0 - c * c
            )
        );

    return TVector3(
        p * s * std::cos(phi),
        p * s * std::sin(phi),
        p * c
    );
}

void pack4(
    const TLorentzVector& v,
    double out[4]
)
{
    out[0] = v.Px();
    out[1] = v.Py();
    out[2] = v.Pz();
    out[3] = v.E();
}

double pdk(
    double parentMass,
    double mA,
    double mB
)
{
    if (!(parentMass > 0.0)) {
        return 0.0;
    }

    const double x =
        (parentMass - mA - mB)
        * (parentMass + mA + mB)
        * (parentMass - mA + mB)
        * (parentMass + mA - mB);

    if (!(x > 0.0)) {
        return 0.0;
    }

    return
        std::sqrt(x)
        /
        (2.0 * parentMass);
}

// Absolute 3-body Lorentz-invariant phase-space measure divided by the
// TGenPhaseSpace proposal for the n=3 Raubold-Lynch construction.
//
// This is the same normalization already validated in the previous
// ROOT/PHSP production framework.
double phi3ProposalWeight(
    const TLorentzVector& parent,
    const TLorentzVector& d0,
    const TLorentzVector& d1,
    double m0,
    double m1,
    double m2
)
{
    const double M =
        parent.M();

    const double deltaM =
        M - m0 - m1 - m2;

    if (!(M > 0.0)
        || !(deltaM > 0.0)) {
        return 0.0;
    }

    const double m01 =
        (d0 + d1).M();

    if (!(m01 > 0.0)) {
        return 0.0;
    }

    const double pInner =
        pdk(
            m01,
            m0,
            m1
        );

    const double pOuter =
        pdk(
            M,
            m01,
            m2
        );

    return
        deltaM
        * pInner
        * pOuter
        /
        (
            16.0
            * std::pow(C::PI, 3)
            * M
        );
}

double normalCDF(double x)
{
    return
        0.5
        * (
            1.0
            + std::erf(
                x / std::sqrt(2.0)
            )
        );
}

// Truncated Gaussian proposal around the Fermi surface.
double electronShellPdf(
    double p,
    double mu,
    double T
)
{
    if (p < 0.0) {
        return 0.0;
    }

    const double sigma =
        std::max(
            4.0 * T,
            1.0e-8
        );

    const double Z =
        std::max(
            normalCDF(
                mu / sigma
            ),
            1.0e-15
        );

    const double z =
        (p - mu)
        / sigma;

    return
        std::exp(
            -0.5 * z * z
        )
        /
        (
            sigma
            * std::sqrt(
                2.0 * C::PI
            )
            * Z
        );
}

double electronBroadScale(
    double mu,
    double T
)
{
    return
        std::max(
            std::max(
                mu,
                10.0 * T
            ),
            1.0e-6
        );
}

double electronBroadPdf(
    double p,
    double mu,
    double T
)
{
    if (p < 0.0) {
        return 0.0;
    }

    const double L =
        electronBroadScale(
            mu,
            T
        );

    return
        std::exp(-p / L)
        / L;
}

double electronMixturePdf(
    double p,
    double mu,
    double T
)
{
    return
        (
            1.0
            - C::ELECTRON_BROAD_FRACTION
        )
        * electronShellPdf(
            p,
            mu,
            T
        )
        +
        C::ELECTRON_BROAD_FRACTION
        * electronBroadPdf(
            p,
            mu,
            T
        );
}

void sampleElectronMomentum(
    double mu,
    double T,
    double& p,
    double& gRadial,
    int& component
)
{
    if (
        gRandom->Rndm()
        <
        (
            1.0
            - C::ELECTRON_BROAD_FRACTION
        )
    ) {

        component = 0;

        const double sigma =
            std::max(
                4.0 * T,
                1.0e-8
            );

        do {
            p =
                gRandom->Gaus(
                    mu,
                    sigma
                );
        }
        while (!(p > 0.0));
    }
    else {

        component = 1;

        const double L =
            electronBroadScale(
                mu,
                T
            );

        p =
            -L
            * std::log(
                safeUniform()
            );

        if (!(p > 0.0)) {
            p =
                std::numeric_limits<double>::min();
        }
    }

    gRadial =
        electronMixturePdf(
            p,
            mu,
            T
        );
}

double targetMaxwellPdf(
    double p,
    double T
)
{
    if (p < 0.0) {
        return 0.0;
    }

    const double sigma =
        std::sqrt(
            C::M_TARGET * T
        );

    if (!(sigma > 0.0)) {
        return 0.0;
    }

    return
        std::sqrt(
            2.0 / C::PI
        )
        * p * p
        / std::pow(sigma, 3)
        * std::exp(
            -p * p
            /
            (
                2.0
                * sigma
                * sigma
            )
        );
}

double targetBroadScale(double T)
{
    const double sigma =
        std::sqrt(
            C::M_TARGET * T
        );

    return
        std::max(
            std::max(
                4.0 * sigma,
                10.0 * T
            ),
            1.0e-6
        );
}

double targetBroadPdf(
    double p,
    double T
)
{
    if (p < 0.0) {
        return 0.0;
    }

    const double L =
        targetBroadScale(T);

    return
        std::exp(-p / L)
        / L;
}

double targetMixturePdf(
    double p,
    double T
)
{
    return
        (
            1.0
            - C::TARGET_BROAD_FRACTION
        )
        * targetMaxwellPdf(
            p,
            T
        )
        +
        C::TARGET_BROAD_FRACTION
        * targetBroadPdf(
            p,
            T
        );
}

void sampleTargetMomentum(
    double T,
    double& p,
    double& gRadial,
    int& component
)
{
    if (
        gRandom->Rndm()
        <
        (
            1.0
            - C::TARGET_BROAD_FRACTION
        )
    ) {

        component = 0;

        const double sigma =
            std::sqrt(
                C::M_TARGET * T
            );

        const double px =
            gRandom->Gaus(
                0.0,
                sigma
            );

        const double py =
            gRandom->Gaus(
                0.0,
                sigma
            );

        const double pz =
            gRandom->Gaus(
                0.0,
                sigma
            );

        p =
            std::sqrt(
                px * px
                + py * py
                + pz * pz
            );
    }
    else {

        component = 1;

        const double L =
            targetBroadScale(T);

        p =
            -L
            * std::log(
                safeUniform()
            );

        if (!(p > 0.0)) {
            p =
                std::numeric_limits<double>::min();
        }
    }

    gRadial =
        targetMixturePdf(
            p,
            T
        );
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
            << " <ee|ez> profile.csv output.root"
            << " [Nproposal] [seed] [R_km]\n";

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

    const std::string csvPath =
        argv[2];

    const std::string outPath =
        argv[3];

    const Long64_t nProposal =
        (argc > 4)
        ? std::stoll(argv[4])
        : 20000000LL;

    const int seed =
        (argc > 5)
        ? std::stoi(argv[5])
        : 20261013;

    const double R_km =
        (argc > 6)
        ? std::stod(argv[6])
        : 18.0;

    SNProfile profile;

    if (!profile.load(csvPath)) {

        std::cerr
            << "ERROR: cannot parse profile "
            << csvPath
            << "\n";

        return 3;
    }

    gRandom->SetSeed(seed);

    TFile fout(
        outPath.c_str(),
        "RECREATE"
    );

    if (fout.IsZombie()) {

        std::cerr
            << "ERROR: cannot create "
            << outPath
            << "\n";

        return 4;
    }

    fout.SetCompressionLevel(4);

    TTree tree(
        "events",
        "screened-bremsstrahlung-v1 full-support thermal proposal + exact ROOT 3-body PHSP"
    );

    // Store double precision four-vectors because the ee soft factor contains
    // cancellations among several large terms.
    double r_km = 0.0;
    double T = 0.0;
    double ne = 0.0;
    double mu = 0.0;
    double kappa = 0.0;

    double p1[4] = {0, 0, 0, 0};
    double p2[4] = {0, 0, 0, 0};

    double k1[4] = {0, 0, 0, 0};
    double k2[4] = {0, 0, 0, 0};
    double kh[4] = {0, 0, 0, 0};

    double g1_radial = 0.0;
    double g2_radial = 0.0;

    int component1 = -1;
    int component2 = -1;

    double w_tgen = 0.0;
    double w_phi3 = 0.0;

    tree.Branch(
        "r_km",
        &r_km,
        "r_km/D"
    );

    tree.Branch(
        "T",
        &T,
        "T/D"
    );

    tree.Branch(
        "ne",
        &ne,
        "ne/D"
    );

    tree.Branch(
        "mu",
        &mu,
        "mu/D"
    );

    tree.Branch(
        "kappa",
        &kappa,
        "kappa/D"
    );

    tree.Branch(
        "p1",
        p1,
        "p1[4]/D"
    );

    tree.Branch(
        "p2",
        p2,
        "p2[4]/D"
    );

    tree.Branch(
        "k1",
        k1,
        "k1[4]/D"
    );

    tree.Branch(
        "k2",
        k2,
        "k2[4]/D"
    );

    tree.Branch(
        "kh",
        kh,
        "kh[4]/D"
    );

    tree.Branch(
        "g1_radial",
        &g1_radial,
        "g1_radial/D"
    );

    tree.Branch(
        "g2_radial",
        &g2_radial,
        "g2_radial/D"
    );

    tree.Branch(
        "component1",
        &component1,
        "component1/I"
    );

    tree.Branch(
        "component2",
        &component2,
        "component2/I"
    );

    tree.Branch(
        "w_tgen",
        &w_tgen,
        "w_tgen/D"
    );

    tree.Branch(
        "w_phi3",
        &w_phi3,
        "w_phi3/D"
    );

    tree.SetAutoFlush(100000);

    TGenPhaseSpace decay;

    Long64_t nValid = 0;
    Long64_t nThresholdFail = 0;
    Long64_t nSetDecayFail = 0;

    Long64_t nComp10 = 0;
    Long64_t nComp11 = 0;
    Long64_t nComp20 = 0;
    Long64_t nComp21 = 0;

    double minMuOverT =
        1.0e100;

    double maxMuOverT =
        0.0;

    for (
        Long64_t i = 0;
        i < nProposal;
        ++i
    ) {

        // Uniform in physical volume.
        const double r =
            R_km
            * std::cbrt(
                gRandom->Rndm()
            );

        const ProfileRow pr =
            profile.eval(r);

        const double Tloc =
            pr.T_MeV
            * 1.0e-3;

        const double rho =
            pr.rho_gcm3
            * C::RHO_TO_GEV4;

        // Charge neutrality, with the existing SN profile convention.
        const double neloc =
            rho
            * pr.yp
            / C::M_TARGET;

        const double muloc =
            std::cbrt(
                3.0
                * C::PI
                * C::PI
                * neloc
            );

        // Same Debye momentum convention used in the existing framework:
        // kappa^2 = 8*pi*alpha*n_e/T for n_p ~= n_e.
        const double kappaloc =
            std::sqrt(
                8.0
                * C::PI
                * C::ALPHA
                * neloc
                / Tloc
            );

        if (Tloc > 0.0) {

            const double ratio =
                muloc / Tloc;

            minMuOverT =
                std::min(
                    minMuOverT,
                    ratio
                );

            maxMuOverT =
                std::max(
                    maxMuOverT,
                    ratio
                );
        }

        TLorentzVector vP1;
        TLorentzVector vP2;

        double pMag1 = 0.0;
        double pMag2 = 0.0;

        sampleElectronMomentum(
            muloc,
            Tloc,
            pMag1,
            g1_radial,
            component1
        );

        const TVector3 vec1 =
            isotropicVector(
                pMag1
            );

        // Frozen handoff convention: all external electrons are massless.
        vP1.SetPxPyPzE(
            vec1.X(),
            vec1.Y(),
            vec1.Z(),
            pMag1
        );

        if (component1 == 0) {
            ++nComp10;
        }
        else {
            ++nComp11;
        }

        double masses[3] = {0.0, 0.0, 0.0};

        if (isEE) {

            sampleElectronMomentum(
                muloc,
                Tloc,
                pMag2,
                g2_radial,
                component2
            );

            const TVector3 vec2 =
                isotropicVector(
                    pMag2
                );

            vP2.SetPxPyPzE(
                vec2.X(),
                vec2.Y(),
                vec2.Z(),
                pMag2
            );

            // e e -> e e h, all massless in this frozen benchmark.
            masses[0] = 0.0;
            masses[1] = 0.0;
            masses[2] = 0.0;
        }
        else {

            sampleTargetMomentum(
                Tloc,
                pMag2,
                g2_radial,
                component2
            );

            const TVector3 vec2 =
                isotropicVector(
                    pMag2
                );

            const double E2 =
                std::sqrt(
                    pMag2 * pMag2
                    + C::M_TARGET
                      * C::M_TARGET
                );

            vP2.SetPxPyPzE(
                vec2.X(),
                vec2.Y(),
                vec2.Z(),
                E2
            );

            // e Z -> e Z h.
            masses[0] = 0.0;
            masses[1] = C::M_TARGET;
            masses[2] = 0.0;
        }

        if (component2 == 0) {
            ++nComp20;
        }
        else {
            ++nComp21;
        }

        TLorentzVector P =
            vP1 + vP2;

        const double M =
            P.M();

        const double threshold =
            masses[0]
            + masses[1]
            + masses[2];

        if (
            !(M > threshold + 1.0e-12)
            || !std::isfinite(M)
        ) {

            ++nThresholdFail;
            continue;
        }

        if (!decay.SetDecay(
            P,
            3,
            masses
        )) {

            ++nSetDecayFail;
            continue;
        }

        w_tgen =
            decay.Generate();

        const TLorentzVector vK1 =
            *decay.GetDecay(0);

        const TLorentzVector vK2 =
            *decay.GetDecay(1);

        const TLorentzVector vH =
            *decay.GetDecay(2);

        w_phi3 =
            phi3ProposalWeight(
                P,
                vK1,
                vK2,
                masses[0],
                masses[1],
                masses[2]
            );

        if (
            !(w_phi3 >= 0.0)
            || !std::isfinite(w_phi3)
        ) {
            continue;
        }

        r_km = r;
        T = Tloc;
        ne = neloc;
        mu = muloc;
        kappa = kappaloc;

        pack4(vP1, p1);
        pack4(vP2, p2);

        pack4(vK1, k1);
        pack4(vK2, k2);
        pack4(vH, kh);

        tree.Fill();

        ++nValid;

        if ((i + 1) % 1000000 == 0) {

            std::cout
                << "\r"
                << channel
                << " proposals: "
                << (i + 1)
                << " / "
                << nProposal
                << std::flush;
        }
    }

    std::cout << "\n";

    tree.Write();

    TNamed(
        "CHANNEL",
        channel.c_str()
    ).Write();

    TNamed(
        "KERNEL_VERSION",
        "bare_amplitude_plus_squared_level_screening_20261013"
    ).Write();

    TNamed(
        "ELECTRON_KINEMATICS",
        "massless"
    ).Write();

    TNamed(
        "PROPOSAL_VERSION",
        "full_support_mixture_v1"
    ).Write();

    TParameter<Long64_t>(
        "N_PROPOSAL",
        nProposal
    ).Write();

    TParameter<Long64_t>(
        "N_VALID",
        nValid
    ).Write();

    TParameter<double>(
        "R_KM",
        R_km
    ).Write();

    TParameter<int>(
        "SEED",
        seed
    ).Write();

    TParameter<double>(
        "TARGET_MASS_GEV",
        C::M_TARGET
    ).Write();

    TParameter<double>(
        "TARGET_CHARGE_Z",
        C::TARGET_Z
    ).Write();

    TParameter<double>(
        "ELECTRON_BROAD_FRACTION",
        C::ELECTRON_BROAD_FRACTION
    ).Write();

    TParameter<double>(
        "TARGET_BROAD_FRACTION",
        C::TARGET_BROAD_FRACTION
    ).Write();

    fout.Close();

    std::cout
        << "=== "
        << channel
        << " SCREENED-BREM-V1 GENERATION SUMMARY ===\n"
        << "profile_rows="
        << profile.size()
        << "\n"
        << "N_PROPOSAL="
        << nProposal
        << "\n"
        << "N_VALID="
        << nValid
        << "\n"
        << "threshold_fail="
        << nThresholdFail
        << "\n"
        << "setDecay_fail="
        << nSetDecayFail
        << "\n"
        << "component1_shell="
        << nComp10
        << "\n"
        << "component1_broad="
        << nComp11
        << "\n"
        << "component2_core="
        << nComp20
        << "\n"
        << "component2_broad="
        << nComp21
        << "\n"
        << "min_mu_over_T="
        << minMuOverT
        << "\n"
        << "max_mu_over_T="
        << maxMuOverT
        << "\n"
        << "R_km="
        << R_km
        << "\n"
        << "seed="
        << seed
        << "\n"
        << "output="
        << outPath
        << "\n";

    return 0;
}
