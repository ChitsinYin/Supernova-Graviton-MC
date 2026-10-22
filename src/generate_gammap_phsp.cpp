// generate_gammap_phsp.cpp
//
// ROOT/PHSP generator for proton photoproduction:
//     gamma + p -> h + p
//
// Initial photon and proton are sampled in the local SN plasma rest frame.
// The physical photon distribution uses the same Maxwell-Boltzmann
// thermal-shape convention as the existing gamma-e photoproduction
// calculation:
//     f_gamma(p) = exp(-p/T),
//     D_gamma = int d^3p f_gamma = 8*pi*T^3.
//
// For better high-energy coverage, photons are generated from a two-component
// importance proposal: 80% Gamma(k=3,T) + 20% Gamma(k=3,4T).  The physical
// distribution is restored exactly by a proposal-ratio weight.
// The proton is sampled from the non-relativistic Maxwell distribution.
//
// ROOT TGenPhaseSpace generates the exact two-body final-state kinematics.
// The stored ROOT file is a reusable kinematic PHSP pool; the matrix element
// and plasma screening are applied later in weight_gammap_spectrum.cpp.

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
#include <sstream>
#include <string>
#include <vector>

namespace C {

constexpr double PI =
    3.14159265358979323846;

constexpr double ALPHA =
    0.00729735;

constexpr double M_PROTON =
    0.9395;  // GeV; kept consistent with the existing SN framework

constexpr double RHO_TO_GEV4 =
    4.31e-18;

// Photon importance-sampling mixture.
//
// 80%: physical MB radial shape Gamma(k=3, scale=T)
// 20%: broader Gamma(k=3, scale=4T)
//
// The second component strongly enhances the high-energy thermal tail,
// while the exact physical distribution is restored by event reweighting.
constexpr double PHOTON_TAIL_FRACTION =
    0.20;

constexpr double PHOTON_TAIL_SCALE =
    4.0;

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
            + f * (hi.rho_gcm3 - lo.rho_gcm3);

        out.yp =
            lo.yp
            + f * (hi.yp - lo.yp);

        out.T_MeV =
            lo.T_MeV
            + f * (hi.T_MeV - lo.T_MeV);

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

// Gamma(k=3, scale=theta) radial PDF:
//
// q(p;theta) = p^2 exp(-p/theta)/(2 theta^3), p>0.
double gammaShape3Pdf(
    double p,
    double theta
)
{
    if (!(p > 0.0)
        || !(theta > 0.0)) {
        return 0.0;
    }

    return
        p * p
        * std::exp(-p / theta)
        /
        (
            2.0
            * theta
            * theta
            * theta
        );
}

double sampleGammaShape3(double theta)
{
    const double u1 = safeUniform();
    const double u2 = safeUniform();
    const double u3 = safeUniform();

    return
        -theta
        * std::log(u1 * u2 * u3);
}

// Photon importance proposal:
//
// g_gamma(p)
//   = (1-eps) q(p;T)
//     + eps q(p;aT),
//
// eps = 0.20, a = 4.
//
// The physical MB photon radial density remains q(p;T); therefore the
// weighting stage multiplies every event by
//
// R_gamma = q(p;T) / g_gamma(p).
//
// This changes only the Monte Carlo variance / tail coverage, not the
// physical integral.
void samplePhotonMomentum(
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
            - C::PHOTON_TAIL_FRACTION
        )
    ) {

        component = 0;

        p =
            sampleGammaShape3(T);
    }
    else {

        component = 1;

        p =
            sampleGammaShape3(
                C::PHOTON_TAIL_SCALE
                * T
            );
    }

    const double qNominal =
        gammaShape3Pdf(
            p,
            T
        );

    const double qBroad =
        gammaShape3Pdf(
            p,
            C::PHOTON_TAIL_SCALE
            * T
        );

    gRadial =
        (
            1.0
            - C::PHOTON_TAIL_FRACTION
        )
        * qNominal
        +
        C::PHOTON_TAIL_FRACTION
        * qBroad;
}

// Non-relativistic Maxwell proton in the plasma rest frame.
//
// Each Cartesian component is Gaussian with
// sigma^2 = m_p T.  This samples
//
// q_p(d^3p) = [2*pi*m_p*T]^(-3/2)
//             exp[-p^2/(2 m_p T)] d^3p,
//
// exactly matching the normalized NR proton thermal distribution used
// in the weight calculation.
TVector3 sampleProtonMomentum(double T)
{
    const double sigma =
        std::sqrt(
            C::M_PROTON * T
        );

    return TVector3(
        gRandom->Gaus(0.0, sigma),
        gRandom->Gaus(0.0, sigma),
        gRandom->Gaus(0.0, sigma)
    );
}

void pack4(
    const TLorentzVector& v,
    float out[4]
)
{
    out[0] =
        static_cast<float>(v.Px());

    out[1] =
        static_cast<float>(v.Py());

    out[2] =
        static_cast<float>(v.Pz());

    out[3] =
        static_cast<float>(v.E());
}

// Absolute two-body Lorentz-invariant PHSP weight divided by the
// isotropic angular proposal q(Omega)=1/(4*pi):
//
// dPhi_2 = p*/(16*pi^2*M) dOmega
//
// => dPhi_2 / q(Omega) = p*/(4*pi*M).
double phi2ProposalWeight(
    const TLorentzVector& parent,
    double m1,
    double m2
)
{
    const double M =
        parent.M();

    if (!(M > m1 + m2)) {
        return 0.0;
    }

    const double x =
        (M - m1 - m2)
        * (M + m1 + m2)
        * (M - m1 + m2)
        * (M + m1 - m2);

    if (!(x > 0.0)) {
        return 0.0;
    }

    const double pStar =
        std::sqrt(x)
        / (2.0 * M);

    return
        pStar
        /
        (4.0 * C::PI * M);
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
            << " profile.csv output.root"
            << " [Nproposal] [seed] [R_km]\n";

        return 1;
    }

    const std::string csvPath =
        argv[1];

    const std::string outPath =
        argv[2];

    const Long64_t nProposal =
        (argc > 3)
        ? std::stoll(argv[3])
        : 5000000LL;

    const int seed =
        (argc > 4)
        ? std::stoi(argv[4])
        : 20261007;

    const double R_km =
        (argc > 5)
        ? std::stod(argv[5])
        : 18.0;

    SNProfile profile;

    if (!profile.load(csvPath)) {

        std::cerr
            << "ERROR: cannot parse profile "
            << csvPath
            << "\n";

        return 2;
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

        return 3;
    }

    fout.SetCompressionLevel(4);

    TTree tree(
        "events",
        "gamma+p -> h+p ROOT PHSP pool"
    );

    float r_km = 0.0f;
    float T = 0.0f;
    float np = 0.0f;
    float kappa = 0.0f;

    float pGamma[4] = {0, 0, 0, 0};
    float pProton[4] = {0, 0, 0, 0};

    float kh[4] = {0, 0, 0, 0};
    float kProton[4] = {0, 0, 0, 0};

    double g_gamma_radial = 0.0;
    int gamma_component = -1;

    double w_tgen = 0.0;
    double w_phi2 = 0.0;

    tree.Branch(
        "r_km",
        &r_km,
        "r_km/F"
    );

    tree.Branch(
        "T",
        &T,
        "T/F"
    );

    tree.Branch(
        "np",
        &np,
        "np/F"
    );

    tree.Branch(
        "kappa",
        &kappa,
        "kappa/F"
    );

    tree.Branch(
        "pGamma",
        pGamma,
        "pGamma[4]/F"
    );

    tree.Branch(
        "pProton",
        pProton,
        "pProton[4]/F"
    );

    tree.Branch(
        "kh",
        kh,
        "kh[4]/F"
    );

    tree.Branch(
        "kProton",
        kProton,
        "kProton[4]/F"
    );

    tree.Branch(
        "g_gamma_radial",
        &g_gamma_radial,
        "g_gamma_radial/D"
    );

    tree.Branch(
        "gamma_component",
        &gamma_component,
        "gamma_component/I"
    );

    tree.Branch(
        "w_tgen",
        &w_tgen,
        "w_tgen/D"
    );

    tree.Branch(
        "w_phi2",
        &w_phi2,
        "w_phi2/D"
    );

    tree.SetAutoFlush(100000);

    TGenPhaseSpace decay;

    Long64_t nValid = 0;
    Long64_t nThresholdFail = 0;
    Long64_t nSetDecayFail = 0;

    Long64_t nPhotonNominal = 0;
    Long64_t nPhotonBroad = 0;

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
            pr.T_MeV * 1.0e-3;

        const double rhoGeV4 =
            pr.rho_gcm3
            * C::RHO_TO_GEV4;

        const double nploc =
            rhoGeV4
            * pr.yp
            / C::M_PROTON;

        // Debye screening:
        //
        // kappa^2 = 4*pi*alpha*(ne + np)/T,
        //
        // and charge neutrality gives ne ~= np.
        const double kappaloc =
            std::sqrt(
                8.0
                * C::PI
                * C::ALPHA
                * nploc
                / Tloc
            );

        // Incoming thermal photon in the plasma frame.
        double pGamMag = 0.0;

        samplePhotonMomentum(
            Tloc,
            pGamMag,
            g_gamma_radial,
            gamma_component
        );

        if (gamma_component == 0) {
            ++nPhotonNominal;
        }
        else {
            ++nPhotonBroad;
        }

        const TVector3 gamVec =
            isotropicVector(pGamMag);

        TLorentzVector vGamma;

        vGamma.SetPxPyPzE(
            gamVec.X(),
            gamVec.Y(),
            gamVec.Z(),
            pGamMag
        );

        // Incoming non-relativistic thermal proton in the plasma frame.
        const TVector3 pVec =
            sampleProtonMomentum(Tloc);

        const double pMag2 =
            pVec.Mag2();

        const double Ep =
            std::sqrt(
                C::M_PROTON
                * C::M_PROTON
                + pMag2
            );

        TLorentzVector vProton;

        vProton.SetPxPyPzE(
            pVec.X(),
            pVec.Y(),
            pVec.Z(),
            Ep
        );

        TLorentzVector P =
            vGamma + vProton;

        const double M =
            P.M();

        if (!(M > C::M_PROTON)) {

            ++nThresholdFail;
            continue;
        }

        // Daughter order:
        //   0 -> graviton h (massless)
        //   1 -> final proton
        double masses[2] = {
            0.0,
            C::M_PROTON
        };

        if (!decay.SetDecay(
            P,
            2,
            masses
        )) {

            ++nSetDecayFail;
            continue;
        }

        w_tgen =
            decay.Generate();

        const TLorentzVector vH =
            *decay.GetDecay(0);

        const TLorentzVector vPout =
            *decay.GetDecay(1);

        w_phi2 =
            phi2ProposalWeight(
                P,
                0.0,
                C::M_PROTON
            );

        if (!(w_phi2 > 0.0)
            || !std::isfinite(w_phi2)) {
            continue;
        }

        r_km =
            static_cast<float>(r);

        T =
            static_cast<float>(Tloc);

        np =
            static_cast<float>(nploc);

        kappa =
            static_cast<float>(kappaloc);

        pack4(
            vGamma,
            pGamma
        );

        pack4(
            vProton,
            pProton
        );

        pack4(
            vH,
            kh
        );

        pack4(
            vPout,
            kProton
        );

        tree.Fill();

        ++nValid;

        if ((i + 1) % 1000000 == 0) {

            std::cout
                << "\rGenerated "
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
        "gammap"
    ).Write();

    TNamed(
        "THERMAL_CONVENTION",
        "photon_MB_shape_plus_NR_Maxwell_proton"
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
        "PHOTON_TAIL_FRACTION",
        C::PHOTON_TAIL_FRACTION
    ).Write();

    TParameter<double>(
        "PHOTON_TAIL_SCALE",
        C::PHOTON_TAIL_SCALE
    ).Write();

    fout.Close();

    std::cout
        << "=== GAMMA-P PHSP GENERATION SUMMARY ===\n"
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
        << "photon_nominal_component="
        << nPhotonNominal
        << "\n"
        << "photon_broad_component="
        << nPhotonBroad
        << "\n"
        << "photon_tail_fraction="
        << C::PHOTON_TAIL_FRACTION
        << "\n"
        << "photon_tail_scale="
        << C::PHOTON_TAIL_SCALE
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
