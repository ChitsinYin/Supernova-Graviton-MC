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
constexpr double PI = 3.14159265358979323846;
constexpr double M_ELECTRON = 5.11e-4;      // GeV

// Electron proposal is a full-support core+tail approximation to
// p^2 f_-(p): a filled Fermi-sea core below mu and a thermal tail
// above mu.  The mixture fraction is chosen locally from the
// approximate core/tail normalizations.
}

struct ProfileRow {
    double r_km;
    double rho_gcm3;
    double yp;
    double T_MeV;
    double mu_old_GeV;
    double mu_precise_GeV;
};

class SNProfile {
public:
    bool load(const std::string& path)
    {
        std::ifstream in(path);
        if (!in) return false;

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;

            std::replace(line.begin(), line.end(), ',', ' ');
            for (char& c : line) {
                if (c == '"') c = ' ';
            }

            std::stringstream ss(line);
            ProfileRow row{};
            if (!(ss >> row.r_km
                     >> row.rho_gcm3
                     >> row.yp
                     >> row.T_MeV
                     >> row.mu_old_GeV
                     >> row.mu_precise_GeV)) {
                continue; // skips the CSV header
            }
            rows_.push_back(row);
        }

        return rows_.size() >= 2;
    }

    ProfileRow eval(double r) const
    {
        if (r <= rows_.front().r_km) return rows_.front();
        if (r >= rows_.back().r_km)  return rows_.back();

        auto it = std::lower_bound(
            rows_.begin(), rows_.end(), r,
            [](const ProfileRow& a, double x) {
                return a.r_km < x;
            });

        const auto& hi = *it;
        const auto& lo = *(it - 1);
        const double f =
            (r - lo.r_km)/(hi.r_km - lo.r_km);

        ProfileRow out{};
        out.r_km = r;
        out.rho_gcm3 = lo.rho_gcm3 + f*(hi.rho_gcm3 - lo.rho_gcm3);
        out.yp = lo.yp + f*(hi.yp - lo.yp);
        out.T_MeV = lo.T_MeV + f*(hi.T_MeV - lo.T_MeV);
        out.mu_old_GeV = lo.mu_old_GeV + f*(hi.mu_old_GeV - lo.mu_old_GeV);
        out.mu_precise_GeV = lo.mu_precise_GeV + f*(hi.mu_precise_GeV - lo.mu_precise_GeV);
        return out;
    }

    size_t size() const { return rows_.size(); }

private:
    std::vector<ProfileRow> rows_;
};

TVector3 isotropicVector(double p)
{
    const double c = 2.0*gRandom->Rndm() - 1.0;
    const double phi = 2.0*C::PI*gRandom->Rndm();
    const double s = std::sqrt(std::max(0.0, 1.0 - c*c));

    return TVector3(
        p*s*std::cos(phi),
        p*s*std::sin(phi),
        p*c
    );
}

// -Li_3(-exp(-eta)) = sum_{k>=1} (-1)^(k+1) exp(-k eta)/k^3.
// In the r <= 18 km profile eta=mu/T is comfortably positive, so the
// series converges rapidly and avoids a non-standard polylog dependency.
double minusLi3MinusExpNeg(double eta)
{
    const double z = std::exp(-eta);
    double zk = z;
    double sum = 0.0;

    for (int k = 1; k <= 80; ++k) {
        const double sign = (k % 2 == 1) ? 1.0 : -1.0;
        const double kd = static_cast<double>(k);
        const double term = sign*zk/(kd*kd*kd);
        sum += term;

        if (std::fabs(term) < 1.0e-18*std::max(1.0, std::fabs(sum))) {
            break;
        }
        zk *= z;
    }
    return sum;
}

// Massless finite-T positron density used in the analytical note:
// n_+ = -(2 T^3/pi^2) Li_3[-exp(-mu/T)].
double positronDensity(double T, double mu)
{
    if (!(T > 0.0) || !(mu >= 0.0)) return 0.0;
    const double eta = mu/T;
    return
        2.0*std::pow(T, 3)/(C::PI*C::PI)
        *minusLi3MinusExpNeg(eta);
}

double electronCorePdf(double p, double mu)
{
    if (!(mu > 0.0) || p < 0.0 || p > mu) return 0.0;
    return 3.0*p*p/std::pow(mu, 3);
}

double electronTailNorm(double mu, double T)
{
    // Integral_0^inf dx (mu+x)^2 exp(-x/T).
    return mu*mu*T + 2.0*mu*T*T + 2.0*std::pow(T, 3);
}

double electronTailPdf(double p, double mu, double T)
{
    if (p < mu || !(T > 0.0)) return 0.0;
    const double x = p - mu;
    const double z = electronTailNorm(mu, T);
    return p*p*std::exp(-x/T)/z;
}

double electronTailFraction(double mu, double T)
{
    const double zCore = std::pow(mu, 3)/3.0;
    const double zTail = electronTailNorm(mu, T);
    return zTail/(zCore + zTail);
}

double electronProposalPdf(double p, double mu, double T)
{
    const double eps = electronTailFraction(mu, T);
    if (p <= mu) {
        return (1.0 - eps)*electronCorePdf(p, mu);
    }
    return eps*electronTailPdf(p, mu, T);
}

// Sample x=p-mu from the normalized tail
// (mu+x)^2 exp(-x/T).  Expanding the polynomial gives a mixture
// of Gamma(k=1,2,3; theta=T) distributions.
double sampleElectronTailX(double mu, double T)
{
    const double z1 = mu*mu*T;
    const double z2 = 2.0*mu*T*T;
    const double z3 = 2.0*std::pow(T, 3);
    const double z = z1 + z2 + z3;

    const double u = gRandom->Rndm();
    int k = 3;
    if (u < z1/z) k = 1;
    else if (u < (z1 + z2)/z) k = 2;

    double x = 0.0;
    for (int j = 0; j < k; ++j) {
        double uj = std::max(gRandom->Rndm(), 1.0e-15);
        x += -T*std::log(uj);
    }
    return x;
}

void sampleElectronMomentum(
    double mu,
    double T,
    double& p,
    double& gRadial,
    int& component)
{
    const double eps = electronTailFraction(mu, T);

    if (gRandom->Rndm() < (1.0 - eps)) {
        component = 0;
        p = mu*std::cbrt(gRandom->Rndm());
    } else {
        component = 1;
        p = mu + sampleElectronTailX(mu, T);
    }

    gRadial = electronProposalPdf(p, mu, T);
}

// For a dilute positron gas, the radial physical density is close to
// p^2 exp(-p/T).  A Gamma(k=3, theta=T) proposal is therefore efficient
// and already has support over the full positive momentum axis.
double positronProposalPdf(double p, double T)
{
    if (p < 0.0 || !(T > 0.0)) return 0.0;
    return p*p*std::exp(-p/T)/(2.0*std::pow(T, 3));
}

void samplePositronMomentum(
    double T,
    double& p,
    double& gRadial)
{
    double u1 = std::max(gRandom->Rndm(), 1.0e-15);
    double u2 = std::max(gRandom->Rndm(), 1.0e-15);
    double u3 = std::max(gRandom->Rndm(), 1.0e-15);

    p = -T*(std::log(u1) + std::log(u2) + std::log(u3));
    gRadial = positronProposalPdf(p, T);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " profile.csv output.root [Nproposal] [seed] [R_km]\n";
        return 1;
    }

    const std::string csvPath = argv[1];
    const std::string outPath = argv[2];

    const Long64_t nProposal =
        (argc > 3) ? std::stoll(argv[3]) : 5000000LL;

    const int seed =
        (argc > 4) ? std::stoi(argv[4]) : 20261006;

    const double R_km =
        (argc > 5) ? std::stod(argv[5]) : 18.0;

    SNProfile profile;
    if (!profile.load(csvPath)) {
        std::cerr << "ERROR: cannot parse profile: " << csvPath << "\n";
        return 2;
    }

    gRandom->SetSeed(seed);

    TFile fout(outPath.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        std::cerr << "ERROR: cannot create " << outPath << "\n";
        return 3;
    }
    fout.SetCompressionLevel(4);

    TTree tree(
        "events",
        "e- e+ -> h gamma thermal initial states + exact two-body PHSP"
    );

    double r_km = 0.0;
    double T = 0.0;
    double np = 0.0;
    double nplus = 0.0;
    double nminus_fd = 0.0;
    double mu = 0.0;

    TLorentzVector em_in;
    TLorentzVector ep_in;
    TLorentzVector h_out;
    TLorentzVector gamma_out;

    double gminus_radial = 0.0;
    double gplus_radial = 0.0;
    int em_component = -1;

    tree.Branch("r_km", &r_km, "r_km/D");
    tree.Branch("T", &T, "T/D");
    tree.Branch("np", &np, "np/D");
    tree.Branch("nplus", &nplus, "nplus/D");
    tree.Branch("nminus_fd", &nminus_fd, "nminus_fd/D");
    tree.Branch("mu", &mu, "mu/D");

    tree.Branch("em_in", &em_in);
    tree.Branch("ep_in", &ep_in);
    tree.Branch("h_out", &h_out);
    tree.Branch("gamma_out", &gamma_out);

    tree.Branch("gminus_radial", &gminus_radial, "gminus_radial/D");
    tree.Branch("gplus_radial", &gplus_radial, "gplus_radial/D");
    tree.Branch("em_component", &em_component, "em_component/I");

    tree.SetAutoFlush(100000);

    TGenPhaseSpace decay;

    Long64_t nValid = 0;
    Long64_t nDegenerateS = 0;
    Long64_t nSetDecayFail = 0;
    Long64_t nElectronCore = 0;
    Long64_t nElectronTail = 0;

    double minMuOverT = 1.0e100;
    double maxMuOverT = 0.0;
    double maxNplusOverNp = 0.0;

    for (Long64_t i = 0; i < nProposal; ++i) {
        // 1. Radius: uniform in physical volume.
        const double r = R_km*std::cbrt(gRandom->Rndm());
        const ProfileRow pr = profile.eval(r);

        const double Tloc = pr.T_MeV*1.0e-3;
        // The dedicated profile supplies both:
        //   mu_old_GeV     = p_F from the T -> 0 neutrality relation,
        //   mu_precise_GeV = finite-T charge-neutrality solution.
        // Recover n_p directly from p_F so the density normalization is
        // exactly consistent with the supplied chemical-potential columns:
        //   n_p = p_F^3/(3 pi^2).
        const double nploc =
            std::pow(pr.mu_old_GeV, 3)/(3.0*C::PI*C::PI);

        const double muloc = pr.mu_precise_GeV;

        if (!(Tloc > 0.0) || !(muloc > 0.0) || !(nploc > 0.0)) {
            continue;
        }

        const double nplusloc = positronDensity(Tloc, muloc);
        const double nminusFDloc = nploc + nplusloc;

        const double muOverT = muloc/Tloc;
        minMuOverT = std::min(minMuOverT, muOverT);
        maxMuOverT = std::max(maxMuOverT, muOverT);
        maxNplusOverNp = std::max(maxNplusOverNp, nplusloc/nploc);

        // 2. Incoming electron proposal in the plasma rest frame.
        double pem = 0.0;
        sampleElectronMomentum(
            muloc, Tloc, pem, gminus_radial, em_component
        );

        if (em_component == 0) ++nElectronCore;
        else ++nElectronTail;

        const TVector3 vem = isotropicVector(pem);
        const double Eem = std::sqrt(pem*pem + C::M_ELECTRON*C::M_ELECTRON);
        em_in.SetPxPyPzE(vem.X(), vem.Y(), vem.Z(), Eem);

        // 3. Incoming positron proposal in the plasma rest frame.
        double pep = 0.0;
        samplePositronMomentum(Tloc, pep, gplus_radial);

        const TVector3 vep = isotropicVector(pep);
        const double Eep = std::sqrt(pep*pep + C::M_ELECTRON*C::M_ELECTRON);
        ep_in.SetPxPyPzE(vep.X(), vep.Y(), vep.Z(), Eep);

        // 4. Exact incoming pair four-momentum.
        TLorentzVector P = em_in + ep_in;
        const double s = P.M2();

        if (!(s > 0.0) || !std::isfinite(s)) {
            ++nDegenerateS;
            continue;
        }

        // 5. Exact two-body final-state PHSP: P -> h + gamma.
        double masses[2] = {0.0, 0.0};
        if (!decay.SetDecay(P, 2, masses)) {
            ++nSetDecayFail;
            continue;
        }

        decay.Generate();
        h_out = *decay.GetDecay(0);
        gamma_out = *decay.GetDecay(1);

        // 6. Store a reusable kinematic integration point.
        r_km = r;
        T = Tloc;
        np = nploc;
        nplus = nplusloc;
        nminus_fd = nminusFDloc;
        mu = muloc;

        tree.Fill();
        ++nValid;

        if ((i + 1) % 500000 == 0) {
            std::cout
                << "\rGenerated proposals: "
                << (i + 1) << " / " << nProposal
                << std::flush;
        }
    }

    std::cout << "\n";

    tree.Write();

    TNamed("CHANNEL", "e- e+ -> h gamma").Write();
    TNamed("CHEMICAL_POTENTIAL", "mu_precise_GeV from input profile").Write();

    TParameter<Long64_t>("N_PROPOSAL", nProposal).Write();
    TParameter<Long64_t>("N_VALID", nValid).Write();
    TParameter<double>("R_KM", R_km).Write();
    TParameter<int>("SEED", seed).Write();

    fout.Close();

    std::cout
        << "=== e- e+ ANNIHILATION PHSP GENERATION SUMMARY ===\n"
        << "profile_rows=" << profile.size() << "\n"
        << "N_PROPOSAL=" << nProposal << "\n"
        << "N_VALID=" << nValid << "\n"
        << "degenerate_s=" << nDegenerateS << "\n"
        << "setDecay_fail=" << nSetDecayFail << "\n"
        << "electron_core_proposals=" << nElectronCore << "\n"
        << "electron_tail_proposals=" << nElectronTail << "\n"
        << "min_mu_over_T=" << minMuOverT << "\n"
        << "max_mu_over_T=" << maxMuOverT << "\n"
        << "max_nplus_over_np=" << maxNplusOverNp << "\n"
        << "R_km=" << R_km << "\n"
        << "seed=" << seed << "\n"
        << "output=" << outPath << "\n";

    return 0;
}
