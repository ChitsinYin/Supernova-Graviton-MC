#include "TFile.h"
#include "TGenPhaseSpace.h"
#include "TLorentzVector.h"
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
constexpr double ALPHA = 0.00729735;
constexpr double M_N = 0.9395;              // GeV
constexpr double RHO_GCM3_TO_GEV4 = 4.31e-18;
constexpr double A_MAP = 0.999999;
constexpr double PHOTON_TS_FACTOR = 10.0;
}

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
        if (!in) return false;

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::stringstream ss(line);
            ProfileRow row{};
            if (ss >> row.r_km >> row.rho_gcm3 >> row.yp >> row.T_MeV)
                rows_.push_back(row);
        }
        return rows_.size() >= 2;
    }

    ProfileRow eval(double r) const
    {
        if (r <= rows_.front().r_km) return rows_.front();
        if (r >= rows_.back().r_km) return rows_.back();

        auto it = std::lower_bound(
            rows_.begin(), rows_.end(), r,
            [](const ProfileRow& a, double x) { return a.r_km < x; });

        const auto& hi = *it;
        const auto& lo = *(it - 1);
        const double f = (r - lo.r_km)/(hi.r_km - lo.r_km);

        return {
            r,
            lo.rho_gcm3 + f*(hi.rho_gcm3 - lo.rho_gcm3),
            lo.yp + f*(hi.yp - lo.yp),
            lo.T_MeV + f*(hi.T_MeV - lo.T_MeV)
        };
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
    return {p*s*std::cos(phi), p*s*std::sin(phi), p*c};
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " profile.csv output.root [Nproposal] [seed] [R_km]\n";
        return 1;
    }

    const std::string profilePath = argv[1];
    const std::string outputPath = argv[2];
    const Long64_t nProposal = (argc > 3) ? std::stoll(argv[3]) : 5000000LL;
    const int seed = (argc > 4) ? std::stoi(argv[4]) : 20260910;
    const double R_km = (argc > 5) ? std::stod(argv[5]) : 18.0;

    SNProfile profile;
    if (!profile.load(profilePath)) {
        std::cerr << "ERROR: cannot parse profile: " << profilePath << "\n";
        return 2;
    }

    gRandom->SetSeed(seed);

    TFile fout(outputPath.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        std::cerr << "ERROR: cannot create " << outputPath << "\n";
        return 3;
    }
    fout.SetCompressionLevel(4);

    TTree tree("events", "gamma e -> h e thermal proposal + ROOT two-body PHSP");

    float r_km = 0.0f, T = 0.0f, ne = 0.0f, ngamma = 0.0f;
    float mu = 0.0f, kappa = 0.0f, pmax_e = 0.0f;
    TLorentzVector pe_in, pg_in, pe_out, h_out;

    tree.Branch("r_km", &r_km, "r_km/F");
    tree.Branch("T", &T, "T/F");
    tree.Branch("ne", &ne, "ne/F");
    tree.Branch("ngamma", &ngamma, "ngamma/F");
    tree.Branch("mu", &mu, "mu/F");
    tree.Branch("kappa", &kappa, "kappa/F");
    tree.Branch("pmax_e", &pmax_e, "pmax_e/F");
    tree.Branch("pe_in", &pe_in);
    tree.Branch("pg_in", &pg_in);
    tree.Branch("pe_out", &pe_out);
    tree.Branch("h_out", &h_out);
    tree.SetAutoFlush(100000);

    Long64_t nValid = 0;
    Long64_t nDegenerateS = 0;
    Long64_t nSetDecayFail = 0;

    for (Long64_t i = 0; i < nProposal; ++i) {
        const double r = R_km*std::cbrt(gRandom->Rndm());
        const ProfileRow pr = profile.eval(r);

        const double Tloc = pr.T_MeV*1.0e-3;
        const double rho = pr.rho_gcm3*C::RHO_GCM3_TO_GEV4;
        const double neloc = rho*pr.yp/C::M_N;
        const double ngloc = 2.0*1.202*std::pow(Tloc, 3)/(C::PI*C::PI);
        const double muloc = std::cbrt(3.0*C::PI*C::PI*neloc);
        const double kappaloc = std::sqrt(8.0*C::PI*C::ALPHA*neloc/Tloc);

        // Current validated photoproduction proposal: radial-uniform electron
        // sampling on [0, mu + 15 T], plus a broad exponential photon proposal.
        const double pmax = muloc + 15.0*Tloc;
        const double peMag = pmax*gRandom->Rndm();
        const TVector3 peVec = isotropicVector(peMag);
        pe_in.SetPxPyPzE(peVec.X(), peVec.Y(), peVec.Z(), peMag); // me = 0 benchmark

        const double Ts = C::PHOTON_TS_FACTOR*Tloc;
        const double u = gRandom->Rndm();
        const double pgMag = -Ts*std::log(1.0 - C::A_MAP*u);
        const TVector3 pgVec = isotropicVector(pgMag);
        pg_in.SetPxPyPzE(pgVec.X(), pgVec.Y(), pgVec.Z(), pgMag);

        TLorentzVector P = pe_in + pg_in;
        if (!(P.M2() > 1.0e-18) || !std::isfinite(P.M2())) {
            ++nDegenerateS;
            continue;
        }

        double masses[2] = {0.0, 0.0};
        TGenPhaseSpace decay;
        if (!decay.SetDecay(P, 2, masses)) {
            ++nSetDecayFail;
            continue;
        }

        decay.Generate();
        pe_out = *decay.GetDecay(0);
        h_out = *decay.GetDecay(1);

        r_km = static_cast<float>(r);
        T = static_cast<float>(Tloc);
        ne = static_cast<float>(neloc);
        ngamma = static_cast<float>(ngloc);
        mu = static_cast<float>(muloc);
        kappa = static_cast<float>(kappaloc);
        pmax_e = static_cast<float>(pmax);
        tree.Fill();
        ++nValid;

        if ((i + 1) % 500000 == 0)
            std::cout << "\rGenerated " << (i + 1) << " / " << nProposal << std::flush;
    }
    std::cout << "\n";

    tree.Write();
    TParameter<Long64_t>("N_PROPOSAL", nProposal).Write();
    TParameter<Long64_t>("N_VALID", nValid).Write();
    TParameter<double>("R_KM", R_km).Write();
    TParameter<int>("SEED", seed).Write();
    fout.Close();

    std::cout << "N_PROPOSAL=" << nProposal << "\n"
              << "N_VALID=" << nValid << "\n"
              << "degenerate_s=" << nDegenerateS << "\n"
              << "setDecay_fail=" << nSetDecayFail << "\n"
              << "output=" << outputPath << "\n";
    return 0;
}
