#ifndef TOPOPT_HPP
#define TOPOPT_HPP
#include "mfem.hpp"
#include "helper.hpp"
#include <functional>


namespace mfem
{
inline double double_exp(const double x) {return std::exp(x);}

inline double safe_log(const double x)
{
   return x < 1e-300 ? -300*std::log(10) : std::log(x);
}

/// @brief Inverse sigmoid function
inline double inv_sigmoid(const double x)
{
   return x < 0.5 ? safe_log(x / (1.0 - x)) : -safe_log((1.0 - x) / x);
}

/// @brief Sigmoid function
inline double sigmoid(const double x)
{
   return x>=0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}

inline double safe_logsigmoid(const double x)
{
   if (std::fabs(x) < 5) { return std::log(sigmoid(x)); }
   else if (x > 0) {return -std::log1p(std::exp(-x));}
   else {return x-std::log1p(std::exp(x));};
}

/// @brief Derivative of sigmoid function
inline double der_sigmoid(const double x)
{
   const double tmp = sigmoid(x);
   return tmp*(1.0 - tmp);
}

/// @brief SIMP function, ρ₀ + (ρ̄ - ρ₀)*x^k
inline double simp(const double x, const double rho_0, const double k,
                   const double rho_max=1.0)
{
   return rho_0 + std::pow(x, k) * (rho_max - rho_0);
}

/// @brief Derivative of SIMP function, k*(ρ̄ - ρ₀)*x^(k-1)
inline double der_simp(const double x, const double rho_0,
                       const double k, const double rho_max=1.0)
{
   return k * std::pow(x, k - 1.0) * (rho_max - rho_0);
}

/// @brief Derivative of SIMP function, k*(ρ̄ - ρ₀)*x^(k-1)
inline double der2_simp(const double x, const double rho_0,
                        const double k, const double rho_max=1.0)
{
   return k * (k - 1.0) * std::pow(x, k - 2.0) * (rho_max - rho_0);
}

void ProjectCoefficient_attr(GridFunction &gf, Coefficient &coeff,
                             int attribute);

class LegendreFunction
{
private:
   std::function<double(const double)> f;
   std::function<double(const double)> df;
   std::function<double(const double)> inv_df;
protected:
public:
   double operator()(const double x) { return f(x); }
   std::function<double(const double)> &GetForwardMap() { return df; }
   std::function<double(const double)> &GetInverseMap() { return inv_df; }
   void SetLegendre(std::function<double(const double)> h) { f = h; }
   void SetForwardMap(std::function<double(const double)> p2l) { df = p2l; }
   void SetInverseMap(std::function<double(const double)> l2p) { inv_df = l2p; }
};

// Fermi-Dirac function, xlog(x) + (1-x)log(1-x)
class FermiDirac: public LegendreFunction
{
public:
   // Fermi-Dirac function with safe logarithm
   FermiDirac():LegendreFunction()
   {
      // Fermi-Dirac
      SetLegendre([](const double x)
      {
         return x*safe_log(x) + (1.0 - x)*safe_log(1.0 - x);
      });
      // Derivative, logit = log(x/(1-x))
      SetForwardMap(inv_sigmoid);
      // Inverse of Derivative, sigmoid
      SetInverseMap(sigmoid);
   }
};

// Shannon entropy, xlog(x) - x
class Shannon : public LegendreFunction
{
public:
   // Shannon entropy, xlog(x) - x
   Shannon():LegendreFunction()
   {
      // Shannon entropy
      SetLegendre([](const double x)
      {
         return x*safe_log(x) - x;
      });
      // Derivative, safe log(x)
      SetForwardMap(safe_log);
      // Inverse of Derivative, exp
      SetInverseMap(double_exp);
   }
};

inline double FermiDiracEntropy(const double x)
{
   const double y = 1.0-x;
   return (x < 1e-13 ? 0 : x*std::log(x))
          + (y < 1e-13 ? 0 : y*std::log(y));
}

inline double ShannonEntropy(const double x)
{
   return x < 1e-13 ? -x : x*std::log(x) - x;
}
// Gradient to symmetric gradient in Voigt notation
/** Integrator for the isotropic linear elasticity form:
    a(u,v) = (Cu : ϵ(v))
    where ϵ(v) = (1/2) (grad(v) + grad(v)^T).
    This is a 'Vector' integrator, i.e. defined for FE spaces
    using multiple copies of a scalar FE space. */
class IsoElasticityIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *E, *nu;
   int ia;
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape; // scalar gradient
   DenseMatrix vshape; // Voigt notation of symmetric gradient
   DenseMatrix C; // elasticity matrix in Voigt notation
   DenseMatrix CVt;
   bool enforce_symmetricity;
#endif
   void VectorGradToVoigt(DenseMatrix &vals, DenseMatrix &voigt);

public:
   IsoElasticityIntegrator(Coefficient &E, Coefficient &nu, int ia=3): E(&E),
      nu(&nu), enforce_symmetricity(false) {}

   void EnforceSymmetricity(bool flag) { enforce_symmetricity = flag; }

   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

class ForcedSymmetricDiffusionIntegrator : public DiffusionIntegrator
{
private:
   bool enforce_symmetricity;
public:
   using DiffusionIntegrator::DiffusionIntegrator;
   void EnforceSymmetricity(bool flag) { enforce_symmetricity = flag; }
   void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Tr,
                              DenseMatrix &elmat) override
   {
      DiffusionIntegrator::AssembleElementMatrix(el, Tr, elmat);
      elmat.Symmetrize();
   }
};

class ForcedSymmetricMassIntegrator : public MassIntegrator
{
private:
   bool enforce_symmetricity;
public:
   using MassIntegrator::MassIntegrator;
   void EnforceSymmetricity(bool flag) { enforce_symmetricity = flag; }
   void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Tr,
                              DenseMatrix &elmat) override
   {
      MassIntegrator::AssembleElementMatrix(el, Tr, elmat);
      elmat.Symmetrize();
   }
};

class L2ProjectionLFIntegrator : public LinearFormIntegrator
{
private:
   const GridFunction *gf;
   Vector gf_val, Mv;
   ForcedSymmetricMassIntegrator mass_loc;
   MixedScalarMassIntegrator mass;
   DenseMatrix M;
   DenseMatrix Minv;
public:
   L2ProjectionLFIntegrator(const GridFunction &gf): LinearFormIntegrator(),
      gf(&gf), mass_loc(), mass() {}
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      const FiniteElement &el_source = *(gf->FESpace()->GetFE(Tr.ElementNo));
      gf->GetElementDofValues(Tr.ElementNo, gf_val);
      mass.AssembleElementMatrix2(el_source, el, Tr, M);
      const int dof = M.NumCols();

      Mv.SetSize(dof);
      M.Mult(gf_val, Mv);

      Minv.SetSize(dof);
      mass_loc.AssembleElementMatrix(el, Tr, Minv);
      Minv.Invert();
      Minv.Symmetrize();

      elvect.SetSize(dof);
      Minv.Mult(Mv, elvect);
   }
   void SetGridFunction(const GridFunction &new_gf) {gf = &new_gf; }
};
class VectorBdrMassIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient &k;
   const int vdim;
   const int oa, ob;
public:
   VectorBdrMassIntegrator(
      Coefficient &k, const int vdim, const int oa=2, const int ob=0):
      BilinearFormIntegrator(NULL), k(k), vdim(vdim), oa(oa), ob(ob) {}
   VectorBdrMassIntegrator(
      Coefficient &k, const int vdim, const IntegrationRule *ir):
      BilinearFormIntegrator(ir), k(k), vdim(vdim), oa(0), ob(0) {}

   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};
class VectorBdrDirectionalMassIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient &k;
   VectorCoefficient &d;
   const int vdim;
   const int oa, ob;
public:
   VectorBdrDirectionalMassIntegrator(
      Coefficient &k, VectorCoefficient &d, const int vdim, const int oa=2,
      const int ob=0):
      BilinearFormIntegrator(NULL), k(k), d(d), vdim(vdim), oa(oa), ob(ob) {}
   VectorBdrDirectionalMassIntegrator(
      Coefficient &k, VectorCoefficient &d, const int vdim,
      const IntegrationRule *ir):
      BilinearFormIntegrator(ir), k(k), d(d), vdim(vdim), oa(0), ob(0) {}

   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

// Elliptic Bilinear Solver
class EllipticSolver
{
protected:
   BilinearForm &a; // LHS
   LinearForm &b; // RHS
   Array2D<int> ess_bdr; // Component-wise essential boundary marker
   Array<int> ess_tdof_list;
   bool symmetric;
   bool iterative_mode;
   int max_it;
   bool elasticity; // use elasticity HypreAMG
#ifdef MFEM_USE_MPI
   bool parallel; // Flag for ParFiniteElementSpace
   MPI_Comm comm;
#endif
public:
   /// @brief Linear solver for elliptic problem with given Essential BC
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Essential boundary marker for boundary attributes]
   EllipticSolver(BilinearForm &a, LinearForm &b, Array<int> &ess_bdr_);
   /// @brief Linear solver for elliptic problem with given component-wise essential BC
   /// ess_bdr[0,:] - All components, ess_bdr[i,:] - ith-direction
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Component-wise essential boundary marker for boundary attributes, [Row0: all, Row1: x, ...]
   EllipticSolver(BilinearForm &a, LinearForm &b, Array2D<int> &ess_bdr);
   void SetMaxIt(int new_max_it) { max_it = new_max_it; }
   void UseElasticityOptions() { elasticity = true; }

   /// @brief Solve linear system and return FEM solution in x.
   /// @param x FEM solution
   /// @param A_assembled If true, skip assembly of LHS (bilinearform)
   /// @param b_Assembled If true, skip assembly of RHS (linearform)
   /// @return convergence flag
   bool Solve(GridFunction &x, bool A_assembled=false, bool b_Assembled=false);
#ifdef MFEM_USE_MPI
   bool isParallel() { return parallel; }
#endif
   bool isSymmetric() { return symmetric; }
   void SetIterativeMode(bool flag=true) {iterative_mode = flag;};
protected:
   /// @brief Get true dofs related to the boundaries in @ess_bdr
   /// @return True dof list
   void GetEssentialTrueDofs();
private:
};

class DensityFilter
{
public:
protected:
   FiniteElementSpace &fes;
private:

public:
   DensityFilter(FiniteElementSpace &fes):fes(fes) {};
   virtual void Apply(const GridFunction &rho, GridFunction &frho,
                      bool apply_bdr=true) = 0;
   virtual void Apply(Coefficient &rho, GridFunction &frho,
                      bool apply_bdr=true) = 0;
   FiniteElementSpace &GetFESpace() {return fes;};
protected:
private:
};

class HelmholtzFilter : public DensityFilter
{
public:
protected:
   std::unique_ptr<BilinearForm> filter;
   std::unique_ptr<LinearForm> rhoform;
   Array<int> ess_bdr;
   Array<int> material_bdr;
   Array<int> void_bdr;
   ConstantCoefficient eps2;
   ConstantCoefficient bdr_eps;
private:

public:
   HelmholtzFilter(FiniteElementSpace &fes, const double eps, Array<int> &ess_bdr,
                   bool enforce_symmetricity=false);
   void Apply(const GridFunction &rho, GridFunction &frho,
              bool apply_bdr=true) override
   {
      GridFunctionCoefficient rho_cf(&rho);
      Apply(rho_cf, frho, apply_bdr);
   }
   void Apply(Coefficient &rho, GridFunction &frho,
              bool apply_bdr=true) override;
   BilinearForm& GetBilinearForm() {return *filter; }
protected:
private:
};

class HelmholtzL2Filter : public DensityFilter
{
public:
protected:
   HelmholtzFilter filter;
   std::unique_ptr<GridFunction> H1frho;
private:

public:
   HelmholtzL2Filter(FiniteElementSpace &fes_, FiniteElementSpace &rho_fes,
                     const double eps,
                     Array<int> &ess_bdr,
                     bool enforce_symmetricity=false):
      DensityFilter(rho_fes), filter(fes_, eps, ess_bdr, enforce_symmetricity),
      H1frho(MakeGridFunction(&fes_))
   {
      *H1frho = 0.0;
   }
   void Apply(const GridFunction &rho, GridFunction &frho,
              bool apply_bdr=true) override
   {
      GridFunctionCoefficient rho_cf(&rho);
      Apply(rho_cf, frho, apply_bdr);
   }
   void Apply(Coefficient &rho, GridFunction &frho,
              bool apply_bdr=true) override;
protected:
private:
};

class DesignDensity
{
   // variables
public:
protected:
   std::unique_ptr<GridFunction> x_gf;
   std::unique_ptr<GridFunction> tmp_gf;
   std::unique_ptr<GridFunction> frho;
   std::unique_ptr<Coefficient> rho_cf;
   DensityFilter &filter;
   double target_volume_fraction;
   double target_volume;
   double current_volume;
   double domain_volume;
   double vol_tol;
   int vol_constraint; // 1: max (default), 0: equal, -1: min
private:

   // functions
public:
   DesignDensity(FiniteElementSpace &fes, DensityFilter &filter,
                 double vol_frac, double volume_tolerance=1e-09);
   FiniteElementSpace *FESpace() {return x_gf->FESpace(); }
   FiniteElementSpace *FESpace_filter() {return frho->FESpace(); }
   double GetVolume() { return current_volume; }
   double GetDomainVolume() { return domain_volume; }
   GridFunction &GetGridFunction() { return *x_gf; }
   Coefficient &GetDensityCoefficient() { return *rho_cf; }
   GridFunction &GetFilteredDensity() { return *frho; }
   virtual void UpdateFilteredDensity()
   {
      filter.Apply(*rho_cf, *frho);
   }
   DensityFilter &GetFilter() { return filter; }
   // Set volume constraint, 1: max (default), 0: equal, -1: min
   void SetVolumeConstraintType(int constraint_type) {vol_constraint = constraint_type;}
   bool VolumeConstraintViolated();

   virtual double Project() = 0;
   virtual double StationarityError(const GridFunction &grad,
                                    const double eps) = 0;
   virtual double ComputeVolume() = 0;
   virtual std::unique_ptr<Coefficient> GetDensityDiffCoeff(
      GridFunction &other_gf) = 0;
protected:
private:
};

class DensityProjector
{
public:
   virtual Coefficient &GetPhysicalDensity(GridFunction &frho) = 0;
   virtual Coefficient &GetDerivative(GridFunction &frho) = 0;
};
class SIMPProjector : public DensityProjector
{
private:
   double k, rho0;
   std::unique_ptr<MappedGridFunctionCoefficient> phys_density, dphys_dfrho;
public:
   SIMPProjector(const double k, const double rho0);
   void SetPower(int new_k) {k = new_k;}
   void SetMin(double new_rho0) {rho0 = new_rho0;}
   Coefficient &GetPhysicalDensity(GridFunction &frho) override;
   Coefficient &GetDerivative(GridFunction &frho) override;
};

class ThresholdProjector : public DensityProjector
{
private:
   double beta, eta, rho0, k;
   std::unique_ptr<MappedGridFunctionCoefficient> phys_density, dphys_dfrho;
public:
   ThresholdProjector(const double beta, const double eta, const double k,
                      const double rho0);
   void SetThresholdParameters(double new_beta, double new_eta) {beta = new_beta; eta = new_eta; }
   void SetPower(int new_k) {k = new_k;}
   void SetMin(double new_rho0) {rho0 = new_rho0;}
   Coefficient &GetPhysicalDensity(GridFunction &frho) override;
   Coefficient &GetDerivative(GridFunction &frho) override;
};


class LatentDesignDensity : public DesignDensity
{
   // variables
public:
protected:
   std::unique_ptr<GridFunction> zero_gf;
private:
   std::function<double(double)> h;
   std::function<double(double)> p2d;
   std::function<double(double)> d2p;
   bool clip_lower, clip_upper;
   bool use_primal_filter;
   std::unique_ptr<Coefficient>
   post_project_clip; // if there are passive elements, project using this.
   // functions
public:
   LatentDesignDensity(FiniteElementSpace &fes,
                       DensityFilter &filter, double vol_frac,
                       std::function<double(double)> h,
                       std::function<double(double)> primal2dual,
                       std::function<double(double)> dual2primal,
                       bool clip_lower=false, bool clip_upper=false);
   void SetPostProjectClip(Coefficient *cf) {post_project_clip.reset(cf);}
   double Project() override;
   double StationarityError(const GridFunction &grad,
                            const double eps=1e-04) override
   {
      return StationarityError(grad, false, eps);
   };
   double StationarityError(const GridFunction &grad, bool useL2norm,
                            const double eps=1e-04);
   double StationarityErrorL2(GridFunction &grad, const double eps=1e-04);
   virtual double ComputeBregmanDivergence(const GridFunction &p,
                                           const GridFunction &q);
   double ComputeVolume() override
   {
      current_volume = zero_gf->ComputeL1Error(*rho_cf);
      return current_volume;
   }
   std::unique_ptr<Coefficient> GetDensityDiffCoeff(GridFunction &other_gf)
   override
   {
      return std::unique_ptr<Coefficient>(new MappedPairGridFunctionCoeffitient(
                                             x_gf.get(), &other_gf,
      [this](double x, double y) {return d2p(x) - d2p(y);}));
   }
   void UsePrimalFilter(bool flag) {use_primal_filter = flag;}
   bool UsingPrimalFilter() {return use_primal_filter;}
   void UpdateFilteredDensity() override
   {
      if (use_primal_filter)
      {
         filter.Apply(*rho_cf, *frho);
      }
      else
      {
         filter.Apply(*x_gf, *frho);
         frho->ApplyMap(d2p);
      }
   }
protected:
private:
};

class FermiDiracDesignDensity : public LatentDesignDensity
{
public:
   FermiDiracDesignDensity(FiniteElementSpace &fes,
                           DensityFilter &filter, double vol_frac,
                           std::function<double(double)> h,
                           std::function<double(double)> primal2dual,
                           std::function<double(double)> dual2primal,
                           bool clip_lower=false, bool clip_upper=false):
      LatentDesignDensity(fes, filter, vol_frac, h, primal2dual, dual2primal,
                          clip_lower, clip_upper) {};
   double ComputeBregmanDivergence(const GridFunction &p,
                                   const GridFunction &q) override;
};

class PrimalDesignDensity : public DesignDensity
{
   // variables
public:
protected:
private:
   std::unique_ptr<GridFunction> zero_gf;
   // functions
public:
   PrimalDesignDensity(FiniteElementSpace &fes, DensityFilter& filter,
                       double vol_frac);
   double Project() override;
   double StationarityError(const GridFunction &grad,
                            const double eps=1e-03) override;
   double ComputeVolume() override
   {
      current_volume = zero_gf->ComputeL1Error(*rho_cf);
      return current_volume;
   }
   std::unique_ptr<Coefficient> GetDensityDiffCoeff(GridFunction &other_gf)
   override
   {
      return std::unique_ptr<Coefficient>(new MappedPairGridFunctionCoeffitient(
                                             x_gf.get(), &other_gf,
      [](double x, double y) {return x - y;}));
   }
protected:
private:
};

class ParametrizedLinearEquation
{
public:
protected:
   std::unique_ptr<BilinearForm> a;
   std::unique_ptr<LinearForm> b;
   GridFunction &frho;
   DensityProjector &projector;
   bool AisStationary;
   bool BisStationary;
   Array2D<int> &ess_bdr;
private:


public:
   ParametrizedLinearEquation(FiniteElementSpace &fes,
                              GridFunction &filtered_density,
                              DensityProjector &projector, Array2D<int> &ess_bdr);
   void SetBilinearFormStationary(bool isStationary=true);
   void SetLinearFormStationary(bool isStationary=true);
   void Solve(GridFunction &x);

   /// @brief Solve a(λ, v)=F(v) assuming a is symmetric.
   /// @param
   /// @param u
   /// @return
   void DualSolve(GridFunction &x, LinearForm &new_b);
   virtual std::unique_ptr<Coefficient> GetdEdfrho(GridFunction &u,
                                                   GridFunction &lambda, GridFunction &frho) = 0;
   FiniteElementSpace *FESpace() { return a->FESpace(); }
   BilinearForm &GetBilinearForm() {return *a;}
   LinearForm &GetLinearForm() {return *b;}
protected:
   virtual void SolveSystem(GridFunction &x) = 0;
private:
};

class TopOptProblem
{
public:
protected:
   LinearForm &obj;
   ParametrizedLinearEquation &state_equation;
   DesignDensity &density;
   std::shared_ptr<GridFunction> gradF;
   std::shared_ptr<GridFunction> gradF_filter;
   std::shared_ptr<GridFunction> state, dual_solution;
   L2ProjectionLFIntegrator *L2projector; // not owned
   std::unique_ptr<LinearForm> filter_to_density;
   std::unique_ptr<Coefficient> dEdfrho;
   const bool solve_dual;
   const bool apply_projection;
   double vol_lagrange;
   double val;
private:
#ifdef MFEM_USE_MPI
   bool parallel;
   MPI_Comm comm;
#endif

public:

   /// @brief Create Topology optimization problem
   /// @param objective Objective linear functional, F(u)
   /// @param state_equation State equation, a(u,v) = b(v)
   /// @param density Density object, ρ
   /// @param solve_dual If true, kip dual solve, a(v,λ)=F(v) and assume λ=u
   /// @note It assume that the state equation is symmetric and objective
   TopOptProblem(LinearForm &objective,
                 ParametrizedLinearEquation &state_equation,
                 DesignDensity &density, bool solve_dual, bool apply_projection);

   double Eval();
   double GetValue() {return val;}
   void UpdateGradient();
   GridFunction &GetGradient() { return *gradF; }
   GridFunction &GetGridFunction() { return density.GetGridFunction(); }
   Coefficient &GetDensity() { return density.GetDensityCoefficient(); }
   double GetVolLagrange() {return vol_lagrange; }
   DesignDensity &GetDesignDensity() {return density;}
   // ρ - ρ_other where ρ_other is the provided density.
   // Assume ρ is constructed by the same mapping.
   // @note If you need different mapping between two grid functions,
   //       use GetDensity().
   std::unique_ptr<Coefficient> GetDensityDiffCoeff(GridFunction &other_gf)
   {
      return density.GetDensityDiffCoeff(other_gf);
   }
   GridFunction &GetState() {return *state;}
protected:
private:
};


/// @brief Strain energy density coefficient
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient &lambda;
   Coefficient &mu;
   GridFunction &u1; // displacement
   GridFunction &u2; // dual-displacement
   Coefficient &dphys_dfrho;
   DenseMatrix grad1, grad2; // auxiliary matrix, used in Eval

public:
   StrainEnergyDensityCoefficient(Coefficient &lambda, Coefficient &mu,
                                  GridFunction &u, DensityProjector &projector, GridFunction &frho)
      : lambda(lambda), mu(mu),  u1(u),  u2(u),
        dphys_dfrho(projector.GetDerivative(frho))
   { }
   StrainEnergyDensityCoefficient(Coefficient &lambda, Coefficient &mu,
                                  GridFunction &u1, GridFunction &u2, DensityProjector &projector,
                                  GridFunction &frho)
      : lambda(lambda), mu(mu),  u1(u1),  u2(u2),
        dphys_dfrho(projector.GetDerivative(frho))
   { }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};


/// @brief Strain energy density coefficient
class IsoStrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient &E;
   Coefficient &nu;
   GridFunction &u1; // displacement
   GridFunction &u2; // dual-displacement
   Coefficient &dphys_dfrho;
   DenseMatrix grad1, grad2; // auxiliary matrix, used in Eval
   Vector voigt1, voigt2; // Voigt notation of symmetric gradient
   DenseMatrix C;

   void VectorGradToVoigt(DenseMatrix &grad, Vector &voigt);
public:
   IsoStrainEnergyDensityCoefficient(Coefficient &E, Coefficient &nu,
                                     GridFunction &u, DensityProjector &projector, GridFunction &frho)
      : E(E), nu(nu),  u1(u),  u2(u),
        dphys_dfrho(projector.GetDerivative(frho))
   { }
   IsoStrainEnergyDensityCoefficient(Coefficient &E, Coefficient &nu,
                                     GridFunction &u1, GridFunction &u2, DensityProjector &projector,
                                     GridFunction &frho)
      : E(E), nu(nu),  u1(u1),  u2(u2),
        dphys_dfrho(projector.GetDerivative(frho))
   { }
   IsoStrainEnergyDensityCoefficient(Coefficient &E, Coefficient &nu,
                                     GridFunction &u1, Coefficient &weight)
      : E(E), nu(nu),  u1(u1),  u2(u1),
        dphys_dfrho(weight)
   { }
   IsoStrainEnergyDensityCoefficient(Coefficient &E, Coefficient &nu,
                                     GridFunction &u1, GridFunction &u2, Coefficient &weight)
      : E(E), nu(nu),  u1(u1),  u2(u2),
        dphys_dfrho(weight)
   { }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

class ParametrizedElasticityEquation : public ParametrizedLinearEquation
{
public:
protected:
   Coefficient &E;
   Coefficient &nu;
   GridFunction &filtered_density;
   ProductCoefficient phys_E;
   VectorCoefficient &f;
private:

public:
   ParametrizedElasticityEquation(FiniteElementSpace &fes,
                                  GridFunction &filtered_density,
                                  DensityProjector &projector,
                                  Coefficient &E, Coefficient &nu,
                                  VectorCoefficient &f, Array2D<int> &ess_bdr, bool enforce_symmetricity=false);
   std::unique_ptr<Coefficient> GetdEdfrho(GridFunction &u,
                                           GridFunction &dual_solution, GridFunction &frho) override
   { return std::unique_ptr<Coefficient>(new IsoStrainEnergyDensityCoefficient(E, nu, u, dual_solution, projector, frho)); }
protected:
   void SolveSystem(GridFunction &x) override
   {
      EllipticSolver solver(*a, *b, ess_bdr);
      solver.UseElasticityOptions();
      solver.SetIterativeMode();
      solver.SetMaxIt(1e06);
      bool converged = solver.Solve(x, AisStationary, BisStationary);
      if (!converged)
      {
#ifdef MFEM_USE_MPI
         if (!Mpi::IsInitialized() || Mpi::Root())
         {
            out << "ParametrizedElasticityEquation::SolveSystem Failed to Converge." <<
                std::endl;
         }
#else
         out << "ParametrizedElasticityEquation::SolveSystem Failed to Converge." <<
             std::endl;
#endif
      }
   }
private:
};


/// @brief Strain energy density coefficient
class ThermalEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient &kappa;
   GridFunction &u1; // displacement
   GridFunction &u2; // displacement
   Coefficient &dphys_dfrho;
   Vector grad1, grad2; // auxiliary matrix, used in Eval

public:
   ThermalEnergyDensityCoefficient(Coefficient &kappa,
                                   GridFunction &u, DensityProjector &projector, GridFunction &frho)
      : kappa(kappa),  u1(u),  u2(u),
        dphys_dfrho(projector.GetDerivative(frho))
   { }
   ThermalEnergyDensityCoefficient(Coefficient &kappa,
                                   GridFunction &u1, GridFunction &u2, DensityProjector &projector,
                                   GridFunction &frho)
      : kappa(kappa),  u1(u1),  u2(u2),
        dphys_dfrho(projector.GetDerivative(frho))
   { }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

class ParametrizedDiffusionEquation : public ParametrizedLinearEquation
{
public:
protected:
   Coefficient &kappa;
   GridFunction &filtered_density;
   ProductCoefficient phys_kappa;
   Coefficient &f;
private:

public:
   ParametrizedDiffusionEquation(FiniteElementSpace &fes,
                                 GridFunction &filtered_density,
                                 DensityProjector &projector,
                                 Coefficient &kappa,
                                 Coefficient &f, Array2D<int> &ess_bdr);
   std::unique_ptr<Coefficient> GetdEdfrho(GridFunction &u,
                                           GridFunction &dual_solution, GridFunction &frho) override
   { return std::unique_ptr<Coefficient>(new ThermalEnergyDensityCoefficient(kappa, u, dual_solution, projector, frho)); }
protected:
   void SolveSystem(GridFunction &x) override
   {
      EllipticSolver solver(*a, *b, ess_bdr);
      solver.SetIterativeMode();
      solver.SetMaxIt(1e06);
      bool converged = solver.Solve(x, AisStationary, BisStationary);
      if (!converged)
      {
#ifdef MFEM_USE_MPI
         if (!Mpi::IsInitialized() || Mpi::Root())
         {
            out << "ParametrizedDiffusionEquation::SolveSystem Failed to Converge." <<
                std::endl;
         }
#else
         out << "ParametrizedDiffusionEquation::SolveSystem Failed to Converge." <<
             std::endl;
#endif
      }
   }
private:
};



/// @brief Find step size α satisfies F(ρ(α)) ≤ F(ρ_0) - c_1 (∇F, ρ(α) - ρ_0) where ρ(α) = P(ρ_0 - α d)
///        where P is a projection, d is the negative search direction.
///
///        We assume, 1) problem is already evaluated at the current point, ρ_0.
///                   2) projection is, if any, performed by problem.Eval()
///        The resulting density, ρ(α) and the function value will be updated in problem.
/// @param problem Topology optimization problem
/// @param x0 Current density gridfunction
/// @param direction Ascending direction
/// @param diff_densityForm Linear from L(v) = (x - x0, v)
/// @param c1 Weights on the directional derivative,
/// @param step_size Step size. Use reference to monitor updated step size.
/// @param max_it Maximum number of updates.
/// @param shrink_factor < 1. Step size will be updated to α <- α * shrink_factor
/// @return The number of re-evaluation during Armijo condition check.
int Step_Armijo(TopOptProblem &problem, const GridFunction &x0,
                const GridFunction &direction,
                LinearForm &diff_densityForm, const double c1,
                double &step_size, const int max_it=20, const double shrink_factor=0.5);

int Step_Bregman(TopOptProblem &problem, const GridFunction &x0,
                 const GridFunction &direction,
                 LinearForm &diff_densityForm,
                 double &step_size, const int max_it=20, const double shrink_factor=0.5);
/// @brief Volumetric force for linear elasticity
class VolumeForceCoefficient : public VectorCoefficient
{
private:
   double r2;
   Vector &center;
   Vector &force;
public:
   VolumeForceCoefficient(double r_,Vector &  center_, Vector & force_);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   void Set(double r_,Vector & center_, Vector & force_);
   void UpdateSize();
};

/// @brief Volumetric force for linear elasticity
class LineVolumeForceCoefficient : public VectorCoefficient
{
private:
   double r2;
   Vector &center;
   Vector &force;
   int direction_dim;
public:
   LineVolumeForceCoefficient(double r_, Vector &center_, Vector & force_,
                              int direction_dim);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   void Set(double r_,Vector & center_, Vector & force_);
   void UpdateSize();
};
}
#endif
