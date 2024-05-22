#ifndef THREE_FIELD_HPP
#define THREE_FIELD_HPP
#include "helper.hpp"
#include "mfem.hpp"

namespace mfem
{
const double LOG_TOL = 1e-12;
const double LOG_MIN_VAL = std::log(LOG_TOL);

// safe log by clipping log value
inline double safe_log(double x)
{
   return x < LOG_TOL ? LOG_MIN_VAL : std::log(x);
}

// fermi-dirac entropy
inline double binary_entropy(double x)
{
   const double logx = safe_log(x);
   const double logy = safe_log(1.0 - x);
   return -x * logx - (1.0 - x) * logy;
}

// compute 1/(1+exp(-x))
inline double sigmoid(double x)
{
   if (x > 0)
   {
      return 1.0 / (1.0 + std::exp(-x));
   }
   else
   {
      const double expx = std::exp(x);
      return expx / (1.0 + expx);
   }
}

inline double sigmoid_derivative(double x)
{
   const double s = sigmoid(x);
   return s * (1.0 - s);
}

// compute log(x/(1-x)) with safe log
inline double logit(double x)
{
   // avoid divide by 0
   const double y = x < 0.5 ? x / (1.0 - x) : (1.0 - x) / x;
   const double logy = safe_log(y);
   return x < 0.5 ? logy : -logy;
}

inline double shannon_entropy(double x)
{
   return -x * safe_log(x);
}


enum VolumeSolver
{
   Bisection,
   Brent
};
class NonlinearEquationSolver
{
protected:
   std::function<double()> f;
   double &x;
   const double tol_res;
   const double tol_x;
public:
   NonlinearEquationSolver(double &x, std::function<double()> f,
                           double tol_res=1e-08, double tol_x=0):
      f(f), x(x), tol_res(tol_res), tol_x(tol_x) {}
   virtual double Solve() = 0;
};
class BisectionSolver : public NonlinearEquationSolver
{
protected:
   double xl, xr;
   bool interval_prescribed=false;
public:
   BisectionSolver(double &x, std::function<double()> f, double tol_x,
                   double tol_res=1e-08): NonlinearEquationSolver(x, f, tol_res, tol_x) {};
   void SetInterval(double new_xl, double new_xr) {xl = new_xl; xr = new_xr; interval_prescribed=true;}
   virtual double Solve()
   {
      if (!interval_prescribed) { MFEM_ABORT("Search Interval not Prescribed. Set Interval Before each Solve"); }
      x = xl; double yl = f();
      x = xr; double yr = f();

      double y;
      while (xr - xl > tol_x)
      {
         x = (xr + xl)*0.5; y = f();
         if (std::abs(y) < tol_res) { break; }

         if (yl * y > 0) { xl = x; yl = y; }
         else { xr = x; yr = y; }
      }
      return y;
   }
};

class BrentSolver : public NonlinearEquationSolver
{
protected:
   double x0, x1;
   bool interval_prescribed=false;
public:
   BrentSolver(double &x, std::function<double()> f, double tol_x,
               double tol_res=1e-08): NonlinearEquationSolver(x, f, tol_res, tol_x) {};
   void SetInterval(double new_xl, double new_xr) {x0 = new_xl; x1 = new_xr; interval_prescribed=true;}
   virtual double Solve()
   {
      if (!interval_prescribed) { MFEM_ABORT("Search Interval not Prescribed. Set Interval Before each Solve"); }
      x = x0; double y0 = f();
      x = x1; double y1 = f();
      if (abs(y0) < abs(y1))
      {
         std::swap(y0, y1);
         std::swap(x0, x1);
      }
      double x2, x3;
      double y2;
      x3 = x2 = x0;
      y2 = y0;
      bool bisection = true;
      double y;

      while (x1 - x0 > tol_x)
      {
         if (std::abs(y0 - y2) > tol_res && std::abs(y1 - y2) > tol_res)
         {
            x =  x0*y1*y2/((y0-y1)*(y0-y2)) +
                 x1*y0*y2/((y1-y0)*(y1-y2)) +
                 x2*y0*y1/((y2-y0)*(y2-y1));
         }
         else
         {
            x = x1 - y1 * (x1 - x0)/(y1 - y0);
         }
         double delta = fabs(2e-12*std::abs(x1));
         double min1 = std::abs(x - x1);
         double min2 = std::abs(x1 - x2);
         double min3 = std::abs(x2 - x3);
         if ((x < (3*x0 + x1) / 4.0 && x > x1) ||
             (bisection &&
              (min1 >= min2*0.5 || min1 >= min3*0.5 || min2 < delta || min3 < delta)))
         {
            x = (x0 + x1) * 0.5;
            bisection = true;
         }
         else
         {
            bisection = false;
         }
         y = f();
         if (std::abs(y) < tol_res)
         {
            return y;
         }
         x3 = x2;
         x2 = x1;
         if (y0 * y < 0)
         {
            x1 = x; y1 = y;
         }
         else
         {
            x0 = x; y0 = y;
         }
         if (std::abs(y0) < std::abs(y1))
         {
            std::swap(x0, x1);
            std::swap(y0, y1);
         }
      }
      return y;
   }
};

enum VolConstraint { VOL_MIN=-1, VOL_EQ=0, VOL_MAX=1 };

class DesignDensity
{
public:
   DesignDensity(FiniteElementSpace &fes, VolConstraint vol_constraint,
                 double vol_frac) : fes(fes), vol_constraint(vol_constraint),
      vol_frac(vol_frac)
   {
      density_gf.reset(MakeGridFunction(&fes));
      grad.reset(MakeGridFunction(&fes));
      density_cf.reset(new GridFunctionCoefficient(density_gf.get()));
      grad_cf.reset(new GridFunctionCoefficient(grad.get()));
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
      if (pfes) {parallel = true; comm = pfes->GetComm();}
#endif
      volume_shift = 0.0;
      density_integral_form.reset(MakeLinearForm(&fes));
      shifted_density.reset(new MappedGridFunctionCoefficient(
                               density_gf.get(), [this](double x)
      {
         return std::max(this->min_val, std::min(this->max_val, x + this->volume_shift));
      }));
      density_integral_form->Assemble();
      target_volume = density_integral_form->Sum();
#ifdef MFEM_USE_MPI
      if (parallel) { MPI_Allreduce(MPI_IN_PLACE, &target_volume, 1, MPI_DOUBLE, MPI_SUM, comm); }
#endif
   }
   FiniteElementSpace &GetFE() {return fes;}
   VolConstraint GetVolConstraint() {return vol_constraint; }
   double GetVolFrac() {return vol_frac; }
   Coefficient &GetDensityCoefficient()
   {
      if (!density_cf) { MFEM_ABORT("Density coefficient is undefined."); }
      return *density_cf;
   }
   GridFunction & GetGradGridFunction() { return *grad; }
   Coefficient & GetGradCoefficient() {return *grad_cf;}
   virtual void SetAdjointData(Coefficient *dFdrho) = 0;
   bool CheckVolumeConstraint()
   {
      switch (vol_constraint)
      {
         case mfem::VolConstraint::VOL_EQ: return std::abs(vol_frac - volume) < 1e-08;
         case mfem::VolConstraint::VOL_MAX: return volume < vol_frac;
         case mfem::VolConstraint::VOL_MIN: return volume > vol_frac;
      }
   }
   void SetVolumeConstraint(VolConstraint new_vol_constraint,
                            double new_vol_frac)
   { vol_constraint = new_vol_constraint; vol_frac = new_vol_frac; }
   virtual double Project()
   {
      if (!volume_solver)
      {
         volume_solver.reset(new BisectionSolver(volume_shift,
                                                 [this]()
         {
            this->density_integral_form->Assemble();
            double sum = this->density_integral_form->Sum();
#ifdef MFEM_USE_MPI
            if (this->parallel)
            {
               MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            }
#endif
            return sum - this->target_volume;
         }, 1e-08));
      }
      if (CheckVolumeConstraint()) { return 0.0; }
      static_cast<BisectionSolver&>(*volume_solver).SetInterval(-1, 1);
      volume_solver->Solve();
      double current_volume_shift = volume_shift;
      density_gf->ProjectCoefficient(*shifted_density);
      volume_shift = 0.0;
      return current_volume_shift;
   }

protected:
   FiniteElementSpace &fes;
   double min_val=0.0, max_val=1.0;
   std::unique_ptr<Coefficient> density_cf;
   std::unique_ptr<GridFunction> density_gf, grad;
   std::unique_ptr<Coefficient> grad_cf;
   VolConstraint vol_constraint;
   double vol_frac;
   double target_volume;
   double volume;
   double volume_shift;
   bool parallel = false;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
   std::unique_ptr<Coefficient> shifted_density;
   std::unique_ptr<NonlinearEquationSolver> volume_solver;
   std::unique_ptr<LinearForm> density_integral_form;
};

class DensityFilter
{
public:
   DensityFilter(FiniteElementSpace &fes) : fes(fes) {}
   virtual void SetDensity(Coefficient &rho) = 0;
   virtual Coefficient& GetFilteredDensity() = 0;
   virtual Coefficient& GetFilteredGradient() = 0;
   virtual void UpdateFilteredDensity(Coefficient &rho) = 0;
   virtual void UpdateFilteredGradient(Coefficient &dfdrho) = 0;
   virtual void SetBoundary(Array<int> &void_material_bdr)
   {
      ess_bdr = void_material_bdr;
      for (auto &v:ess_bdr) { v = v != 0; }
      void_bdr = void_material_bdr;
      for (auto &v:void_bdr) { v = v == -1; }
      material_bdr = void_material_bdr;
      for (auto &v:material_bdr) { v = v == 1; }
   }

protected:
   FiniteElementSpace &fes;
   std::unique_ptr<Coefficient> filter_coeff;
   std::unique_ptr<Coefficient> grad_coeff;
   Array<int> ess_bdr, void_bdr, material_bdr;
};

class MappedCoefficient : public Coefficient
{
private:
   Coefficient &cf;
   std::function<double(double)> f;
public:
   MappedCoefficient(Coefficient &cf, std::function<double(double)> f):cf(cf),
      f(f) {};
   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return f(cf.Eval(T, ip));
   };
};

class DifferentiableMap
{
public:
   DifferentiableMap(std::function<double(double)> f,
                     std::function<double(double)> df) : f(f), df(df) {}
   void SetBaseCoefficient(Coefficient &c)
   {
      f_cf.reset(new MappedCoefficient(c, f));
      df_cf.reset(new MappedCoefficient(c, df));
   };

   MappedCoefficient& GetCoefficient()
   {
      if (!f_cf) {MFEM_ABORT("Base coefficient is not set. Run SetCoefficient first.");}
      return *f_cf;
   }

   MappedCoefficient& GetGradient()
   {
      if (!df_cf) {MFEM_ABORT("Base coefficient is not set. Run SetCoefficient first.");}
      return *df_cf;
   }

   MappedCoefficient* NewCoefficient(Coefficient &c)
   {
      return new MappedCoefficient(c, f);
   }
   MappedCoefficient* NewGradCoefficient(Coefficient &c)
   {
      return new MappedCoefficient(c, df);
   }

private:
   std::function<double(double)> f;
   std::function<double(double)> df;
   std::unique_ptr<MappedCoefficient> f_cf, df_cf;
};

class ThreeFieldDensity: public DesignDensity
{
public:
   ThreeFieldDensity(FiniteElementSpace &fes,
                     DensityFilter &filter,
                     DifferentiableMap &projector,
                     VolConstraint vol_constraint,
                     double vol_frac)
      : DesignDensity(fes, vol_constraint, vol_frac), filter(filter),
        projector(projector)
   {
      raw_density_cf.reset(density_cf.release());
      filter.SetDensity(*raw_density_cf);
      density_cf.reset(projector.NewCoefficient(filter.GetFilteredDensity()));
   }

protected:
private:
public:
protected:
   std::unique_ptr<Coefficient> raw_density_cf;
   DensityFilter &filter;
   DifferentiableMap &projector;

private:
};

class LegendreFunction : public DifferentiableMap
{
public:
   LegendreFunction(std::function<double(double)> f,
                    std::function<double(double)> df,
                    std::function<double(double)> df_inv)
      : DifferentiableMap(f, df), df_inv(df_inv) {}
   void SetCoefficient(Coefficient &c)
   {
      DifferentiableMap::SetBaseCoefficient(c);
      df_inv_cf.reset(new MappedCoefficient(c, df_inv));
   }
   MappedCoefficient &GetPrimal2Dual()
   {
      return DifferentiableMap::GetGradient();
   }
   MappedCoefficient &GetDual2Primal()
   {
      if (!df_inv_cf) {MFEM_ABORT("Base coefficient is not set. Run SetCoefficient first.");}
      return *df_inv_cf;
   }

private:
   std::function<double(double)> df_inv;
   std::unique_ptr<MappedCoefficient> df_inv_cf;
};

inline LegendreFunction FermiDiracEntropy()
{
   return LegendreFunction(binary_entropy, logit, sigmoid);
}

inline LegendreFunction ShannonEntropy()
{
   return LegendreFunction(shannon_entropy, safe_log, [](double x) {return std::exp(x);});
}

inline double simp(double x, double x0, double p)
{
   return x0 + (1.0-x0)*std::pow(x, p);
}
inline double der_simp(double x, double x0, double p)
{
   return p*(1.0-x0)*std::pow(x, p-1.0);
}
inline DifferentiableMap SIMPProjector(const double &x0, const double &p)
{
   DifferentiableMap f([&x0, &p](double x) {return simp(x, x0, p);}, [&x0,
                                                                      &p](double x) {return der_simp(x, x0, p);});
   return f;
}

inline double ramp(double x, double x0, double q)
{
   return x0 + (1.0-x0)*x/(1.0 + q*(1.0-x));
}
inline double der_ramp(double x, double x0, double q)
{
   return (1.0-x0)*(1.0+q)/std::pow(1.0 + q*(1.0-x), 2.0);
}
inline DifferentiableMap RAMPProjector(const double &x0, const double &p)
{
   DifferentiableMap f([&x0, &p](double x) {return ramp(x, x0, p);}, [&x0,
                                                                      &p](double x) {return ramp(x, x0, p);});
   return f;
}

inline double tanh_proj(double x, double x0, double beta, double eta)
{
   const double tanh_be = std::tanh(beta*eta);
   return x0 + (1.0 - x0)*(tanh_be + std::tanh(beta*(x - eta)))/
          (tanh_be + std::tanh(beta*(1.0-eta)));
}
inline double der_tanh_proj(double x, double x0, double beta, double eta)
{
   return beta*std::pow(1.0/std::cosh(beta*(x-eta)),
                        2.0) / (std::tanh(beta*eta) + std::tanh((1-eta)*beta));
}
inline DifferentiableMap TANHProjector(const double &x0, const double &beta,
                                       const double &eta)
{
   DifferentiableMap f([&x0, &beta, &eta](double x) {return tanh_proj(x, x0, beta, eta);}, [&x0,
         &beta, &eta](double x) {return der_tanh_proj(x, x0, beta, eta);});
   return f;
}

// Elliptic Bilinear Solver
class EllipticSolver
{
protected:
   BilinearForm &a;      // LHS
   LinearForm &b;        // RHS
   Array2D<int> ess_bdr; // Component-wise essential boundary marker
   Array<int> ess_tdof_list;
   bool symmetric;
   bool iterative_mode;
#ifdef MFEM_USE_MPI
   bool parallel; // Flag for ParFiniteElementSpace
   MPI_Comm comm;
#endif
public:
   /// @brief Linear solver for elliptic problem with given Essential BC
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Essential boundary marker for boundary attributes]
   EllipticSolver(BilinearForm &a, LinearForm &b);
   /// @brief Linear solver for elliptic problem with given Essential BC
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Essential boundary marker for boundary attributes]
   EllipticSolver(BilinearForm &a, LinearForm &b, Array<int> &ess_bdr_);
   /// @brief Linear solver for elliptic problem with given component-wise
   /// essential BC ess_bdr[0,:] - All components, ess_bdr[i,:] - ith-direction
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Component-wise essential boundary marker for boundary
   /// attributes, [Row0: all, Row1: x, ...]
   EllipticSolver(BilinearForm &a, LinearForm &b, Array2D<int> &ess_bdr);

   /// @brief Solve linear system and return FEM solution in x.
   /// @param x FEM solution
   /// @param A_assembled If true, skip assembly of LHS (bilinearform)
   /// @param b_Assembled If true, skip assembly of RHS (linearform)
   /// @return convergence flag
   bool Solve(GridFunction &x, bool A_assembled = false,
              bool b_Assembled = false);
   bool SolveTranspose(GridFunction &x, LinearForm &f, bool A_assembled = false,
                       bool b_Assembled = false);
#ifdef MFEM_USE_MPI
   bool isParallel() { return parallel; }
#endif
   bool isSymmetric() { return symmetric; }
   void SetIterativeMode(bool flag = true) { iterative_mode = flag; };
   void SetEssBoundary(Array<int> &ess_bdr_)
   {
      ess_bdr.SetSize(1, ess_bdr_.Size());
      for (int i = 0; i < ess_bdr_.Size(); i++)
      {
         ess_bdr(0, i) = ess_bdr_[i];
      }
      GetEssentialTrueDofs();
   }
   void SetEssBoundary(Array2D<int> &ess_bdr_) { ess_bdr = ess_bdr_; GetEssentialTrueDofs(); }

protected:
   /// @brief Get true dofs related to the boundaries in @ess_bdr
   /// @return True dof list
   void GetEssentialTrueDofs();

private:
};

class HelmholtzFilter : public DensityFilter
{
public:
   HelmholtzFilter(FiniteElementSpace &fes, const double r_min):
      DensityFilter(fes), eps2(std::pow(r_min/(2.0*std::sqrt(3)), 2))
   {
      filter_form.reset(MakeBilinearForm(&fes));
      filter_form->AddDomainIntegrator(new DiffusionIntegrator(eps2));
      filter_form->AddDomainIntegrator(new MassIntegrator());
      filter_form->Assemble();

      filter_rhs.reset(MakeLinearForm(&fes));

      elliptic_solver.reset(new EllipticSolver(*filter_form, *filter_rhs));
      elliptic_solver->SetIterativeMode(true);

      filtered_density.reset(MakeGridFunction(&fes));
      filtered_density_cf.reset(new GridFunctionCoefficient(filtered_density.get()));

      filtered_grad.reset(MakeGridFunction(&fes));
      filtered_grad_cf.reset(new GridFunctionCoefficient(filtered_grad.get()));
   }
   void SetDensity(Coefficient &rho) override
   {
   }
   Coefficient& GetFilteredDensity() override
   {
      return *filtered_density_cf;
   }
   Coefficient& GetFilteredGradient() override
   {
      return *filtered_grad_cf;
   }
   void UpdateFilteredDensity(Coefficient &rho) override
   {
      filter_rhs->GetDLFI()->DeleteAll();
      filter_rhs->AddDomainIntegrator(new DomainLFIntegrator(rho));
      elliptic_solver->Solve(*filtered_density, true, false);
   }
   void UpdateFilteredGradient(Coefficient &dfdrho) override
   {
      filter_rhs->GetDLFI()->DeleteAll();
      filter_rhs->AddDomainIntegrator(new DomainLFIntegrator(dfdrho));
      elliptic_solver->Solve(*filtered_grad, true, false);
   }
   void SetBoundary(Array<int> &void_material_bdr) override
   {
      DensityFilter::SetBoundary(void_material_bdr);
      elliptic_solver->SetEssBoundary(ess_bdr);
      ConstantCoefficient zero_cf(0.0), one_cf(1.0);
      // Fix void and material for filter
      filtered_density->ProjectBdrCoefficient(zero_cf, void_bdr);
      filtered_density->ProjectBdrCoefficient(one_cf, material_bdr);
      // Fix all for grad
      filtered_grad->ProjectBdrCoefficient(zero_cf, ess_bdr);
   }

private:
   ConstantCoefficient eps2;
   std::unique_ptr<BilinearForm> filter_form;
   std::unique_ptr<LinearForm> filter_rhs;
   std::unique_ptr<EllipticSolver> elliptic_solver;
   std::unique_ptr<GridFunction> filtered_density, filtered_grad;
   std::unique_ptr<GridFunctionCoefficient> filtered_density_cf, filtered_grad_cf;
};
} // namespace mfem

#endif
