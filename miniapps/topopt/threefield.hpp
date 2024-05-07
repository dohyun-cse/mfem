#ifndef THREE_FIELD_HPP
#define THREE_FIELD_HPP
#include "fem/bilininteg.hpp"
#include "fem/coefficient.hpp"
#include "general/error.hpp"
#include "helper.hpp"
#include "mfem.hpp"

namespace mfem
{

class DesignDensity
{
public:
   DesignDensity(FiniteElementSpace &fes) : fes(fes) {}
   Coefficient *GetDensityCoefficient() { return density_coeff; }

protected:
   FiniteElementSpace &fes;
   Coefficient *density_coeff;
};

class DensityFilter
{
public:
   DensityFilter(FiniteElementSpace &fes) : fes(fes) {}
   Coefficient *GetFilterCoefficient() { return filter_coeff.get(); }
   virtual void SetDensity(Coefficient *rho) = 0;
   virtual void SetGradientData(Coefficient *dFdrho) = 0;
   virtual void UpdateFilter() = 0;
   virtual void UpdateGradient() = 0;

   /**
    * Set the boundary of the void material
    * @param void_material_bdr void (-1) or material (1) boundary
    */
   virtual void SetBoundary(Array<int> &void_material_bdr_)
   {
      void_material_bdr = void_material_bdr_;
   }

protected:
   FiniteElementSpace &fes;
   std::unique_ptr<Coefficient> filter_coeff;
   Array<int> void_material_bdr;
};

class DifferentiableMap
{
public:
   DifferentiableMap(double (*f)(double), double (*df)(double)) : f(f), df(df) {}

   TransformedCoefficient GetCoefficient(Coefficient *c)
   {
      return TransformedCoefficient(c, f);
   }

   TransformedCoefficient GetGradient(Coefficient *c)
   {
      return TransformedCoefficient(c, df);
   }

private:
   double (*f)(double);
   double (*df)(double);
};

class ThreeFieldDensity
{
public:
   ThreeFieldDensity(DesignDensity &design_density,
                     DensityFilter &density_filter,
                     DifferentiableMap &filter_to_physical)
      : design_density(design_density), density_filter(density_filter),
        filter_to_physical(filter_to_physical) {}

   TransformedCoefficient GetPhysicalDensity()
   {
      return filter_to_physical.GetCoefficient(
                density_filter.GetFilterCoefficient());
   }

protected:
private:
public:
protected:
   DesignDensity &design_density;
   DensityFilter &density_filter;
   DifferentiableMap &filter_to_physical;

private:
};

class LegendreFunction
{
public:
   LegendreFunction(double (*f)(double), double (*df)(double),
                    double (*df_inv)(double))
      : f(f), df(df), df_inv(df_inv) {}
   TransformedCoefficient GetCoefficient(Coefficient *c)
   {
      return TransformedCoefficient(c, f);
   }
   TransformedCoefficient GetGradient(Coefficient *c)
   {
      return TransformedCoefficient(c, df);
   }
   TransformedCoefficient GetGradientInverse(Coefficient *c)
   {
      return TransformedCoefficient(c, df_inv);
   }

private:
   double (*f)(double);
   double (*df)(double);
   double (*df_inv)(double);
};

inline double binary_entropy(double x)
{
   if (x < 1e-08 || x > 1 - 1e-08)
   {
      return 0.0;
   }
   return -x * std::log(x) - (1 - x) * std::log(1 - x);
}

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
   return s * (1 - s);
}

inline double logit(double x)
{
   if (x < 1e-08)
   {
      return -40.0;
   }
   if (x > 1 - 1e-08)
   {
      return 40.0;
   }
   return std::log(x / (1 - x));
}

inline double shannon_entropy(double x)
{
   if (x < 1e-08 || x > 1 - 1e-08)
   {
      return 0.0;
   }
   return -x * std::log(x);
}

inline double exp_double(double x) { return std::exp(x); }

inline double log_double(double x) { return std::log(x); }

inline LegendreFunction FermiDiracEntropy()
{
   return LegendreFunction(binary_entropy, logit, sigmoid);
}

inline LegendreFunction ShannonEntropy()
{
   return LegendreFunction(shannon_entropy, log_double, exp_double);
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
   HelmholtzFilter(FiniteElementSpace &fes, const double r_min);
   void SetDensity(Coefficient *rho) override;
   void SetGradientData(Coefficient *dFdrho) override;
   void UpdateFilter() override;
   void UpdateGradient() override;
   void SetBoundary(Array<int> &void_material_bdr_) override;

private:
   ConstantCoefficient eps2;
   Array<int> ess_bdr;
   std::unique_ptr<BilinearForm> filter_form;
   std::unique_ptr<EllipticSolver> ellipticSolver;
   std::unique_ptr<LinearForm> rho_form;
   std::unique_ptr<GridFunction> filter;
};

class ProximalHelmholtzFilter : public DensityFilter
{
   ProximalHelmholtzFilter(FiniteElementSpace &fes, FiniteElementSpace &dual_fes,
                           const double r_min)
      : DensityFilter(fes), dual_fes(dual_fes),
        eps2(r_min * r_min / (4.0 / 3.0))
   {
      primal_filter.reset(MakeGridFunction(&fes));
      dual_filter.reset(MakeGridFunction(&dual_fes));
      dual_filter_cf.reset(new GridFunctionCoefficient(dual_filter.get()));
      filter_coeff.reset(
         new MappedGridFunctionCoefficient(dual_filter.get(), sigmoid));

      filter_form.reset(MakeBilinearForm(&fes));
      filter_form->AddDomainIntegrator(new DiffusionIntegrator(eps2));
      filter_form->AddDomainIntegrator(new MassIntegrator());
      filter_form->Assemble();
      filter_form->Finalize();

      prox_form.reset(MakeMixedBilinearForm(&fes, &dual_fes));
      prox_form->AddDomainIntegrator(new MixedScalarMassIntegrator());
      prox_form->Assemble();
      prox_form->Finalize();

      prox_formT.reset(new TransposeOperator(prox_form.get()));

      expMass.reset(MakeBilinearForm(&fes));
      expMass->AddDomainIntegrator(new MassIntegrator(*filter_coeff));

      offsets.SetSize(3);
      offsets[0] = 0;
      offsets[1] = fes.GetTrueVSize();
      offsets[2] = dual_fes.GetTrueVSize();
      offsets.PartialSum();

      saddle_point.reset(new BlockOperator(offsets));
      saddle_point->SetBlock(0, 0, filter_form.get());
      saddle_point->SetBlock(1, 0, prox_form.get());
      saddle_point->SetBlock(0, 1, prox_formT.get());
      saddle_point->SetBlock(2, 2, expMass.get());
   }
   void SetDensity(Coefficient *rho) override
   {
      alpha_rho.reset(new ProductCoefficient(1.0, *rho));
      primal_rhs.reset(MakeLinearForm(&fes));
      primal_rhs->AddDomainIntegrator(new DomainLFIntegrator(*alpha_rho));
      primal_rhs->AddDomainIntegrator(new DomainLFIntegrator(*dual_filter_cf));

      dual_rhs.reset(MakeLinearForm(&dual_fes));
      dual_rhs->AddDomainIntegrator(new DomainLFIntegrator(*filter_coeff));
   }
   void SetGradientData(Coefficient *dFdrho) override {}
   void UpdateFilter() override
   {
      for (int i = 0; i < 10; i++)
      {
      }
   }
   void UpdateGradient() override;

private:
   ConstantCoefficient eps2;
   Array<int> offsets;
   FiniteElementSpace &dual_fes;

   std::unique_ptr<ProductCoefficient> alpha_rho;
   std::unique_ptr<GridFunctionCoefficient> dual_filter_cf;
   std::unique_ptr<BlockOperator> saddle_point;
   std::unique_ptr<BilinearForm> filter_form;
   std::unique_ptr<MixedBilinearForm> prox_form;
   std::unique_ptr<TransposeOperator> prox_formT;
   std::unique_ptr<BilinearForm> expMass;
   std::unique_ptr<LinearForm> primal_rhs;
   std::unique_ptr<LinearForm> dual_rhs;
   std::unique_ptr<GridFunction> primal_filter;
   std::unique_ptr<GridFunction> dual_filter;
};

} // namespace mfem

#endif
