//                  MFEM Example 35 - Serial/Parallel Shared Code
//
//

#include <fstream>
#include <iostream>

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Inverse of sigmoid function
double inv_sigmoid(const double x) {
  double tol = 1e-12;
  const double tmp = min(max(tol, x), 1.0 - tol);
  return log(max(tmp / (1.0 - tmp), 1.e-12));
}

// sigmoid function
double sigmoid(const double x) {
  if (x >= 0) {
    return 1.0 / (1.0 + exp(-x));
  } else {
    return exp(x) / (1.0 + exp(x));
  }
}
// derivative of sigmoid function
double dsigmoiddx(const double x) {
  double tmp = sigmoid(-x);
  return tmp - pow(tmp, 2);
}

/**
 * @brief Sigmoid of a grid function, ψ ↦ sigmoid(ψ) = ρ
 *
 */
class SigmoidCoefficient : public Coefficient {
 protected:
  GridFunction *rho;

 public:
  SigmoidCoefficient(GridFunction *rho_) : rho(rho_) {}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) {
    return sigmoid(rho->GetValue(T, ip));
  }
};

/**
 * @brief Derivative of Sigmoid of a grid function, ψ ↦ d(sigmoid(ψ))/dψ
 * @a ./ex35.cpp::projit
 *
 */
class DerSigmoidCoefficient : public Coefficient {
 protected:
  GridFunction *rho;

 public:
  DerSigmoidCoefficient(GridFunction *rho_) : rho(rho_) {}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) {
    return dsigmoiddx(rho->GetValue(T, ip));
  }
};

/**
 * @brief r(ρ̃) in SIMP, r : ρ̃ ↦ ρₘᵢₙ + (ρₘₐₓ-ρₘᵢₙ)ρ̃ᵖ
 * where ρₘᵢₙ, ρₘₐₓ are the min/max value, and p is the exponent
 *
 */
class SIMPCoefficient : public Coefficient {
 protected:
  GridFunction *rho_filter;  // grid function
  const double min_val;
  const double max_val;
  const double exponent;

 public:
  SIMPCoefficient(GridFunction *rho_filter_, double min_val_ = 1e-3,
                  double max_val_ = 1.0, double exponent_ = 3)
      : rho_filter(rho_filter_),
        min_val(min_val_),
        max_val(max_val_),
        exponent(exponent_) {}

  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) {
    // return min_val +
    //        pow(max(rho_filter->GetValue(T, ip), 0.0), exponent) * (max_val -
    //        min_val);
    return max(min_val,
               min(max_val, pow(rho_filter->GetValue(T, ip), exponent)));
  }
};

/**
 * @brief Logarithmic difference coefficient function, log(u/w)
 *
 */
class LogDiffCoefficient : public Coefficient {
 private:
  const GridFunction *u;
  const GridFunction *w;
  const double log_tol;

 public:
  /**
   * @brief Logarithmic difference coefficient function, log(u/w)
   *
   * @param u_ u
   * @param w_ w
   * @param tol tolerance for log to avoid negative value
   */
  LogDiffCoefficient(GridFunction *u_, GridFunction *w_,
                     const double tol = 1.e-12)
      : u(u_), w(w_), log_tol(tol){};
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) {
    const double u_val = max(u->GetValue(T, ip), log_tol);
    const double w_val = max(w->GetValue(T, ip), log_tol);
    return log(u_val / w_val);
  }
};

class ThermalEnergyCoefficient : public Coefficient {
 private:
  GridFunction *u;
  GridFunction *w;
  GridFunction *rho_filter;
  const double exponent;
  const double rho_min;
  const double rho_max;

  Vector grad_u;
  Vector grad_w;

 public:
  ThermalEnergyCoefficient(GridFunction *u_, GridFunction *w_,
                           GridFunction *rho_filter_, const double exponent_,
                           const double rho_min_, const double rho_max_)
      : u(u_),
        w(w_),
        rho_filter(rho_filter_),
        exponent(exponent_),
        rho_min(rho_min),
        rho_max(rho_max){};
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) {
    u->GetGradient(T, grad_u);
    w->GetGradient(T, grad_w);
    const double energy = grad_u * grad_w;
    const double rho_val = rho_filter->GetValue(T, ip);
    const double dr_drho =
        -exponent * (rho_max - rho_min) * pow(rho_val, exponent - 1.0);
    return -dr_drho * energy;
  };
};

class EllipticBilinearSolver {
 private:
  BilinearForm *a;
  LinearForm *b;
  Array<int> *ess_tdof_list;
  OperatorPtr A;
  Vector B, X;

  const bool pa;
  const bool algebraic_ceed;

 public:
  EllipticBilinearSolver(BilinearForm *a_, LinearForm *b_,
                         Array<int> *ess_dof_list_, const bool pa_ = false,
                         const bool algebraic_ceed_ = false)
      : a(a_),
        b(b_),
        ess_tdof_list(ess_dof_list_),
        pa(pa_),
        algebraic_ceed(algebraic_ceed_) {}

  inline void AssembleRHS() { b->Assemble(); }
  inline void AssembleSystem() {
    a->Assemble();
    b->Assemble();
  }

  inline void setRHS(LinearForm *b_, const bool assemble = false) {
    b = b_;
    if (assemble) b->Assemble();
  };

  void Solve(GridFunction *sol) {
    a->FormLinearSystem(*ess_tdof_list, *sol, *b, A, X, B);

    // 11. Solve the linear system A X = B.
    if (!pa) {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix &)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
    } else {
      if (UsesTensorBasis(*(sol->FESpace()))) {
        if (algebraic_ceed) {
          ceed::AlgebraicSolver M(*a, *ess_tdof_list);
          PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
        } else {
          OperatorJacobiSmoother M(*a, *ess_tdof_list);
          PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
        }
      } else {
        CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
    }

    // 12. Recover the solution as a finite element grid function.
    a->RecoverFEMSolution(X, *b, *sol);
  };
};
