#pragma once
#include "mfem.hpp"
#include "bregman.hpp"

namespace mfem
{
class PGOperator : public Operator
{
private:
   bool debug = false;
   FiniteElementSpace &fespace;
   Operator &A;
   Operator &B;
   std::unique_ptr<Operator> neg_Bt;
   std::unique_ptr<BlockMatrix> pg_blockmat;
   const real_t &alpha;
   std::unique_ptr<PrimalCoefficient> primal_cf;
   std::unique_ptr<PrimalJacobianCoefficient> primal_jacobian_cf;
   std::unique_ptr<GridFunction> psi_k;
   std::unique_ptr<GridFunction> psi;
   std::unique_ptr<BilinearForm> dualhess;
   std::unique_ptr<LinearForm> dualgrad;
   mutable std::unique_ptr<SparseMatrix> pg_op;
   mutable SparseMatrix dualH_serial;
   mutable Array<int> latent_ess_tdof;
   Array<int> offsets;
   bool parallel = false;
#ifdef MFEM_USE_MPI
   mutable std::unique_ptr<HypreParMatrix> dualH;
   mutable std::unique_ptr<HypreParMatrix> pg_op_par;
#endif

public:
   void SetDebug(bool debug_) { debug = debug_; }
   PGOperator(Operator &A_,
              Operator &B_,
              FiniteElementSpace &fespace_,
              LegendreFunction &entropy,
              const real_t &alpha_);

#ifdef MFEM_USE_MPI
   PGOperator(Operator &A_,
              Operator &B_,
              ParFiniteElementSpace &fespace_,
              LegendreFunction &entropy,
              const real_t &alpha_);
#endif
   void SetLatentEssentialBC(const Array<int> &is_bdr_ess)
   {
      fespace.GetEssentialTrueDofs(is_bdr_ess, latent_ess_tdof);
   }

   void Mult(const Vector &x, Vector &y) const override;
   Operator &GetGradient(const Vector &x) const override;
   // psi_k <- psi = psi_k - alpha*lambda
   void ProxUpdate(const GridFunction &lambda)
   {
      psi_k->Add(-alpha, lambda);
      psi_k->SetTrueVector();
   }
};

} // namespace mfem
