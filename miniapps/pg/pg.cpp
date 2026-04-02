#include "pg.hpp"

namespace mfem
{

PGOperator::PGOperator(Operator &A_,
                       Operator &B_,
                       FiniteElementSpace &fespace_,
                       LegendreFunction &entropy,
                       const real_t &alpha_)
   : fespace(fespace_), A(A_), B(B_), alpha(alpha_), latent_ess_tdof(0)
{
   MFEM_VERIFY(dynamic_cast<SparseMatrix*>(&A) != nullptr,
               "A is not a sparse matrix. Currently, it only supports sparse matrix")
   MFEM_VERIFY(dynamic_cast<SparseMatrix*>(&B) != nullptr,
               "B is not a sparse matrix. Currently, it only supports sparse matrix")
   MFEM_VERIFY(A.Height() == A.Width(),
               "A must be a square matrix: " << A.Height() << " x " << A.Width())
   MFEM_VERIFY(B.Width() == A.Height(),
               "Incompatible dimensions of B and A: B(" << B.Height() << " x " << B.Width()
               << ") vs A(" << A.Height() << " x " << A.Width() << ")")
   MFEM_VERIFY(B.Height() == fespace.GetVSize(),
               "Incompatible dimensions of B and fespace: B(" << B.Height() << " x "
               << B.Width() << ") vs fespace(" << fespace.GetVSize() << ")")

   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = A.Height();
   offsets[2] = B.Height();
   offsets.PartialSum();
   width=height=offsets.Last();


   auto * neg_Bt_tmp = Transpose(static_cast<SparseMatrix&>(B));
   *neg_Bt_tmp *= -1.0;
   neg_Bt.reset(neg_Bt_tmp);

   psi = std::make_unique<GridFunction>(&fespace_);
   *psi = 0.0; psi->SetTrueVector();
   psi_k = std::make_unique<GridFunction>(&fespace_);
   *psi_k = 0.0; psi_k->SetTrueVector();

   primal_cf = std::make_unique<PrimalCoefficient>(*psi, entropy);
   dualgrad  = std::make_unique<LinearForm>(&fespace_);
   dualgrad->AddDomainIntegrator(new DomainLFIntegrator(*primal_cf));

   primal_jacobian_cf = std::make_unique<PrimalJacobianCoefficient>(*psi, entropy);
   dualhess  = std::make_unique<BilinearForm>(&fespace_);
   dualhess->AddDomainIntegrator(new MassIntegrator(*primal_jacobian_cf));

   pg_blockmat = std::make_unique<BlockMatrix>(offsets);
   pg_blockmat->SetBlock(0, 0, static_cast<SparseMatrix*>(&A));
   pg_blockmat->SetBlock(0, 1, static_cast<SparseMatrix*>(neg_Bt.get()));
   pg_blockmat->SetBlock(1, 0, static_cast<SparseMatrix*>(&B));
}

#ifdef MFEM_USE_MPI
// ---------------------------------------------------------------------------
// Parallel constructor
// ---------------------------------------------------------------------------
PGOperator::PGOperator(Operator &A_,
                       Operator &B_,
                       ParFiniteElementSpace &fespace_,
                       LegendreFunction &entropy,
                       const real_t &alpha_)
   : fespace(fespace_), A(A_), B(B_), alpha(alpha_), latent_ess_tdof(0)
{
   MFEM_VERIFY(dynamic_cast<HypreParMatrix*>(&A) != nullptr,
               "A is not a hypre sparse matrix. Currently, it only supports sparse matrix")
   MFEM_VERIFY(dynamic_cast<HypreParMatrix*>(&B) != nullptr,
               "B is not a hypre sparse matrix. Currently, it only supports sparse matrix")
   MFEM_VERIFY(A.Height() == A.Width(),
               "A must be a square matrix: " << A.Height() << " x " << A.Width())
   MFEM_VERIFY(B.Width() == A.Height(),
               "Incompatible dimensions of B and A: B(" << B.Height() << " x " << B.Width()
               << ") vs A(" << A.Height() << " x " << A.Width() << ")")
   MFEM_VERIFY(B.Height() == fespace.GetTrueVSize(),
               "Incompatible dimensions of B and fespace: B(" << B.Height() << " x "
               << B.Width() << ") vs fespace(" << fespace.GetTrueVSize() << ")")

   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = A.Height();
   offsets[2] = B.Height();
   offsets.PartialSum();
   width=height=offsets.Last();

   auto neg_Bt_tmp = static_cast<HypreParMatrix&>(B).Transpose();
   *neg_Bt_tmp *= -1.0;
   neg_Bt.reset(neg_Bt_tmp);

   psi = std::make_unique<ParGridFunction>(&fespace_);
   *psi = 0.0; psi->SetTrueVector();
   psi_k = std::make_unique<ParGridFunction>(&fespace_);
   *psi_k = 0.0; psi_k->SetTrueVector();

   primal_cf = std::make_unique<PrimalCoefficient>(*psi, entropy);
   dualgrad  = std::make_unique<ParLinearForm>(&fespace_);
   dualgrad->AddDomainIntegrator(new DomainLFIntegrator(*primal_cf));

   primal_jacobian_cf = std::make_unique<PrimalJacobianCoefficient>(*psi, entropy);
   dualhess  = std::make_unique<ParBilinearForm>(&fespace_);
   dualhess->AddDomainIntegrator(new MassIntegrator(*primal_jacobian_cf));

   parallel = true;
}
#endif

// Au - B^T lambda
// Bu - grad R^*(psi^k - alpha*lambda)
void PGOperator::Mult(const Vector &x, Vector &y) const
{
   // [u, lambda]
   BlockVector X(const_cast<Vector&>(x), offsets);
   Vector &u = X.GetBlock(0);
   Vector &lambda = X.GetBlock(1);

   // psi = psi_k - alpha*lambda
   add(psi_k->GetTrueVector(), -alpha, lambda, psi->GetTrueVector());
   psi->SetFromTrueVector();

   y.SetSize(Height());
   BlockVector Y(y, offsets);
   Vector &res_u = Y.GetBlock(0);
   Vector &res_lambda = Y.GetBlock(1);

   A.Mult(u, res_u);
   neg_Bt->AddMult(lambda, res_u);

   if (parallel)
   {
#ifdef MFEM_USE_MPI
      dualgrad->Assemble();
      static_cast<ParLinearForm*>(dualgrad.get())->ParallelAssemble(res_lambda);
#endif
   }
   else
   {
      dualgrad->Update(&fespace, res_lambda, 0);
      dualgrad->Assemble();
   }
   res_lambda.Neg();
   B.AddMult(u, res_lambda);
}

// ---------------------------------------------------------------------------
// GetGradient
// ---------------------------------------------------------------------------
Operator &PGOperator::GetGradient(const Vector &x) const
{
   BlockVector X(const_cast<Vector&>(x), offsets);
   // Vector &u = X.GetBlock(0);
   Vector &lambda = X.GetBlock(1);

   add(psi_k->GetTrueVector(), -alpha, lambda, psi->GetTrueVector());
   psi->SetFromTrueVector();

   dualhess->Update();
   dualhess->Assemble(false);
   if (parallel)
   {
#ifdef MFEM_USE_MPI
      dualH.reset(new HypreParMatrix);
      dualhess->FormSystemMatrix(latent_ess_tdof, *dualH);
      *dualH *= alpha;
      Array2D<const HypreParMatrix*> blocks(2, 2);
      blocks(0, 0) = static_cast<const HypreParMatrix*>(&A);
      blocks(0, 1) = static_cast<const HypreParMatrix*>(neg_Bt.get());
      blocks(1, 0) = static_cast<const HypreParMatrix*>(&B);
      blocks(1, 1) = dualH.get();
      pg_op_par.reset(HypreParMatrixFromBlocks(blocks));
      return *pg_op_par;
#endif
   }
   // serial
   SparseMatrix H;
   dualhess->FormSystemMatrix(latent_ess_tdof, H);
   H *= alpha;
   pg_blockmat->SetBlock(1, 1, &H);
   pg_op.reset(pg_blockmat->CreateMonolithic());
   MemoryType mt = Device::GetMemoryType();
   pg_op->UseGPUSparse(mt == MemoryType::DEVICE);
   return *pg_op;
}

} // namespace mfem
