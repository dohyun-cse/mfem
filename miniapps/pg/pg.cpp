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
   dualhess->UsePrecomputedSparsity();

   pg_blockmat = std::make_unique<BlockMatrix>(offsets);
   pg_blockmat->SetBlock(0, 0, static_cast<SparseMatrix*>(&A));
   pg_blockmat->SetBlock(0, 1, static_cast<SparseMatrix*>(neg_Bt.get()));
   pg_blockmat->SetBlock(1, 0, static_cast<SparseMatrix*>(&B));
   pg_blockmat->owns_blocks = false;
   H.SetType(Operator::MFEM_SPARSEMAT);
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
   H.SetType(Operator::Hypre_ParCSR);
}
#endif

// Au - B^T lambda
// Bu - grad R^*(psi^k - alpha*lambda)
void PGOperator::Mult(const Vector &x, Vector &y) const
{
   if (debug) { out << "PGOperator::Mult" << std::endl; }
   // [u, lambda]
   BlockVector X(const_cast<Vector&>(x), offsets);
   Vector &u = X.GetBlock(0);
   Vector &lambda = X.GetBlock(1);

   if (debug) { out << "PGOperator::Mult #1: latent update" << std::endl; }
   // psi = psi_k - alpha*lambda
   add(psi_k->GetTrueVector(), -alpha, lambda, psi->GetTrueVector());
   psi->SetFromTrueVector();
   psi->HostRead(); // sync GPU → CPU before coefficient evaluation in Assemble()

   if (debug) { out << "PGOperator::Mult #2: setup output vector" << std::endl; }
   y.SetSize(Height());
   BlockVector Y(y, offsets);
   Vector &res_u = Y.GetBlock(0);
   Vector &res_lambda = Y.GetBlock(1);

   if (debug)
   {
      u.HostRead(); lambda.HostRead();
      out << "PGOperator::Mult norms: ||u|| = " << u.Norml2()
          << ", ||lambda|| = " << lambda.Norml2()
          << ", ||psi|| = " << psi->Norml2() << std::endl;
   }

   if (debug) { out << "PGOperator::Mult #3: compute A*u - B^T*lambda" << std::endl; }
   // res_u = A*u - B^T*lambda
   A.Mult(u, res_u);
   neg_Bt->AddMult(lambda, res_u);

   if (debug)
   {
      res_u.HostRead();
      out << "  ||A*u - B^T*lambda|| = " << res_u.Norml2() << std::endl;
   }

   if (debug) { out << "PGOperator::Mult #4: compute B*u - grad R^*(psi)" << std::endl; }
   // res_lambda = B*u - gradinv(psi)
   // Assemble gradinv into dualgrad's own memory to avoid alias sync issues
   // with device memory, then combine via standard vector operations.
   dualgrad->Assemble();
   if (debug)
   {
      dualgrad->HostRead();
      out << "  ||dualgrad|| = " << dualgrad->Norml2() << std::endl;
   }
   if (parallel)
   {
#ifdef MFEM_USE_MPI
      static_cast<ParLinearForm*>(dualgrad.get())->ParallelAssemble(res_lambda);
      if (debug)
      {
         res_lambda.HostRead();
         out << "  ||ParAssemble(dualgrad)|| = " << res_lambda.Norml2() << std::endl;
      }
      res_lambda.Neg();
      B.AddMult(u, res_lambda);
      if (debug)
      {
         res_lambda.HostRead();
         out << "  ||B*u - dualgrad|| = " << res_lambda.Norml2() << std::endl;
      }
#endif
   }
   else
   {
      B.Mult(u, res_lambda);
      res_lambda.Add(-1.0, *dualgrad);
   }

   // Block ops wrote correct data to y's device memory (shared via aliases).
   // Update y's flags: device valid, host stale.
   y.Write();
   if (debug) { out << "PGOperator::Mult done" << std::endl; }
}

// ---------------------------------------------------------------------------
// GetGradient
// ---------------------------------------------------------------------------
Operator &PGOperator::GetGradient(const Vector &x) const
{
   if (debug) {out << "PGOperator::GetGradient" << std::endl; }
   BlockVector X(const_cast<Vector&>(x), offsets);
   // Vector &u = X.GetBlock(0);
   Vector &lambda = X.GetBlock(1);

   if (debug) {out << "PGOperator::GetGradient #1: latent update" << std::endl; }
   add(psi_k->GetTrueVector(), -alpha, lambda, psi->GetTrueVector());
   psi->SetFromTrueVector();
   psi->HostRead(); // sync GPU → CPU before coefficient evaluation in Assemble()

   if (debug) {out << "PGOperator::GetGradient #2: update dual Hessian" << std::endl; }
   dualhess->Assemble(false);
   dualhess->SpMat() *= alpha;
   dualhess->FormSystemMatrix(latent_ess_tdof, H);

   if (debug) {out << "PGOperator::GetGradient #3: creating monolithic" << std::endl; }
   if (parallel)
   {
#ifdef MFEM_USE_MPI
      Array2D<const HypreParMatrix*> blocks(2, 2);
      blocks(0, 0) = static_cast<const HypreParMatrix*>(&A);
      blocks(0, 1) = static_cast<const HypreParMatrix*>(neg_Bt.get());
      blocks(1, 0) = static_cast<const HypreParMatrix*>(&B);
      blocks(1, 1) = H.As<HypreParMatrix>();
      pg_op.Reset(HypreParMatrixFromBlocks(blocks));
#endif
   }
   else
   {
      // serial
      pg_blockmat->SetBlock(1, 1, H.As<SparseMatrix>());
      pg_op.Reset(pg_blockmat->CreateMonolithic());
   }
   dualhess->Update(); // no longer needed after this.
   if (debug) {out << "PGOperator::GetGradient done" << std::endl; }
   return *pg_op;
}

} // namespace mfem
