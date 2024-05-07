#include "threefield.hpp"
#include "mfem.hpp"

namespace mfem
{

EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b):
   a(a), b(b), ess_bdr(0, 0), ess_tdof_list(0), symmetric(false)
{
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
   if (pfes) {parallel = true; comm = pfes->GetComm(); }
   else {parallel = false;}
#endif
}

EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b,
                               Array<int> &ess_bdr_list):
   a(a), b(b), ess_bdr(1, ess_bdr_list.Size()), ess_tdof_list(0), symmetric(false)
{
   for (int i=0; i<ess_bdr_list.Size(); i++)
   {
      ess_bdr(0, i) = ess_bdr_list[i];
   }
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
   if (pfes) {parallel = true; comm = pfes->GetComm(); }
   else {parallel = false;}
#endif
   GetEssentialTrueDofs();
}

EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b,
                               Array2D<int> &ess_bdr):
   a(a), b(b), ess_bdr(ess_bdr), ess_tdof_list(0), symmetric(false),
   iterative_mode(false)
{
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
   if (pfes) {parallel = true; comm = pfes->GetComm(); }
   else {parallel = false;}
#endif
   GetEssentialTrueDofs();
}

void EllipticSolver::GetEssentialTrueDofs()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
      if (ess_bdr.NumRows() == 1)
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
         pfes->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list);
      }
      else
      {
         Array<int> ess_tdof_list_comp, ess_bdr_list;
         ess_bdr_list.MakeRef(ess_bdr.GetRow(0),
                              ess_bdr.NumCols());
         pfes->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, -1);
         ess_tdof_list.Append(ess_tdof_list_comp);

         for (int i=1; i<ess_bdr.NumRows(); i++)
         {
            ess_bdr_list.MakeRef(ess_bdr.GetRow(i),
                                 ess_bdr.NumCols());
            pfes->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, i - 1);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
      }
   }
   else
   {

      if (ess_bdr.NumRows() == 1)
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list);
      }
      else
      {
         Array<int> ess_tdof_list_comp, ess_bdr_list;
         ess_bdr_list.MakeRef(ess_bdr.GetRow(0),
                              ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, -1);
         ess_tdof_list.Append(ess_tdof_list_comp);

         for (int i=1; i<ess_bdr.NumRows(); i++)
         {
            ess_bdr_list.MakeRef(ess_bdr.GetRow(i),
                                 ess_bdr.NumCols());
            a.FESpace()->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, i - 1);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
      }
   }
#else
   if (ess_bdr.NumRows() == 1)
   {
      Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
      a.FESpace()->GetEssentialTrueDofs(
         ess_bdr_list, ess_tdof_list);
   }
   else
   {
      Array<int> ess_tdof_list_comp, ess_bdr_list;
      ess_bdr_list.MakeRef(ess_bdr.GetRow(0),
                           ess_bdr.NumCols());
      a.FESpace()->GetEssentialTrueDofs(
         ess_bdr_list, ess_tdof_list_comp, -1);
      ess_tdof_list.Append(ess_tdof_list_comp);

      for (int i=1; i<ess_bdr.NumRows(); i++)
      {
         ess_bdr_list.MakeRef(ess_bdr.GetRow(i),
                              ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, i - 1);
         ess_tdof_list.Append(ess_tdof_list_comp);
      }
   }
#endif
}

bool EllipticSolver::Solve(GridFunction &x, bool A_assembled,
                           bool b_Assembled)
{
   OperatorPtr A;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      A.Reset(new HypreParMatrix);
   }
#endif
   Vector B, X;
   if (!A_assembled) { a.Update(); a.Assemble(); }
   if (!b_Assembled) { b.Assemble(); }

   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
   bool converged = true;
#else
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> M;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto M_ptr = new HypreBoomerAMG(static_cast<HypreParMatrix&>(*A));
      M_ptr->SetPrintLevel(0);

      M.reset(M_ptr);
      cg.reset(new CGSolver(comm));
   }
   else
   {
      M.reset(new GSSmoother(static_cast<SparseMatrix&>(*A)));
      cg.reset(new CGSolver);
   }
#else
   M.reset(new GSSmoother(static_cast<SparseMatrix&>(*A)));
   cg.reset(new CGSolver);
#endif
   cg->SetRelTol(1e-14);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->iterative_mode = iterative_mode;
   cg->Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
   bool converged = cg->GetConverged();
#endif

   return converged;
}

bool EllipticSolver::SolveTranspose(GridFunction &x, LinearForm &f,
                                    bool A_assembled, bool f_Assembled)
{
   OperatorPtr A;
   Vector B, X;

   if (!A_assembled) { a.Assemble(); }
   if (!f_Assembled) { f.Assemble(); }

   a.FormLinearSystem(ess_tdof_list, x, f, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
   a.RecoverFEMSolution(X, *f, x);
   bool converged = true;
#else
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> M;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M.reset(new HypreBoomerAMG);
      auto M_ptr = new HypreBoomerAMG;
      M_ptr->SetPrintLevel(0);

      M.reset(M_ptr);
      cg.reset(new CGSolver((dynamic_cast<ParFiniteElementSpace*>
                             (a.FESpace()))->GetComm()));
   }
   else
   {
      M.reset(new GSSmoother((SparseMatrix&)(*A)));
      cg.reset(new CGSolver);
   }
#else
   M.reset(new GSSmoother((SparseMatrix&)(*A)));
   cg.reset(new CGSolver);
#endif
   cg->SetRelTol(1e-14);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->iterative_mode = iterative_mode;
   cg->Mult(B, X);
   a.RecoverFEMSolution(X, f, x);
   bool converged = cg->GetConverged();
#endif

   return converged;
}

HelmholtzFilter::HelmholtzFilter(FiniteElementSpace &fes, const double r_min)
   : DensityFilter(fes), eps2(r_min*r_min/(4.0/3.0))
{
   filter.reset(new GridFunction(&fes));
   filter_coeff.reset(new GridFunctionCoefficient(filter.get()));

   filter_form.reset(MakeBilinearForm(&fes));
   filter_form->AddDomainIntegrator(new DiffusionIntegrator(eps2));
   filter_form->AddDomainIntegrator(new MassIntegrator());
   filter_form->Assemble();
   filter_form->Finalize();

   ellipticSolver.reset(new EllipticSolver(*filter_form, *filter));
}



void HelmholtzFilter::SetDensity(Coefficient *rho)
{
   rho_form.reset(MakeLinearForm(&fes));
   rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho));
}

void HelmholtzFilter::UpdateFilter()
{
   MFEM_ABORT("Not implemented yet");
}

void HelmholtzFilter::UpdateGradient()
{

}
} // namespace mfem
