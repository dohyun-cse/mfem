// Obstacle problem with Shannon Entropy
// This example demonstrates CuDSS for nonlinear problems
// The Newton linear system has a fixed sparsity pattern.
// Therefore, the symbolic factorization is done once,
// and then the numerical factorization is updated at each Newton iteration.
//
// This implements PG method for the obstacle problem with Shannon entropy:
// Continuous problem:
//    min_u (grad u, grad u) / 2 + D_R(u, u^k)/alpha
// R encodes the obstacle constraint, so that dom(R) = {u | u >= obstacle}.
// The optimality condition is:
//    -Delta u - (grad R(u^k) - grad R(u)) / alpha = 0
//
// Introducing psi = grad R(u) and lambda = (grad R(u^k) - grad R(u)) / alpha,
// -Delta u -                        lambda  = 0
//        u - grad R^*(psi^k - alpha lambda) = 0
//
// Therefore, the discrete system becomes
// [K, M^T] [ u     ] = [0]
// [ M, grad R^* ] [ lambda ] = [0]
//
#include "mfem.hpp"
#include <iostream>
#include "bregman.hpp"
#include "pg.hpp"

using namespace std;
using namespace mfem;
real_t spherical_obstacle(const Vector &x);
real_t exact_solution_obstacle(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   // const char *mesh_file = "";
   int order = 1;
   int ref_levels = 0;
   const char *device_config = "cpu";
   bool visualization = true;
   bool use_cudss = false;
   real_t primal_tol = 1e-08;
   real_t dual_tol = 1e-08;
   bool debug = false;

   OptionsParser args(argc, argv);
   // args.AddOption(&mesh_file, "-m", "--mesh",
   //                "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_cudss, "-cudss", "--cudss-solver", "-no-cudss",
                  "--no-cudss-solver", "Use the cuDSS Solver.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&debug, "-db", "--debug", "-no-debug", "--no-debug",
                  "Enable or disable debug output.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();
   MemoryType mt = device.GetMemoryType();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   // Mesh mesh(mesh_file, 1, 1);
   Mesh mesh = Mesh::MakeCartesian2D(10, 10, Element::QUADRILATERAL);
   mesh.Transform([](const Vector &x, Vector &y) { y = x; y *= 2.0; y -= 1.0; });
   int dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }


   FunctionCoefficient obstacle(spherical_obstacle);
   FunctionCoefficient u_ex(exact_solution_obstacle);

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   H1_FECollection primal_fec(order+1, dim);
   L2_FECollection latent_fec(order-1, dim);
   FiniteElementSpace primal_fes(&mesh, &primal_fec);
   FiniteElementSpace latent_fes(&mesh, &latent_fec);
   cout << "Number of finite element unknowns: "
        << primal_fes.GetTrueVSize() + latent_fes.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the external boundary attributes from the mesh as essential (Dirichlet)
   //    and converting them to a list of true dofs.
   Array<int> ess_bdr(mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0);
   ess_bdr = 1;
   Array<int> ess_tdofs;
   primal_fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = primal_fes.GetTrueVSize(); // u
   offsets[2] = latent_fes.GetTrueVSize(); // lambda
   offsets.PartialSum();

   BlockVector X(offsets, mt), F(offsets, mt), Xk(offsets, mt);
   X = 0.0; F = 0.0;
   GridFunction u(&primal_fes, X.GetBlock(0));
   GridFunction u_k(&primal_fes, Xk.GetBlock(0));
   u.ProjectBdrCoefficient(u_ex, ess_bdr);
   X.SyncFromBlocks();
   GridFunction lambda(&latent_fes, X.GetBlock(1));
   GridFunction lambda_k(&latent_fes, Xk.GetBlock(1));

   GridFunctionCoefficient u_cf(&u), lambda_cf(&lambda);


   BilinearForm diffusion(&primal_fes);
   diffusion.AddDomainIntegrator(new DiffusionIntegrator);
   diffusion.Assemble();
   SparseMatrix A;
   diffusion.FormSystemMatrix(ess_tdofs, A);
   diffusion.EliminateVDofsInRHS(ess_tdofs, u, F.GetBlock(0));

   MixedBilinearForm mass(&primal_fes, &latent_fes);
   mass.AddDomainIntegrator(new MassIntegrator);
   mass.Assemble();
   SparseMatrix B;
   Array<int> dummy(0);
   mass.FormRectangularSystemMatrix(ess_tdofs, dummy, B);
   mass.EliminateTrialVDofsInRHS(ess_tdofs, u, F.GetBlock(1));
   F.SyncFromBlocks();

   ConstantCoefficient one_cf(1.0);
   CoefficientScaledLegendreFunction entropy(new Shannon, one_cf, obstacle);
   real_t alpha=1.0;
   PGOperator pg_op(A, B, latent_fes, entropy, alpha);
   pg_op.SetDebug(debug);

   std::unique_ptr<Solver> linear_solver;
   if (use_cudss && Device::Allows(Backend::CUDA_MASK))
   {
#ifdef MFEM_USE_CUDSS
      auto * cudss_solver = new CuDSSSolver;
      cudss_solver->SetReorderingReuse(true);
      linear_solver.reset(cudss_solver);
#endif
   }
   else
   {
#ifdef MFEM_USE_SUITESPARSE
      linear_solver.reset(new UMFPackSolver);
#else
      MFEM_ABORT("Either GPU or SuiteSparse must be enabled");
#endif
   }


   // 14. Send the solution by socket to a GLVis server.
   std::unique_ptr<socketstream> sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock = std::make_unique<socketstream>(vishost, visport);
      sol_sock->precision(8);
      *sol_sock << "solution\n" << mesh << u << flush;
   }
   NewtonSolver pg_solver;
   pg_solver.iterative_mode = true;
   pg_solver.SetRelTol(1e-08);
   pg_solver.SetAbsTol(1e-12);
   pg_solver.SetMaxIter(20);
   pg_solver.SetPrintLevel(0);
   pg_solver.SetSolver(*linear_solver);
   pg_solver.SetOperator(pg_op);

   real_t err0 = u.ComputeL2Error(u_ex);
   cout << "Initial L2 error: " << err0 << endl;
   for (int i=0; i<100; i++)
   {
      Xk = X;
      Xk.HostRead();
      pg_solver.Mult(F, X);
      X.HostRead();
      out << "PG iteration " << i << ", Newton it: " << pg_solver.GetNumIterations()
          << ", residual norm: " << pg_solver.GetFinalNorm() << endl;
      real_t primal_diff = u_k.ComputeL2Error(u_cf);
      real_t dual_diff = lambda_k.ComputeL1Error(lambda_cf);
      real_t primal_err = u.ComputeL2Error(u_ex);
      out << "   primal diff = " << primal_diff
          << ", primal error = " << primal_err
          << ", dual diff = " << dual_diff << endl;
      if (primal_diff < primal_tol && dual_diff < dual_tol)
      {
         break;
      }
      pg_op.ProxUpdate(lambda);
      if (visualization)
      {
         *sol_sock << "solution\n" << mesh << u << flush;
      }
   }
   real_t err = u.ComputeL2Error(u_ex);
   cout << "L2 error: " << err << endl;

   return EXIT_SUCCESS;
}

real_t spherical_obstacle(const Vector &pt)
{
   real_t x = pt(0), y = pt(1);
   real_t r = sqrt(x*x + y*y);
   real_t r0 = 0.5;
   real_t beta = 0.9;

   real_t b = r0*beta;
   real_t tmp = sqrt(r0*r0 - b*b);
   real_t B = tmp + b*b/tmp;
   real_t C = -b/tmp;

   if (r > b)
   {
      return B + r * C;
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

real_t exact_solution_obstacle(const Vector &pt)
{
   real_t x = pt(0), y = pt(1);
   real_t r = sqrt(x*x + y*y);
   real_t r0 = 0.5;
   real_t a =  0.348982574111686;
   real_t A = -0.340129705945858;

   if (r > a)
   {
      return A * log(r);
   }
   else
   {
      return sqrt(r0*r0-r*r);
   }
}
