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
#include "bregman.hpp"
#include "mfem.hpp"
#include "pg.hpp"
#include <iostream>

using namespace std;
using namespace mfem;
real_t spherical_obstacle(const Vector &x);
real_t exact_solution_obstacle(const Vector &x);

int main(int argc, char *argv[]) {
  // 1. Initialize MPI and HYPRE.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  MPI_Comm comm = MPI_COMM_WORLD;
  Hypre::Init();
  OutStream pout(std::cout);
  if (myid != 0) {
    pout.Disable();
  }

  // 1. Parse command-line options.
  // const char *mesh_file = "";
  int order = 1;
  int ser_ref_levels = 0;
  int par_ref_levels = 0;
  const char *device_config = "cpu";
  bool visualization = false;
  bool use_cudss = false;
  real_t primal_tol = 1e-08;
  real_t dual_tol = 1e-08;
  bool debug = false;
  real_t alpha = 1.0;
  real_t grow_factor = 1.5;

  OptionsParser args(argc, argv);
  // args.AddOption(&mesh_file, "-m", "--mesh",
  //                "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&ser_ref_levels, "-sr", "--ser-refine",
                 "Number of times to serially refine the mesh uniformly.");
  args.AddOption(&par_ref_levels, "-pr", "--par-refine",
                 "Number of times to parallely refine the mesh uniformly.");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.AddOption(&use_cudss, "-cudss", "--cudss-solver", "-no-cudss",
                 "--no-cudss-solver", "Use the cuDSS Solver.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&debug, "-db", "--debug", "-no-debug", "--no-debug",
                 "Enable or disable debug output.");
  args.AddOption(&alpha, "-a", "--alpha", "PG alpha parameter.");
  args.AddOption(&grow_factor, "-ag", "--alpha-growth-factor",
                 "PG alpha grow factor.");
  args.ParseCheck();

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(device_config);
  device.Print(pout);
  MemoryType mt = device.GetMemoryType();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  Mesh ser_mesh = Mesh::MakeCartesian2D(10, 10, Element::QUADRILATERAL);
  ser_mesh.Transform([](const Vector &x, Vector &y) {
    y = x;
    y *= 2.0;
    y -= 1.0;
  });
  for (int l = 0; l < ser_ref_levels; l++) {
    ser_mesh.UniformRefinement();
  }

  ParMesh mesh(comm, ser_mesh);
  ser_mesh.Clear();
  for (int l = 0; l < par_ref_levels; l++) {
    mesh.UniformRefinement();
  }

  int dim = mesh.Dimension();

  FunctionCoefficient obstacle(spherical_obstacle);
  FunctionCoefficient u_ex(exact_solution_obstacle);

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  H1_FECollection primal_fec(order + 1, dim);
  L2_FECollection latent_fec(order - 1, dim);
  ParFiniteElementSpace primal_fes(&mesh, &primal_fec);
  ParFiniteElementSpace latent_fes(&mesh, &latent_fec);

  auto dofs = primal_fes.GlobalTrueVSize() + latent_fes.GlobalTrueVSize();
  pout << "Number of finite element unknowns: " << dofs << endl;

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the external boundary attributes from the mesh as essential (Dirichlet)
  //    and converting them to a list of true dofs.
  Array<int> ess_bdr(mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max()
                                                : 0);
  ess_bdr = 1;
  Array<int> ess_tdofs;
  primal_fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

  Array<int> loffsets(3);
  loffsets[0] = 0;
  loffsets[1] = primal_fes.GetVSize(); // u
  loffsets[2] = latent_fes.GetVSize(); // lambda
  loffsets.PartialSum();

  Array<int> toffsets(3);
  toffsets[0] = 0;
  toffsets[1] = primal_fes.GetTrueVSize(); // u
  toffsets[2] = latent_fes.GetTrueVSize(); // lambda
  toffsets.PartialSum();

  BlockVector X(loffsets, mt), F(loffsets, mt), Xk(loffsets, mt);
  BlockVector tX(toffsets, mt), tF(toffsets, mt);
  X = 0.0;
  F = 0.0;
  tX = 0.0;
  tF = 0.0;
  X.SyncToBlocks();
  F.SyncToBlocks();
  Xk.SyncToBlocks();
  tX.SyncToBlocks();
  tF.SyncToBlocks();

  ParGridFunction u(&primal_fes, X.GetBlock(0));
  ParGridFunction u_k(&primal_fes, Xk.GetBlock(0));
  u.ProjectBdrCoefficient(u_ex, ess_bdr);
  X.SyncFromBlocks();
  u.GetTrueDofs(tX.GetBlock(0));
  tX.SyncFromBlocks();
  ParGridFunction lambda(&latent_fes, X.GetBlock(1));
  ParGridFunction lambda_k(&latent_fes, Xk.GetBlock(1));

  GridFunctionCoefficient u_cf(&u), lambda_cf(&lambda);

  ParBilinearForm diffusion(&primal_fes);
  diffusion.AddDomainIntegrator(new DiffusionIntegrator);
  diffusion.Assemble();
  OperatorHandle A_h;
  diffusion.FormSystemMatrix(ess_tdofs, A_h);
  diffusion.ParallelEliminateTDofsInRHS(ess_tdofs, tX.GetBlock(0),
                                        tF.GetBlock(0));

  ParMixedBilinearForm mass(&primal_fes, &latent_fes);
  mass.AddDomainIntegrator(new MassIntegrator);
  mass.Assemble();
  OperatorHandle B_h;
  Array<int> dummy(0);
  mass.FormRectangularSystemMatrix(ess_tdofs, dummy, B_h);
  mass.ParallelEliminateTrialTDofsInRHS(ess_tdofs, tX.GetBlock(0),
                                        tF.GetBlock(1));
  tX.SyncFromBlocks();
  tF.SyncFromBlocks();

  // tX.HostRead();
  // tF.HostRead();
  // out << "primal_true:     " << tX.GetBlock(0).Norml2() << std::endl;
  // out << "dual_true:       " << tX.GetBlock(1).Norml2() << std::endl;
  // out << "primal_rhs_true: " << tF.GetBlock(0).Norml2() << std::endl;
  // out << "dual_rhs_true:   " << tF.GetBlock(1).Norml2() << std::endl;

  ConstantCoefficient one_cf(1.0);
  CoefficientScaledLegendreFunction entropy(new Shannon, one_cf, obstacle);
  PGOperator pg_op(*A_h.As<HypreParMatrix>(), *B_h.As<HypreParMatrix>(),
                   latent_fes, entropy, alpha);
  pg_op.SetDebug(debug);

  std::unique_ptr<Solver> linear_solver;
  if (use_cudss && Device::Allows(Backend::CUDA_MASK)) {
#ifdef MFEM_USE_CUDSS
    auto *cudss_solver = new CuDSSSolver(comm);
    cudss_solver->SetReorderingReuse(true);
    cudss_solver->SetMatrixSymType(CuDSSSolver::MatType::NONSYMMETRIC);
    linear_solver.reset(cudss_solver);
#endif
  } else {
#ifdef MFEM_USE_PETSC
    linear_solver.reset(new MUMPSSolver(comm));
#else
    MFEM_ABORT("Either GPU or SuiteSparse must be enabled");
#endif
  }

  // 14. Send the solution by socket to a GLVis server.
  std::unique_ptr<socketstream> sol_sock;
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    sol_sock = std::make_unique<socketstream>(vishost, visport);
    *sol_sock << "parallel " << num_procs << " " << myid << "\n";
    sol_sock->precision(8);
    *sol_sock << "solution\n" << mesh << u << flush;
  }
  NewtonSolver pg_solver(comm);
  pg_solver.iterative_mode = true;
  pg_solver.SetRelTol(1e-08);
  pg_solver.SetAbsTol(1e-12);
  pg_solver.SetMaxIter(20);
  pg_solver.SetPrintLevel(NewtonSolver::PrintLevel().Iterations());
  pg_solver.SetSolver(*linear_solver);
  pg_solver.SetOperator(pg_op);

  real_t err0 = u.ComputeL2Error(u_ex);
  pout << "Initial L2 error: " << err0 << endl;
  for (int i = 0; i < 100; i++) {
    Xk = X;
    Xk.HostRead();
    pg_solver.Mult(tF, tX);
    tX.HostRead();
    u.SetFromTrueDofs(tX.GetBlock(0));
    lambda.SetFromTrueDofs(tX.GetBlock(1));
    X.SyncFromBlocks();
    X.HostRead();
    pout << "PG iteration " << i
         << ", Newton it: " << pg_solver.GetNumIterations()
         << ", residual norm: " << pg_solver.GetFinalNorm() << endl;
    real_t primal_diff = u_k.ComputeL2Error(u_cf);
    real_t dual_diff = lambda_k.ComputeL1Error(lambda_cf);
    real_t primal_err = u.ComputeL2Error(u_ex);
    pout << "   primal diff = " << primal_diff
         << ", primal error = " << primal_err << ", dual diff = " << dual_diff
         << endl;
    if (primal_diff < primal_tol && dual_diff < dual_tol) {
      break;
    }
    pg_op.ProxUpdate(lambda);
    if (visualization) {
      *sol_sock << "parallel " << num_procs << " " << myid << "\n";
      *sol_sock << "solution\n" << mesh << u << flush;
    }
    alpha *= grow_factor;
  }
  real_t err = u.ComputeL2Error(u_ex);
  pout << "L2 error: " << err << endl;

  return EXIT_SUCCESS;
}

real_t spherical_obstacle(const Vector &pt) {
  real_t x = pt(0), y = pt(1);
  real_t r = sqrt(x * x + y * y);
  real_t r0 = 0.5;
  real_t beta = 0.9;

  real_t b = r0 * beta;
  real_t tmp = sqrt(r0 * r0 - b * b);
  real_t B = tmp + b * b / tmp;
  real_t C = -b / tmp;

  if (r > b) {
    return B + r * C;
  } else {
    return sqrt(r0 * r0 - r * r);
  }
}

real_t exact_solution_obstacle(const Vector &pt) {
  real_t x = pt(0), y = pt(1);
  real_t r = sqrt(x * x + y * y);
  real_t r0 = 0.5;
  real_t a = 0.348982574111686;
  real_t A = -0.340129705945858;

  if (r > a) {
    return A * log(r);
  } else {
    return sqrt(r0 * r0 - r * r);
  }
}
