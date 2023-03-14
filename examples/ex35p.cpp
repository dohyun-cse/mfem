//                            MFEM Example 35 - Parallel Version
//
//
// Compile with: make ex35p
//
// Sample runs:
// mpirun -np 6 ex35p -lambda 0.1 -mu 0.1
// mpirun -np 6 ex35p -r 5 -o 2 -alpha 5.0 -epsilon 0.01 -mi 50 -mf 0.5 -tol
// 1e-5 mpirun -np 8 ex35p -r 6 -o 2 -alpha 10.0 -epsilon 0.02 -mi 50 -mf 0.5
// -tol 1e-5
//
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
//
//                  minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L²(Ω)
//
//                  subject to
//
//                    -Div(r(ρ̃)Cε(u)) = f       in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, C is the elasticity tensor for an
//              isotropic linearly elastic material, ϵ > 0 is the design
//              length scale, and 0 < θ < 1 is the volume fraction.
//
//              The problem is discretized and gradients are computing using
//              finite elements [1]. The design is optimized using an entropic
//              mirror descent algorithm introduced by Keith and Surowiec [2]
//              that is tailored to the bound constraint 0 ≤ ρ ≤ 1.
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to inverse design problems and showcases how
//              to set up and solve PDE-constrained optimization problems
//              using the so-called reduced space approach.
//
//
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund,
//     O. (2011). Efficient topology optimization in MATLAB using 88 lines of
//    code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) The entropic finite element method
//     (in preparation).
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include <fstream>
#include <iostream>

#include "ex35.hpp"
#include "mfem.hpp"

/**
 * @brief Nonlinear projection of 0 < τ < 1 onto the subspace
 *        ∫_Ω τ dx = θ vol(Ω) as follows.
 *
 *        1. Compute the root of the R → R function
 *            f(c) = ∫_Ω sigmoid(inv_sigmoid(τ) + c) dx - θ vol(Ω)
 *        2. Set τ ← sigmoid(inv_sigmoid(τ) + c).
 *
 */
void projit(GridFunction &tau, double &c, LinearForm &vol_form,
            double volume_fraction, double tol = 1e-12, int max_its = 10) {
  GridFunction ftmp(tau.FESpace());
  GridFunction dftmp(tau.FESpace());
  for (int k = 0; k < max_its; k++) {
    // Compute f(c) and dfdc(c)
    for (int i = 0; i < tau.Size(); i++) {
      ftmp[i] = sigmoid(inv_sigmoid(tau[i]) + c) - volume_fraction;
      dftmp[i] = dsigmoiddx(inv_sigmoid(tau[i]) + c);
    }
    double f = vol_form(ftmp);
    double df = vol_form(dftmp);

    MPI_Allreduce(MPI_IN_PLACE, &f, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &df, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double dc = -f / df;
    c += dc;
    if (abs(dc) < tol) {
      break;
    }
  }
  tau = ftmp;
  tau += volume_fraction;
}

using namespace std;
using namespace mfem;
/**
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE
 * ---------------------------------------------------------------
 *
 *  The Lagrangian for this problem is
 *
 *          L(u,ρ,ρ̃,w,w̃) = (f,u) - (r(ρ̃) C ε(u),ε(w)) + (f,w)
 *                       - (ϵ² ∇ρ̃,∇w̃) - (ρ̃,w̃) + (ρ,w̃)
 *
 *  where
 *
 *    r(ρ̃) = ρ₀ + ρ̃³ (1 - ρ₀)       (SIMP rule)
 *
 *    ε(u) = (∇u + ∇uᵀ)/2           (symmetric gradient)
 *
 *    C e = λtr(e)I + 2μe           (isotropic material)
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ V ⊂ (H¹)ᵈ (order p)
 *     ρ ∈ L² (order p - 1)
 *     ρ̃ ∈ H¹ (order p - 1)
 *     w ∈ V  (order p)
 *     w̃ ∈ H¹ (order p - 1)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  1. Initialize density field 0 < ρ(x) < 1.
 *
 *  While not converged:
 *
 *     2. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ H¹.
 *
 *     3. Solve primal problem ∂_w L = 0; i.e.,
 *
 *      (λ(ρ̃) ∇⋅u, ∇⋅v) + (2 μ(ρ̃) ε(u), ε(v)) = (f,v)   ∀ v ∈ V,
 *
 *     where λ(ρ̃) := λ r(ρ̃) and  μ(ρ̃) := μ r(ρ̃).
 *
 *     NB. The dual problem ∂_u L = 0 is the same as the primal problem due to
 * symmetry.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-r'(ρ̃) ( λ(ρ̃) |∇⋅u|² + 2 μ(ρ̃) |ε(u)|²),v) ∀
 * v ∈ H¹.
 *
 *     5. Construct gradient G ∈ L²; i.e.,
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Mirror descent update until convergence; i.e.,
 *
 *                      ρ ← projit(sigmoid(linit(ρ) - αG)),
 *
 *     where
 *
 *          α > 0                            (step size parameter)
 *
 *          sigmoid(x) = eˣ/(1+eˣ)             (sigmoid)
 *
 *          linit(y) = ln(y) - ln(1-y)       (inverse of sigmoid)
 *
 *     and projit is a (compatible) projection operator enforcing ∫_Ω ρ dx = θ
 * vol(Ω).
 *
 *  end
 *
 */

int main(int argc, char *argv[]) {
  // 0. Initialize MPI and HYPRE.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // 1. Parse command-line options.
  int ref_levels = 4;
  int order = 2;
  bool visualization = true;
  double alpha = 1.0;
  double epsilon = 0.01;
  double mass_fraction = 0.5;
  int max_it = 1e2;
  double tol = 1e-4;
  double rho_min = 1e-6;
  double lambda = 1.0;
  double mu = 1.0;

  OptionsParser args(argc, argv);
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                 "Step length for gradient descent.");
  args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                 "epsilon phase field thickness");
  args.AddOption(&max_it, "-mi", "--max-it",
                 "Maximum number of gradient descent iterations.");
  args.AddOption(&tol, "-tol", "--tol", "Exit tolerance for ρ ");
  args.AddOption(&mass_fraction, "-mf", "--mass-fraction",
                 "Mass fraction for diffusion coefficient.");
  args.AddOption(&lambda, "-lambda", "--lambda", "Lame constant λ");
  args.AddOption(&mu, "-mu", "--mu", "Lame constant μ");
  args.AddOption(&rho_min, "-rmin", "--rho-min",
                 "Minimum of density coefficient.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");

  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  Mesh mesh = Mesh::MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL,
                                    true, 3.0, 1.0);

  int dim = mesh.Dimension();

  // 2. Set BCs.
  for (int i = 0; i < mesh.GetNBE(); i++) {
    Element *be = mesh.GetBdrElement(i);
    Array<int> vertices;
    be->GetVertices(vertices);

    double *coords1 = mesh.GetVertex(vertices[0]);
    double *coords2 = mesh.GetVertex(vertices[1]);

    Vector center(2);
    center(0) = 0.5 * (coords1[0] + coords2[0]);
    center(1) = 0.5 * (coords1[1] + coords2[1]);

    if (abs(center(0) - 0.0) < 1e-10) {
      // the left edge
      be->SetAttribute(1);
    } else {
      // all other boundaries
      be->SetAttribute(2);
    }
  }
  mesh.SetAttributes();

  // 3. Refine the mesh.
  for (int lev = 0; lev < ref_levels; lev++) {
    mesh.UniformRefinement();
  }

  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();

  // 4. Define the necessary finite element spaces on the mesh.
  H1_FECollection state_fec(order, dim);       // space for u
  H1_FECollection filter_fec(order - 1, dim);  // space for ρ̃
  L2_FECollection control_fec(order - 1, dim,
                              BasisType::Positive);  // space for ρ
  ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim);
  ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
  ParFiniteElementSpace control_fes(&pmesh, &control_fec);

  HYPRE_BigInt state_size = state_fes.GlobalTrueVSize();
  HYPRE_BigInt control_size = control_fes.GlobalTrueVSize();
  HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of state unknowns: " << state_size << endl;
    cout << "Number of filter unknowns: " << filter_size << endl;
    cout << "Number of control unknowns: " << control_size << endl;
  }

  // 5. Set the initial guess for ρ.
  ParGridFunction u(&state_fes);
  ParGridFunction rho(&control_fes);
  ParGridFunction rho_old(&control_fes);
  ParGridFunction rho_filter(&filter_fes);
  u = 0.0;
  rho_filter = 0.0;
  rho = 0.5;
  rho_old = 0.5;

  // 6. Set-up the physics solver.
  int maxat = pmesh.bdr_attributes.Max();
  Array<int> ess_bdr(maxat);
  ess_bdr = 0;
  ess_bdr[0] = 1;
  ConstantCoefficient one(1.0);
  ConstantCoefficient lambda_cf(lambda);
  ConstantCoefficient mu_cf(mu);
  LinearElasticitySolver *ElasticitySolver = new LinearElasticitySolver();
  ElasticitySolver->SetMesh(&pmesh);
  ElasticitySolver->SetOrder(state_fec.GetOrder());
  ElasticitySolver->SetupFEM();
  Vector center(2);
  center(0) = 2.9;
  center(1) = 0.5;
  Vector force(2);
  force(0) = 0.0;
  force(1) = -1.0;
  double r = 0.05;
  VolumeForceCoefficient vforce_cf(r, center, force);
  ElasticitySolver->SetRHSCoefficient(&vforce_cf);
  ElasticitySolver->SetEssentialBoundary(ess_bdr);

  // 7. Set-up the filter solver.
  ConstantCoefficient eps2_cf(epsilon * epsilon);
  DiffusionSolver *FilterSolver = new DiffusionSolver();
  FilterSolver->SetMesh(&pmesh);
  FilterSolver->SetOrder(filter_fec.GetOrder());
  FilterSolver->SetDiffusionCoefficient(&eps2_cf);
  FilterSolver->SetMassCoefficient(&one);
  Array<int> ess_bdr_filter;
  if (pmesh.bdr_attributes.Size()) {
    ess_bdr_filter.SetSize(pmesh.bdr_attributes.Max());
    ess_bdr_filter = 0;
  }
  FilterSolver->SetEssentialBoundary(ess_bdr_filter);
  FilterSolver->SetupFEM();

  ParBilinearForm mass(&control_fes);
  mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
  mass.Assemble();
  HypreParMatrix M;
  Array<int> empty;
  mass.FormSystemMatrix(empty, M);

  // 8. Define the Lagrange multiplier and gradient functions
  ParGridFunction grad(&control_fes);
  ParGridFunction w_filter(&filter_fes);

  // 9. Define some tools for later
  ConstantCoefficient zero(0.0);
  ParGridFunction onegf(&control_fes);
  onegf = 1.0;
  ParLinearForm vol_form(&control_fes);
  vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
  vol_form.Assemble();
  double domain_volume = vol_form(onegf);

  // 10. Connect to GLVis. Prepare for VisIt output.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sout_u, sout_r, sout_rho;
  if (visualization) {
    sout_u.open(vishost, visport);
    sout_rho.open(vishost, visport);
    sout_r.open(vishost, visport);
    sout_u.precision(8);
    sout_rho.precision(8);
    sout_r.precision(8);
  }

  mfem::ParaViewDataCollection paraview_dc("Elastic_compliance", &pmesh);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(order);
  paraview_dc.SetCycle(0);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetTime(0.0);
  paraview_dc.RegisterField("displacement", &u);
  paraview_dc.RegisterField("density", &rho);
  paraview_dc.RegisterField("filtered_density", &rho_filter);

  // 11. Iterate
  int step = 0;
  double c0 = 0.0;
  for (int k = 1; k < max_it; k++) {
    if (k > 1) {
      alpha *= ((double)k) / ((double)k - 1);
    }
    step++;

    if (myid == 0) {
      cout << "\nStep = " << k << endl;
    }

    // Step 1 - Filter solve
    // Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)
    GridFunctionCoefficient rho_cf(&rho);
    FilterSolver->SetRHSCoefficient(&rho_cf);
    FilterSolver->Solve();
    rho_filter = *FilterSolver->GetFEMSolution();

    // Step 2 - State solve
    // Solve (λ(ρ̃) ∇⋅u, ∇⋅v) + (2 μ(ρ̃) ε(u), ε(v)) = (f,v)
    SIMPCoefficient SIMP_cf(&rho_filter, rho_min, 1.0);
    ProductCoefficient lambda_SIMP_cf(lambda_cf, SIMP_cf);
    ProductCoefficient mu_SIMP_cf(mu_cf, SIMP_cf);
    ElasticitySolver->SetLameCoefficients(&lambda_SIMP_cf, &mu_SIMP_cf);
    ElasticitySolver->Solve();
    u = *ElasticitySolver->GetFEMSolution();

    // Step 3 - Adjoint filter solve
    // Solve (ϵ² ∇ w̃, ∇ v) + (w̃ ,v) = (-r'(ρ̃) ( λ(ρ̃) |∇⋅u|² + 2 μ(ρ̃) |ε(u)|²),v)
    StrainEnergyDensityCoefficient rhs_cf(&lambda_cf, &mu_cf, &u, &rho_filter,
                                          rho_min);
    FilterSolver->SetRHSCoefficient(&rhs_cf);
    FilterSolver->Solve();
    w_filter = *FilterSolver->GetFEMSolution();

    // Step 4 - Compute gradient
    // Solve G = M⁻¹w̃
    GridFunctionCoefficient w_cf(&w_filter);
    ParLinearForm w_rhs(&control_fes);
    w_rhs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
    w_rhs.Assemble();
    M.Mult(w_rhs, grad);

    // Step 5 - Update design variable ρ ← projit(sigmoid(linit(ρ) - αG))
    for (int i = 0; i < rho.Size(); i++) {
      rho[i] = sigmoid(inv_sigmoid(rho[i]) - alpha * grad[i]);
    }
    projit(rho, c0, vol_form, mass_fraction);

    GridFunctionCoefficient tmp(&rho_old);
    double norm_reduced_gradient = rho.ComputeL2Error(tmp) / alpha;
    rho_old = rho;

    double compliance = (*(ElasticitySolver->GetLinearForm()))(u);
    MPI_Allreduce(MPI_IN_PLACE, &compliance, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    double material_volume = vol_form(rho);
    if (myid == 0) {
      mfem::out << "norm of reduced gradient = " << norm_reduced_gradient
                << endl;
      mfem::out << "compliance = " << compliance << endl;
      mfem::out << "mass_fraction = " << material_volume / domain_volume
                << endl;
    }

    if (visualization) {
      sout_u << "parallel " << num_procs << " " << myid << "\n";
      sout_u << "solution\n"
             << pmesh << u << "window_title 'Displacement u'" << flush;

      sout_rho << "parallel " << num_procs << " " << myid << "\n";
      sout_rho << "solution\n"
               << pmesh << rho << "window_title 'Control variable ρ'" << flush;

      ParGridFunction r_gf(&control_fes);
      r_gf.ProjectCoefficient(SIMP_cf);
      sout_r << "parallel " << num_procs << " " << myid << "\n";
      sout_r << "solution\n"
             << pmesh << r_gf << "window_title 'Design density r(ρ̃)'" << flush;

      paraview_dc.SetCycle(k);
      paraview_dc.SetTime((double)k);
      paraview_dc.Save();
    }

    if (norm_reduced_gradient < tol) {
      break;
    }
  }

  return 0;
}