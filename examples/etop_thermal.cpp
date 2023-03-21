//                              MFEM Example 35
//
//
// Compile with: make ex35
//
// Sample runs:
//     ex35 -alpha 10
//     ex35 -lambda 0.1 -mu 0.1
//     ex35 -r 5 -o 2 -alpha 5.0 -epsilon 0.01 -mi 50 -mf 0.5 -tol 1e-5
//
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
//
//                  minimize ∫_Ω f⋅u dx over u ∈ H¹(Ω) and ρ ∈ L²(Ω)
//
//                  subject to
//
//                    -∇ ⋅(r(ρ̃)∇ u) = f         in Ω + BCs
//                        -ϵ²Δρ̃ + ρ̃ = sig(ψ)    in Ω + Neumann BCs
//                    ∫_Ω sig(ψ) dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, ϵ > 0 is the design length scale,
//              and 0 < θ < 1 is the volume fraction.
//
//              The problem is discretized and gradients are computing using
//              finite elements [1]. The design is optimized using an entropic
//              mirror descent algorithm introduced by Keith and Surowiec [2]
//              that is tailored to the bound constraint 0 ≤ ρ ≤ 1 by taking
//
//                    ρ = sig(ψ) = exp(ψ)/(1+exp(ψ)) = 1/(1 + exp(-ψ)).
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

#include "etop_thermal.hpp"

#include <fstream>
#include <iostream>

#include "mfem.hpp"

// Bound ψ = inv_sigmoid(ρ) to prevent blow-up (ρ -> 0 or 1)
void clip(GridFunction &psi, const double maxval) {
  for (int i = 0; i < psi.Size(); i++) {
    psi[i] = max(min(psi[i], maxval), -maxval);
  }
}

/**
 * @brief Shift ψ ↦ ψ + c where c enforces the volume constraint:
 * ∫_Ω sigmoid(ψ + c) = ∫_Ω ρ = θ vol(Ω)
 *
 * @param psi auxiliary variable, inv_sigmoid(ρ)
 * @param rho density, sigmoid(ψ)
 * @param target_volume θ vol(Ω)
 * @param tol Newton iteration tolerance
 * @param max_its Newton maximum iteration
 */
void projit(GridFunction &psi, SigmoidCoefficient &rho,
            const double target_volume, const double tol = 1e-12,
            const int max_its = 10) {
  LinearForm int_rho(psi.FESpace());
  int_rho.AddDomainIntegrator(new DomainLFIntegrator(rho));

  DerSigmoidCoefficient drho(&psi);
  LinearForm int_drho(psi.FESpace());
  int_drho.AddDomainIntegrator(new DomainLFIntegrator(drho));

  GridFunction one(psi.FESpace());
  one = 1.0;

  for (int k = 0; k < max_its; k++) {
    int_rho.Assemble();
    int_drho.Assemble();

    const double f = int_rho(one) - target_volume;
    const double df = int_drho(one);
    cout << k << ": (" << f << ", " << df << ")" << endl;

    const double dc = -f / df;
    psi += dc;
    if (abs(dc) < tol) {
      break;
    }
  }
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
 *          L(u,ρ,ρ̃,w,w̃) = (f,u) - (r(ρ̃) ∇ u, ∇ w) - (f,w)
 *                       - (ϵ² ∇ ρ̃,∇ w̃) - (ρ̃,w̃) + (ρ,w̃)
 *                       + ((1-u) log((1-u)/(1-u_k)) + (u-u_k), 1)
 *                       + ((1-ρ) log((1-ρ)/(1-ρ_k)) + (ρ-ρ_k), 1)
 *                       + (ρ log(ρ/ρ_k) - (ρ-ρ_k), 1)
 *
 *  where
 *
 *    r(ρ̃) = ρ₀ + ρ̃³ (1 - ρ₀)       (SIMP rule)
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ Vh (H¹, order p)
 *     ψ ∈ Dl (L², order p - 1)
 *     ρ̃ ∈ Vl (H¹, order p - 1)
 *     w ∈ Vh (H¹, order p)
 *     w̃ ∈ Vl (H¹, order p - 1)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  0. Initialize inverse sigmoid density field ψ = inv_sigmoid(ρ).
 *
 *  While not converged:
 *
 *     1. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (sig(ψ),v)   ∀ v ∈ H¹.
 *
 *     2. Solve primal problem ∂_w L = 0; i.e.,
 *
 *                  (r(ρ̃) ∇ u, ∇ v) = (f,v)   ∀ v ∈ V.
 *
 *     3. Solve dual problem ∂_u L = 0; i.e.,
 *
 *                  (λ(ρ̃) ∇ w, ∇ v) = - (f,v) - (log(u/u_k), v)   ∀ v ∈ V.
 *
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ w̃, ∇ v ) + (w̃,v) = -(r'(ρ̃)∇ u ⋅ ∇ w, v)  ∀ v ∈ H¹.
 *
 *
 *     5. Construct gradient G ∈ L²; i.e.,
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Mirror descent update until convergence; i.e.,
 *
 *                           ψ ← projit(clip(ψ - αG)),
 *
 *     where
 *
 *          α > 0                            (step size parameter)
 *
 *          clip: ψ(x) -> ψ̄(x)               (strong bound (-max_val, max_val))
 *
 *     and projit is a (compatible) projection operator ψ ↦ ψ + c,
 *     enforcing the volume constraint ∫_Ω sigmoid(ψ + c) dx = θ vol(Ω).
 *
 *  end
 *
 */

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  int ref_levels = 4;
  int order = 2;
  bool visualization = true;
  double alpha = 1.0;
  double epsilon = 0.01;
  double mass_fraction = 0.5;
  int max_it = 1e2;
  double tol = 1e-6;
  double rho_min = 1e-6;
  double lambda = 1.0;
  double mu = 1.0;
  double psi_maxval = 100;
  double update_epsilon = 1.e-08;
  int vis_refine_levels = 2;

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
  args.AddOption(&vis_refine_levels, "-vr", "--visualization-refinement",
                 "The number of refinement for GLVis visualization.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

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

  // 4. Define the necessary finite element spaces on the mesh.
  H1_FECollection H1high(order, dim);     // space for u, w
  H1_FECollection H1low(order - 1, dim);  // space for ρ̃, w̃
  L2_FECollection L2low(order - 1, dim);  // space for ψ
  FiniteElementSpace Vh(&mesh, &H1high);
  FiniteElementSpace Vl(&mesh, &H1low);
  FiniteElementSpace Dl(&mesh, &L2low);

  int state_size = Vh.GetTrueVSize();
  int control_size = Dl.GetTrueVSize();
  int filter_size = Vl.GetTrueVSize();
  cout << "Number of state unknowns: " << state_size << endl;
  cout << "Number of filter unknowns: " << filter_size << endl;
  cout << "Number of control unknowns: " << control_size << endl;

  // 5. Set the initial guess for ρ.
  GridFunction u(&Vh);
  GridFunction u_old(&Vh);
  GridFunction w(&Vh);
  GridFunction psi(&Dl);         // inv_sigmoid(ρ)
  GridFunction psi_old(&Dl);     // inv_sigmoid(ρ_old)
  GridFunction rho_filter(&Vl);  // ρ̃
  GridFunction w_filter(&Vl);
  u = 0.0;
  rho_filter = 0.0;
  psi = inv_sigmoid(mass_fraction);  // equivalent to ρ = mass_fraction
  psi_old = inv_sigmoid(mass_fraction);

  SigmoidCoefficient rho(&psi);  // ρ = sigmoid(ψ)
  SigmoidCoefficient rho_old(&psi_old);

  LogDiffCoefficient logu_by_uk(&u, &u_old);

  ThermalEnergyCoefficient drdrho_gradu_gradv(&u, &w, &rho_filter, 3, 1.e-12,
                                              1.0);

  int maxat = mesh.bdr_attributes.Max();
  Array<int> ess_bdr(maxat);
  ess_bdr = 0;
  ess_bdr[0] = 1;
  ConstantCoefficient one(1.0);

  // // 6. Set-up the physics solver.
  // int maxat = mesh.bdr_attributes.Max();
  // Array<int> ess_bdr(maxat);
  // ess_bdr = 0;
  // ess_bdr[0] = 1;
  // ConstantCoefficient one(1.0);
  // ConstantCoefficient lambda_cf(lambda);
  // ConstantCoefficient mu_cf(mu);
  // LinearElasticitySolver *ElasticitySolver = new LinearElasticitySolver();
  // ElasticitySolver->SetMesh(&mesh);
  // ElasticitySolver->SetOrder(H1high.GetOrder());
  // ElasticitySolver->SetupFEM();
  // Vector center(2);
  // center(0) = 2.9;
  // center(1) = 0.5;
  // Vector force(2);
  // force(0) = 0.0;
  // force(1) = -1.0;
  // double r = 0.05;
  // VolumeForceCoefficient vforce_cf(r, center, force);
  // ElasticitySolver->SetRHSCoefficient(&vforce_cf);
  // ElasticitySolver->SetEssentialBoundary(ess_bdr);

  // // 7. Set-up the filter solver.
  // ConstantCoefficient eps2_cf(epsilon * epsilon);
  // DiffusionSolver *FilterSolver = new DiffusionSolver();
  // FilterSolver->SetMesh(&mesh);
  // FilterSolver->SetOrder(H1low.GetOrder());
  // FilterSolver->SetDiffusionCoefficient(&eps2_cf);
  // FilterSolver->SetMassCoefficient(&one);
  // Array<int> ess_bdr_filter;
  // if (mesh.bdr_attributes.Size()) {
  //   ess_bdr_filter.SetSize(mesh.bdr_attributes.Max());
  //   ess_bdr_filter = 0;
  // }
  // FilterSolver->SetEssentialBoundary(ess_bdr_filter);
  // FilterSolver->SetupFEM();

  // BilinearForm inv_mass(&Dl);
  // inv_mass.AddDomainIntegrator(new InverseIntegrator(new
  // MassIntegrator(one))); inv_mass.Assemble();

  // // 8. Define the Lagrange multiplier and gradient functions
  // GridFunction grad(&Dl);
  // GridFunction w_filter(&Vl);

  // // 9. Define some tools for later

  // ConstantCoefficient zero(0.0);     // zero coefficient
  // GridFunction onegf(&Dl);  // one grid function
  // onegf = 1.0;

  // double domain_volume = 0.0;
  // for (int i = 0; i < mesh.GetNE(); i++) {
  //   domain_volume += mesh.GetElementVolume(i);
  // }

  // // 10. Connect to GLVis. Prepare for VisIt output.
  // char vishost[] = "localhost";
  // int visport = 19916;
  // socketstream sout_u, sout_r, sout_rho;
  // if (visualization) {
  //   sout_u.open(vishost, visport);
  //   sout_rho.open(vishost, visport);
  //   sout_r.open(vishost, visport);
  //   sout_u.precision(8);
  //   sout_rho.precision(8);
  //   sout_r.precision(8);
  // }

  // // mfem::ParaViewDataCollection paraview_dc("Elastic_compliance", &mesh);
  // // paraview_dc.SetPrefixPath("ParaView");
  // // paraview_dc.SetLevelsOfDetail(order);
  // // paraview_dc.SetCycle(0);
  // // paraview_dc.SetDataFormat(VTKFormat::BINARY);
  // // paraview_dc.SetHighOrderOutput(true);
  // // paraview_dc.SetTime(0.0);
  // // paraview_dc.RegisterField("displacement", &u);
  // // paraview_dc.RegisterField("density", &rho);
  // // paraview_dc.RegisterField("filtered_density", &rho_filter);

  // // 11. Iterate
  // int step = 0;
  // double c0 = 0.0;

  // LinearForm int_rho(&Dl);
  // int_rho.AddDomainIntegrator(new DomainLFIntegrator(rho));

  // for (int k = 1; k < max_it; k++) {
  //   if (k > 1) {
  //     alpha *= ((double)k) / (k - 1.0);
  //   }
  //   step++;

  //   cout << "\nStep = " << k << endl;

  //   // Step 1 - Filter solve
  //   // Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v) = (sigmoid(ψ), v)
  //   // GridFunctionCoefficient psi_cf(&psi);
  //   FilterSolver->SetRHSCoefficient(&rho);
  //   FilterSolver->Solve();
  //   rho_filter = *FilterSolver->GetFEMSolution();

  //   // Step 2 - State solve
  //   // Solve (λ(ρ̃) ∇⋅u, ∇⋅v) + (2 μ(ρ̃) ε(u), ε(v)) = (f,v)
  //   SIMPCoefficient SIMP_cf(&rho_filter, rho_min, 1.0);
  //   ProductCoefficient lambda_SIMP_cf(lambda_cf, SIMP_cf);
  //   ProductCoefficient mu_SIMP_cf(mu_cf, SIMP_cf);
  //   ElasticitySolver->SetLameCoefficients(&lambda_SIMP_cf, &mu_SIMP_cf);
  //   ElasticitySolver->Solve();
  //   u = *ElasticitySolver->GetFEMSolution();

  //   // Step 3 - Adjoint filter solve
  //   // Solve (ϵ² ∇ w̃, ∇ v) + (w̃ ,v) = (-r'(ρ̃) ( λ(ρ̃) |∇⋅u|² + 2 μ(ρ̃)
  //   |ε(u)|²),v) StrainEnergyDensityCoefficient rhs_cf(&lambda_cf, &mu_cf, &u,
  //   &rho_filter,
  //                                         rho_min, 1.0);
  //   FilterSolver->SetRHSCoefficient(&rhs_cf);
  //   FilterSolver->Solve();
  //   w_filter = *FilterSolver->GetFEMSolution();

  //   // Step 4 - Compute gradient
  //   // Solve G = M⁻¹w̃
  //   GridFunctionCoefficient w_cf(&w_filter);
  //   LinearForm w_rhs(&Dl);
  //   w_rhs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
  //   w_rhs.Assemble();
  //   inv_mass.Mult(w_rhs, grad);

  //   // Step 5 - Update design variable ψ ← projit(clip(ψ - αG))
  //   grad *= alpha;
  //   psi *= (1 - alpha * update_epsilon);
  //   psi -= grad;
  //   clip(psi, psi_maxval);
  //   projit(psi, rho, mass_fraction * domain_volume);

  //   cout << "psi in (" << psi.Min() << ", " << psi.Max() << ")" << endl;

  //   double norm_reduced_gradient;
  //   {
  //     GridFunction rho_gf(&Dl);
  //     rho_gf.ProjectCoefficient(rho);

  //     SigmoidCoefficient rho_old(&psi_old);
  //     GridFunction err(&Dl);
  //     err.ProjectCoefficient(rho_old);
  //     err -= rho_gf;

  //     norm_reduced_gradient = err.ComputeL2Error(zero);
  //     psi_old = psi;
  //   }

  //   double compliance = (*(ElasticitySolver->GetLinearForm()))(u);
  //   int_rho.Assemble();
  //   double material_volume = int_rho(onegf);
  //   // double material_volume = vol_form(rho);
  //   mfem::out << "norm of reduced gradient = " << norm_reduced_gradient <<
  //   endl; mfem::out << "compliance = " << compliance << endl; mfem::out <<
  //   "mass_fraction = " << material_volume / domain_volume << endl;

  //   if (visualization) {
  //     sout_u << "solution\n"
  //            << mesh << u << "window_title 'Displacement u'" << flush;

  //     GridFunction rho_gf(&Dl);
  //     rho_gf.ProjectCoefficient(rho);
  //     sout_rho << "solution\n"
  //              << mesh << rho_gf << "window_title 'Control variable ρ'"
  //              << flush;

  //     GridFunction r_gf(&Dl);
  //     r_gf.ProjectCoefficient(SIMP_cf);
  //     sout_r << "solution\n"
  //            << mesh << r_gf << "window_title 'Design density r(ρ̃)'" <<
  //            flush;

  //     // paraview_dc.SetCycle(k);
  //     // paraview_dc.SetTime((double)k);
  //     // paraview_dc.Save();
  //   }

  //   if (norm_reduced_gradient < tol) {
  //     break;
  //   }
  // }

  // // Final visualization on a refined mesh
  // for (int i = 0; i < vis_refine_levels; i++) {
  //   mesh.UniformRefinement();
  //   Dl.Update();
  //   psi.Update();
  //   Dl.UpdatesFinished();
  //   mesh.UniformRefinement();
  //   Dl.Update();
  //   psi.Update();
  //   Dl.UpdatesFinished();
  // }

  // // Make a finite element space
  // // Here, we use Q1 basis function with vertex to prevent overshoot
  // L2_FECollection display_fec(1, 2, BasisType::GaussLobatto);
  // FiniteElementSpace display_fes(&mesh, &display_fec);
  // GridFunction rho_gf(&display_fes);
  // rho_gf.ProjectCoefficient(rho);

  // socketstream sout_refine_rho;
  // sout_refine_rho.open(vishost, visport);

  // sout_refine_rho << "solution\n"
  //                 << mesh << rho_gf
  //                 << "window_title 'Control variable ρ - refined'" << flush;

  return 0;
}