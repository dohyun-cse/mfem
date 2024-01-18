//                              MFEM Example 37
//
//
// Compile with: make ex37
//
// Sample runs:
//     ex37 -alpha 10
//     ex37 -alpha 10 -pv
//     ex37 -lambda 0.1 -mu 0.1
//     ex37 -o 2 -alpha 5.0 -mi 50 -vf 0.4 -ntol 1e-5
//     ex37 -r 6 -o 1 -alpha 25.0 -epsilon 0.02 -mi 50 -ntol 1e-5
//
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
//
//                  minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L¹(Ω)
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
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//    (2011). Efficient topology optimization in MATLAB using 88 lines of
//    code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "topopt.hpp"
#include "helper.hpp"

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
 *  NOTE: The Lame parameters can be computed from Young's modulus E
 *        and Poisson's ratio ν as follows:
 *
 *             λ = E ν/((1+ν)(1-2ν)),      μ = E/(2(1+ν))
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ V ⊂ (H¹)ᵈ (order p)
 *     ψ ∈ L² (order p - 1), ρ = sigmoid(ψ)
 *     ρ̃ ∈ H¹ (order p)
 *     w ∈ V  (order p)
 *     w̃ ∈ H¹ (order p)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  1. Initialize ψ = inv_sigmoid(vol_fraction) so that ∫ sigmoid(ψ) = θ vol(Ω)
 *
 *  While not converged:
 *
 *     2. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ H¹.
 *
 *     3. Solve primal problem ∂_w L = 0; i.e.,
 *
 *      (λ r(ρ̃) ∇⋅u, ∇⋅v) + (2 μ r(ρ̃) ε(u), ε(v)) = (f,v)   ∀ v ∈ V.
 *
 *     NB. The dual problem ∂_u L = 0 is the negative of the primal problem due to symmetry.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-r'(ρ̃) ( λ |∇⋅u|² + 2 μ |ε(u)|²),v)   ∀ v ∈ H¹.
 *
 *     5. Project the gradient onto the discrete latent space; i.e., solve
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Bregman proximal gradient update; i.e.,
 *
 *                            ψ ← ψ - αG + c,
 *
 *     where α > 0 is a step size parameter and c ∈ R is a constant ensuring
 *
 *                     ∫_Ω sigmoid(ψ - αG + c) dx = θ vol(Ω).
 *
 *  end
 *
 */

enum Problem
{
   Cantilever,
   MBB,
   LBracket,
   Cantilever3,
   Torsion3
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   int seq_ref_levels = 0;
   int par_ref_levels = 6;
   int order = 1;
   double epsilon = 2e-2;
   double vol_fraction = 0.5;
   int max_it = 2e2;
   double rho_min = 1.;
   double exponent = 3.0;
   double lambda = 1.0;
   double mu = 1.0;
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = true;
   bool paraview = true;

   ostringstream solfile, solfile2, meshfile;

   int problem = Problem::Cantilever;

   OptionsParser args(argc, argv);
   args.AddOption(&seq_ref_levels, "-sr", "--seq-refine",
                  "Number of times to refine the sequential mesh uniformly.");
   args.AddOption(&par_ref_levels, "-pr", "--par-refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) Cantilever, 1) MBB, 2) LBracket.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lamé constant λ.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lamé constant μ.");
   args.AddOption(&rho_min, "-rmin", "--psi-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();


   Mesh mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   Vector center(2), force(2);
   double r = 0.05;
   std::unique_ptr<VectorCoefficient> vforce_cf;
   std::unique_ptr<VectorCoefficient> torsion_cf(nullptr);
   string mesh_file;
   switch (problem)
   {
      case Problem::Cantilever:
         mesh = mesh.MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL, true, 3.0,
                                     1.0);
         ess_bdr.SetSize(3, 4);
         ess_bdr_filter.SetSize(4);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(2, 3) = 1;
         center(0) = 2.9; center(1) = 0.5;
         force(0) = 0.0; force(1) = -1.0;
         vforce_cf.reset(new VolumeForceCoefficient(r,center,force));
         solfile << "Cantilever-";
         solfile2 << "Cantilever-";
         meshfile << "Cantilever";
         break;
      case Problem::LBracket:
         mesh_file = "../../data/lbracket_square.mesh";
         seq_ref_levels--;
         mesh = mesh.LoadFromFile(mesh_file);
         ess_bdr.SetSize(3, 6);
         ess_bdr_filter.SetSize(6);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(2, 4) = 1;
         center(0) = 0.95; center(1) = 0.35;
         force(0) = 0.0; force(1) = -1.0;
         vforce_cf.reset(new VolumeForceCoefficient(r,center,force));
         solfile << "LBracket-";
         solfile2 << "LBracket-";
         meshfile << "LBracket";
         break;
      case Problem::MBB:
         mesh = mesh.MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL, true, 3.0,
                                     1.0);
         ess_bdr.SetSize(3, 5);
         ess_bdr_filter.SetSize(5);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 0) = 1; // left : y-roller -> x fixed
         ess_bdr(1, 4) = 1; // right-bottom : x-roller -> y fixed
         center(0) = 0.05; center(1) = 0.95;
         force(0) = 0.0; force(1) = -1.0;
         vforce_cf.reset(new VolumeForceCoefficient(r,center,force));
         solfile << "MBB-";
         solfile2 << "MBB-";
         meshfile << "MBB";
         break;

      case Problem::Cantilever3:
         // [1: bottom, 2: front, 3: right, 4: back, 5: left, 6: top]
         mesh = mesh.MakeCartesian3D(2, 1, 1, mfem::Element::Type::HEXAHEDRON, 2.0, 1.0,
                                     1.0);
         ess_bdr.SetSize(4, 6);
         ess_bdr_filter.SetSize(6);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(3, 4) = 1;

         vol_fraction = 0.12;

         center.SetSize(3); force.SetSize(3);
         center(0) = 1.9; center(1) = 0.1; center(2) = 0.25;
         force(0) = 0.0; force(1) = 0.0; force(2) = -1.0;
         vforce_cf.reset(new LineVolumeForceCoefficient(r,center,force,1));
         solfile << "Cantilever3-";
         solfile2 << "Cantilever3-";
         meshfile << "Cantilever3";
         break;

      case Problem::Torsion3:
         // [1: bottom, 2: front, 3: right, 4: back, 5: left, 6: top]

         r = 0.2;
         mesh = mesh.MakeCartesian3D(6, 5, 5, mfem::Element::Type::HEXAHEDRON, 1.2, 1.0,
                                     1.0);
         ess_bdr.SetSize(4, 7);
         ess_bdr_filter.SetSize(7);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(4, 6) = 1;

         center.SetSize(3); force.SetSize(3);
         vol_fraction = 0.1;
         epsilon = 0.025;
         force = 0.0;

         vforce_cf.reset(new VectorConstantCoefficient(force));
         center(0) = 1.2; center(1) = 0.5; center(2) = 0.5;
         torsion_cf.reset(new VectorFunctionCoefficient(3, [r, center](const Vector &x,
                                                                       Vector &force)
         {
            force = 0.0;
            if (x.DistanceTo(center) < r)
            {
               force[1] = (x[2] - 0.5); force[2] = -(x[1] - 0.5);
            }
         }));
         solfile << "Torsion3-";
         solfile2 << "Torsion3-";
         meshfile << "Torsion3";
         break;

      default:
         mfem_error("Undefined problem.");
   }

   int dim = mesh.Dimension();
   double h = std::pow(mesh.GetElementVolume(0), 1.0 / dim);
   // 3. Refine the mesh.
   for (int lev = 0; lev < seq_ref_levels; lev++)
   {
      mesh.UniformRefinement();
      h *= 0.5;
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
      h *= 0.5;
   }
   switch (problem)
   {
      case Problem::MBB:
      {
         MarkBoundary(pmesh, [](const Vector &fc) {return ((fc(0) > (3 - std::pow(2, -5))) && (fc(1) < 1e-10)); },
         5);
         break;
      }
      case Problem::Torsion3:
      {
         // left center: Dirichlet
         MarkBoundary(pmesh, [r](const Vector &fc) {return (fc.Norml2() < r); }, 7);
         break;
      }
   }
   pmesh.SetAttributes();

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   ParFiniteElementSpace state_fes(&pmesh, &state_fec,dim);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   int state_size = state_fes.GlobalTrueVSize();
   int control_size = control_fes.GlobalTrueVSize();
   int filter_size = filter_fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "Number of state unknowns: " << state_size << std::endl;
      mfem::out << "Number of filter unknowns: " << filter_size << std::endl;
      mfem::out << "Number of control unknowns: " << control_size << std::endl;
   }

   // 5. Set the initial guess for ρ.
   SIMPProjector simp_rule(exponent, rho_min);
   HelmholtzFilter filter(filter_fes, epsilon, ess_bdr_filter);
   LatentDesignDensity density(control_fes, filter, filter_fes, vol_fraction);

   ConstantCoefficient lambda_cf(lambda), mu_cf(mu);
   ParametrizedElasticityEquation elasticity(state_fes,
                                             density.GetFilteredDensity(), simp_rule, lambda_cf, mu_cf, *vforce_cf, ess_bdr);
   if (torsion_cf.get())
   {
      elasticity.GetLinearForm().AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(
                                                         *torsion_cf));
      elasticity.GetLinearForm().Assemble();
   }

   TopOptProblem optprob(elasticity.GetLinearForm(), elasticity, density, false,
                         true);


   meshfile << "-" << par_ref_levels;
   solfile << par_ref_levels << "-";
   solfile2 << par_ref_levels << "-";
   solfile << "PMD-0." << setfill('0') << setw(6) << myid;
   solfile2 << "PMD-f." << setfill('0') << setw(6) << myid;
   meshfile << "." << setfill('0') << setw(6) << myid;

   ParGridFunction &u = *dynamic_cast<ParGridFunction*>(&optprob.GetState());
   ParGridFunction &rho_filter = *dynamic_cast<ParGridFunction*>
                                 (&density.GetFilteredDensity());
   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_SIMP, sout_r;
   std::unique_ptr<ParGridFunction> designDensity_gf, rho_gf;
   if (glvis_visualization)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      designDensity_gf.reset(new ParGridFunction(&filter_fes));
      rho_gf.reset(new ParGridFunction(&filter_fes));
      designDensity_gf->ProjectCoefficient(simp_rule.GetPhysicalDensity(
                                              density.GetFilteredDensity()));
      rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
      sout_SIMP.open(vishost, visport);
      if (sout_SIMP.is_open())
      {
         sout_SIMP << "parallel " << num_procs << " " << myid << "\n";
         sout_SIMP.precision(8);
         sout_SIMP << "solution\n" << pmesh << *designDensity_gf
                   << "window_title 'Design density r(ρ̃) - PMD "
                   << problem << "'\n"
                   << "keys Rjl***************\n"
                   << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }
      sout_r.open(vishost, visport);
      if (sout_r.is_open())
      {
         sout_r << "parallel " << num_procs << " " << myid << "\n";
         sout_r.precision(8);
         sout_r << "solution\n" << pmesh << *rho_gf
                << "window_title 'Raw density ρ - PMD "
                << problem << "'\n"
                << "keys Rjl***************\n"
                << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }
   }
   std::unique_ptr<ParaViewDataCollection> pd;
   if (paraview)
   {
      pd.reset(new ParaViewDataCollection("TopPMD", &pmesh));
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("state", &u);
      pd->RegisterField("psi", &density.GetGridFunction());
      pd->RegisterField("frho", &rho_filter);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0);
      pd->Save();
   }

   mfem::ParaViewDataCollection paraview_dc("ex37", &pmesh);
   ofstream mesh_ofs(meshfile.str().c_str());
   mesh_ofs.precision(8);
   pmesh.Print(mesh_ofs);
   // 11. Iterate
   ParGridFunction &grad(*dynamic_cast<ParGridFunction*>(&optprob.GetGradient()));
   ParGridFunction &psi(*dynamic_cast<ParGridFunction*>
                        (&density.GetGridFunction()));
   ParGridFunction old_grad(&control_fes), old_psi(&control_fes);
   old_psi = psi; old_grad = grad;

   ParLinearForm diff_rho_form(&control_fes);
   std::unique_ptr<Coefficient> diff_rho(optprob.GetDensityDiffForm(old_psi));
   diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(*diff_rho));

   double compliance = optprob.Eval();
   double step_size, volume, stationarityError, stationarityError_bregman;
   int num_check;
   double old_compliance;

   TableLogger logger;
   logger.Append(std::string("Volume"), volume);
   logger.Append(std::string("Compliance"), compliance);
   logger.Append(std::string("Stationarity"), stationarityError);
   logger.Append(std::string("Re-evel"), num_check);
   logger.Append(std::string("Step Size"), step_size);
   logger.Append(std::string("Stationarity-Bregman"), stationarityError_bregman);

   optprob.UpdateGradient();
   for (int k = 0; k < max_it; k++)
   {
      // Compute Step size
      if (k == 0) { step_size = 1.0; }
      else
      {
         diff_rho_form.Assemble();
         old_psi -= psi;
         old_grad -= grad;
         step_size = std::fabs(diff_rho_form(old_psi)  / diff_rho_form(old_grad));
      }

      // Store old data
      old_compliance = compliance;
      old_psi = psi;
      old_grad = grad;

      // Step and upate gradient
      num_check = Step_Armijo(optprob, compliance, c1, step_size);
      compliance = optprob.GetValue();
      volume = density.GetVolume();
      optprob.UpdateGradient();

      // Visualization
      if (glvis_visualization)
      {
         if (sout_SIMP.is_open())
         {
            designDensity_gf->ProjectCoefficient(simp_rule.GetPhysicalDensity(
                                                    density.GetFilteredDensity()));
            sout_SIMP << "parallel " << num_procs << " " << myid << "\n";
            sout_SIMP << "solution\n" << pmesh << *designDensity_gf
                      << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
         if (sout_r.is_open())
         {
            rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
            sout_r << "parallel " << num_procs << " " << myid << "\n";
            sout_r << "solution\n" << pmesh << *rho_gf
                   << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
      }
      if (paraview)
      {
         pd->SetCycle(k);
         pd->SetTime(k);
         pd->Save();
      }

      // Check convergence
      stationarityError = density.StationarityErrorL2(grad);
      stationarityError_bregman = density.StationarityError(grad);

      logger.Print();

      if (stationarityError < 5e-05 && std::fabs(old_compliance - compliance) < 5e-05)
      {
         if (Mpi::Root()) { out << "Total number of iteration = " << k + 1 << std::endl; }
         break;
      }
   }
   if (save)
   {
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << psi;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << density.GetFilteredDensity();
   }

   return 0;
}