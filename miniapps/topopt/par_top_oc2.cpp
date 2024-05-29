
// Compliancemi minimization with projected mirror descent in parallel
//
//                  minimize F(ρ) = ∫_Ω f⋅u dx over ρ ∈ L¹(Ω)
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
//              Update is done by
//
//              ρ_new = sigmoid(ψ_new) = sigmoid(ψ_cur - α ∇F(ρ_cur) + c)
//
//              where c is a constant volume correction. The step size α is
//              determined by a generalized Barzilai-Borwein method with
//              Armijo condition check
//
//              BB:        α_init = |(δψ, δρ) / (δ∇F(ρ), δρ)|
//
//              Armijo:   F(ρ(α)) ≤ F(ρ_cur) + c_1 (∇F(ρ_cur), ρ(α) - ρ_cur)
//                        with ρ(α) = sigmoid(ψ_cur - α∇F(ρ_cur) + c)
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
#include "prob_elasticity.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   if (Mpi::Root()) { mfem::out << "Parallel run using " << num_procs << " processes" << std::endl; }

   // 1. Parse command-line options.
   int seq_ref_levels = 0;
   int par_ref_levels = 6;
   int order = 1;
   // filter radius. Use problem-dependent default value if not provided.
   // See switch statements below
   double filter_radius = -1;
   // Volume fraction. Use problem-dependent default value if not provided.
   // See switch statements below
   double vol_fraction = -1;
   int max_it = 1e2;
   double rho_min = 1e-06;
   double exponent = 3.0;
   double E = 1.0;
   double nu = 0.3;
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = false;
   bool paraview = true;
   double tol_stationarity = 1e-04;
   double tol_compliance = 5e-05;
   bool use_bregman = true;
   bool armijo = true;
   bool use_BGG = true;;
   ostringstream filename_prefix;
   filename_prefix << "OC-";

   int problem = ElasticityProblem::Cantilever;

   OptionsParser args(argc, argv);
   args.AddOption(&seq_ref_levels, "-rs", "--seq-refine",
                  "Number of times to refine the sequential mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--par-refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) Cantilever, 1) MBB, 2) LBracket.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&E, "-E", "--E",
                  "Lamé constant λ.");
   args.AddOption(&nu, "-nu", "--nu",
                  "Lamé constant μ.");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&use_bregman, "-bregman", "--use-bregman", "-L2",
                  "--use-L2",
                  "Use Bregman divergence as a stopping criteria");
   args.AddOption(&armijo, "-armijo", "--use-armijo", "-pos-bregman",
                  "--positive-bregman-condition",
                  "Use Armijo condition for step size selection. Otherwise, use positivity of bregman divergence");
   args.AddOption(&use_BGG, "-bgg", "--use-BGG", "-exp",
                  "--exponential-stepsize",
                  "Use Barzilai-Borwein step size selection. Otherwise, use exponential step size");
   args.Parse();
   if (!args.Good()) {if (Mpi::Root()) args.PrintUsage(mfem::out);}


   std::unique_ptr<Mesh> mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   std::unique_ptr<VectorCoefficient> vforce_cf;
   std::string prob_name;
   GetElasticityProblem((ElasticityProblem)problem, filter_radius, vol_fraction,
                        mesh, vforce_cf,
                        ess_bdr, ess_bdr_filter,
                        prob_name, seq_ref_levels, par_ref_levels);
   filename_prefix << prob_name << "-" << seq_ref_levels + par_ref_levels;
   int dim = mesh->Dimension();
   const int num_el = mesh->GetNE();
   std::unique_ptr<ParMesh> pmesh(static_cast<ParMesh*>(mesh.release()));

   if (Mpi::Root())
      mfem::out << "\n"
                << "Compliance Minimization with Projected Mirror Descent.\n"
                << "Problem: " << filename_prefix.str() << "\n"
                << "The number of elements: " << num_el << "\n"
                << "Order: " << order << "\n"
                << "Volume Fraction: " << vol_fraction << "\n"
                << "Filter Radius: " << filter_radius << "\n"
                << "Maximum iteration: " << max_it << "\n"
                << "GLVis: " << glvis_visualization << "\n"
                << "Paraview: " << paraview << std::endl;

   if (glvis_visualization && dim == 3)
   {
      glvis_visualization = false;
      paraview = true;
      if (Mpi::Root()) { mfem::out << "GLVis for 3D is disabled. Use ParaView" << std::endl; }
   }
   pmesh->SetAttributes();

   if (save)
   {
      ostringstream meshfile;
      meshfile << filename_prefix.str() << "." << setfill('0') << setw(6) << myid;
      ofstream mesh_ofs(meshfile.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);
   }

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   ParFiniteElementSpace state_fes(pmesh.get(), &state_fec,dim, Ordering::byNODES);
   ParFiniteElementSpace filter_fes(pmesh.get(), &filter_fec);
   ParFiniteElementSpace control_fes(pmesh.get(), &control_fec);

   int state_size = state_fes.GlobalTrueVSize();
   int control_size = control_fes.GlobalTrueVSize();
   int filter_size = filter_fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "\n"
                << "Number of state unknowns: " << state_size << "\n"
                << "Number of filter unknowns: " << filter_size << "\n"
                << "Number of control unknowns: " << control_size << std::endl;
   }

   // 5. Set the initial guess for ρ.
   // ThresholdProjector densityProjector(0.5, 32, exponent, rho_min);
   SIMPProjector densityProjector(exponent, rho_min);
   HelmholtzFilter filter(filter_fes, filter_radius/(2.0*sqrt(3.0)),
                          ess_bdr_filter);
   LatentDesignDensity density_in_latent(control_fes, filter, vol_fraction,
                                         ShannonEntropy, log_d, exp_d, false, true);
   PrimalDesignDensity density(control_fes, filter, vol_fraction);

   ConstantCoefficient E_cf(E), nu_cf(nu);
   ParametrizedElasticityEquation elasticity(state_fes,
                                             density.GetFilteredDensity(), densityProjector, E_cf, nu_cf, *vforce_cf,
                                             ess_bdr);
   TopOptProblem optprob(elasticity.GetLinearForm(), elasticity, density, false,
                         false);

   ParGridFunction &u = dynamic_cast<ParGridFunction&>(optprob.GetState());
   ParGridFunction &rho_filter = dynamic_cast<ParGridFunction&>
                                 (density.GetFilteredDensity());
   ParGridFunction &grad(dynamic_cast<ParGridFunction&>(optprob.GetGradient()));
   ParGridFunction &rho(dynamic_cast<ParGridFunction&>(density.GetGridFunction()));
   rho_filter = density.GetDomainVolume()*vol_fraction;

   std::unique_ptr<ParGridFunction> gradH1_selfload;
   std::unique_ptr<ParLinearForm> projected_grad_selfload;
   std::unique_ptr<ScalarVectorProductCoefficient> self_weight;
   std::unique_ptr<VectorConstantCoefficient> gravity;
   std::unique_ptr<VectorGridFunctionCoefficient> u_cf;
   std::unique_ptr<GridFunctionCoefficient> rho_filter_cf;
   std::unique_ptr<InnerProductCoefficient> ug;
   if (problem >= ElasticityProblem::MBB_selfloading)
   {
      Vector g(dim); g = 0.0; g(dim - 1) = -1.0;
      gravity.reset(new VectorConstantCoefficient(g));
      rho_filter_cf.reset(new GridFunctionCoefficient(&rho_filter));
      u_cf.reset(new VectorGridFunctionCoefficient(&u));
      self_weight.reset(new ScalarVectorProductCoefficient(*rho_filter_cf, *gravity));
      ug.reset(new InnerProductCoefficient(*gravity, *u_cf));

      elasticity.GetLinearForm().AddDomainIntegrator(new VectorDomainLFIntegrator(
                                                        *self_weight));
      elasticity.SetLinearFormStationary(false);
      elasticity.GetLinearForm().Assemble();

      gradH1_selfload.reset(new ParGridFunction(&filter_fes));
      *gradH1_selfload = 0.0;

      projected_grad_selfload.reset(new ParLinearForm(&control_fes));
      projected_grad_selfload->AddDomainIntegrator(new L2ProjectionLFIntegrator(
                                                      *gradH1_selfload));
      density.SetVolumeConstraintType(-1);
      density.GetGridFunction() = 0.0;
   }
   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_SIMP, sout_r;
   std::unique_ptr<ParGridFunction> designDensity_gf, rho_gf;
   if (glvis_visualization)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      designDensity_gf.reset(new ParGridFunction(&filter_fes));
      rho_gf.reset(new ParGridFunction(&control_fes));
      designDensity_gf->ProjectCoefficient(densityProjector.GetPhysicalDensity(
                                              density.GetFilteredDensity()));
      rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
      sout_SIMP.open(vishost, visport);
      if (sout_SIMP.is_open())
      {
         sout_SIMP << "parallel " << num_procs << " " << myid << "\n";
         sout_SIMP.precision(8);
         sout_SIMP << "solution\n" << *pmesh << *designDensity_gf
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
         sout_r << "solution\n" << *pmesh << *rho_gf
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
      if (!rho_gf)
      {
         // rho_gf.reset(new ParGridFunction(&filter_fes));
         // rho_gf->ProjectDiscCoefficient(density.GetDensityCoefficient(),
         //                                GridFunction::AvgType::ARITHMETIC);
         rho_gf.reset(new ParGridFunction(rho));
         rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
      }
      pd.reset(new ParaViewDataCollection(filename_prefix.str(), pmesh.get()));
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("state", &u);
      pd->RegisterField("rho", rho_gf.get());
      pd->RegisterField("frho", &rho_filter);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(order > 1);
      pd->SetCycle(0);
      pd->SetTime(0);
      pd->Save();
   }

   // 11. Iterate
   ParGridFunction old_grad(&control_fes), old_rho(&control_fes);
   ParGridFunction old_psi(&control_fes);

   ParLinearForm diff_rho_form(&control_fes);
   std::unique_ptr<Coefficient> diff_rho(optprob.GetDensityDiffCoeff(old_rho));
   diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(*diff_rho));

   if (Mpi::Root())
      mfem::out << "\n"
                << "Initialization Done." << "\n"
                << "Start Projected Mirror Descent Step." << "\n" << std::endl;

   double compliance = optprob.Eval();
   optprob.UpdateGradient();
   density_in_latent.GetGridFunction() = rho;
   density_in_latent.GetGridFunction().ApplyMap([](double x) {return std::log(std::max(1e-10, x));});
   double step_size(1.0), volume(density.GetVolume() / density.GetDomainVolume()),
          stationarityError(density.StationarityError(grad));
   double stationarityError_bregman(density_in_latent.StationarityError(grad));
   int num_reeval(0);
   double old_compliance;
   double stationarityError0(stationarityError);
   double stationarityError_bregman0(stationarityError_bregman);
   double relative_stationarity(1.0), relative_stationarity_bregman(1.0);
   TableLogger logger;
   logger.Append(std::string("Volume"), volume);
   logger.Append(std::string("Compliance"), compliance);
   logger.Append(std::string("Stationarity (Rel)"), relative_stationarity);
   logger.Append(std::string("Re-evel"), num_reeval);
   logger.Append(std::string("Step Size"), step_size);
   logger.Append(std::string("Stationarity Bregman (Rel)"),
                 relative_stationarity_bregman);
   logger.SaveWhenPrint(filename_prefix.str());
   logger.Print();

   bool converged = false;
   ConstantCoefficient zero_cf(0.0);
   ParBilinearForm mass(&control_fes);
   mass.AddDomainIntegrator(new MassIntegrator());
   mass.Assemble();
   ParGridFunction d(&control_fes);
   ParGridFunction lower(old_rho), upper(old_rho);
   const double mv = 0.2;

   ParBilinearForm inv_mass(&control_fes);
   inv_mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   inv_mass.Assemble();

   for (int k = 0; k < max_it; k++)
   {
      // Step 1. Compute Step size
      // if (use_BGG)
      // {
      //    if (k > 0)
      //    {
      //       diff_rho_form.Assemble();
      //       old_psi -= psi;
      //       old_grad -= grad;
      //       step_size = std::fabs(diff_rho_form(old_psi)  / diff_rho_form(old_grad));
      //    }
      // }
      // else
      // {
      //    step_size *= 2.0;
      // }
      step_size = 0.5;

      // Step 2. Store old data
      old_compliance = compliance;
      old_rho = rho;
      old_psi = density_in_latent.GetGridFunction();
      old_grad = grad;
      inv_mass.Mult(grad, d);
      d.ApplyMap([](double x) {return std::sqrt(-x);});

      lower = old_rho; lower -= mv;
      upper = old_rho; upper += mv;
      lower.Clip(0.0, 1.0); upper.Clip(0.0, 1.0);
      double l1(0.0), l2(1e09);
      while ((l2 - l1) > 1e-05)
      {
         double lmid = 0.5*(l1 + l2);
         rho = old_rho;
         rho *=d;
         rho *= 1.0 / std::sqrt(lmid);
         rho.Clip(lower, upper);
         volume = density.ComputeVolume();
         if (volume > density.GetDomainVolume()*vol_fraction)
         {
            l1 = lmid;
         }
         else
         {
            l2 = lmid;
         }
      }
      optprob.Eval();
      compliance = optprob.GetValue();
      volume = density.GetVolume() / density.GetDomainVolume();
      optprob.UpdateGradient();

      density_in_latent.GetGridFunction() = rho;
      density_in_latent.GetGridFunction().ApplyMap([](double x) {return std::log(std::max(1e-10, x));});
      // Step 4. Visualization
      if (rho_gf)
      {
         // rho_gf->ProjectDiscCoefficient(density.GetDensityCoefficient(),
         //                                GridFunction::AvgType::ARITHMETIC);
         // MappedGridFunctionCoefficient bdr_rho(&psi, [](double x)
         // {
         //    return x > 0.0 ? 0.0 : sigmoid(x);
         // });
         // Array<int> bdr(ess_bdr_filter);
         // bdr = 1;
         // for (int i=0; i<bdr.Size(); i++)
         // {
         //    for (int j=0; j<ess_bdr.NumRows(); j++)
         //    {
         //       if (ess_bdr(j, i))
         //       {
         //          bdr[i] = 0;
         //       }
         //    }
         // }
         // rho_gf->ProjectBdrCoefficient(bdr_rho, bdr);
         // rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
      }
      if (glvis_visualization)
      {
         if (sout_SIMP.is_open())
         {
            // designDensity_gf->ProjectCoefficient(densityProjector.GetPhysicalDensity(
            //                                         density.GetFilteredDensity()));
            sout_SIMP << "parallel " << num_procs << " " << myid << "\n";
            sout_SIMP << "solution\n" << *pmesh << rho_filter
                      << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
         if (sout_r.is_open())
         {
            sout_r << "parallel " << num_procs << " " << myid << "\n";
            sout_r << "solution\n" << *pmesh << rho
                   << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
      }
      if (paraview)
      {
         pd->SetCycle(k+1);
         pd->SetTime(k+1);
         pd->Save();
      }

      // Check convergence
      stationarityError = density.StationarityError(grad);
      stationarityError_bregman = density_in_latent.StationarityError(grad);

      relative_stationarity = stationarityError/stationarityError0;
      relative_stationarity_bregman =
         stationarityError_bregman/stationarityError_bregman0;

      if ((use_bregman ? relative_stationarity_bregman : relative_stationarity) <
          tol_stationarity &&
          std::fabs((old_compliance - compliance)/compliance) < tol_compliance)
      {
         converged = true;
         if (Mpi::Root()) { mfem::out << "Total number of iteration = " << k + 1 << std::endl; }
         break;
      }

      logger.Print();
      relative_stationarity = stationarityError/stationarityError0;

      if ((relative_stationarity) <
          tol_stationarity &&
          std::fabs((old_compliance - compliance)/compliance) < tol_compliance)
      {
         converged = true;
         if (Mpi::Root()) { mfem::out << "Total number of iteration = " << k + 1 << std::endl; }
         break;
      }
   }
   if (!converged)
   {
      if (Mpi::Root()) { mfem::out << "Total number of iteration = " << max_it << std::endl; }
      if (Mpi::Root()) { mfem::out << "Maximum iteration reached." << std::endl; }
   }
   if (save)
   {
      ostringstream solfile, solfile2;
      solfile << filename_prefix.str() << "-" << seq_ref_levels << "-" <<
              par_ref_levels << "-0." << setfill('0') << setw(6) << myid;
      solfile2 << filename_prefix.str() << "-" << seq_ref_levels << "-" <<
               par_ref_levels << "-f." << setfill('0') << setw(6) << myid;
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << rho;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << density.GetFilteredDensity();
   }

   return 0;
}