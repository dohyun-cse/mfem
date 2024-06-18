// Compliancemi minimization with method of moving asymptotes
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

#include "helper.hpp"
#include "mfem.hpp"
#include "prob_elasticity.hpp"
#include "topopt.hpp"
#include <fstream>
#include <iostream>
#include "MMA.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   const char *petscrc_file = "";
   mfem::MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   if (Mpi::Root())
   {
      mfem::out << "Parallel run using " << num_procs << " processes"
                << std::endl;
   }

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
   int max_it = 2e3;
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
   ostringstream filename_prefix;
   filename_prefix << "MMA-";

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
   args.AddOption(&E, "-E", "--E", "Lamé constant λ.");
   args.AddOption(&nu, "-nu", "--nu", "Lamé constant μ.");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&use_bregman, "-bregman", "--use-bregman", "-L2", "--use-L2",
                  "Use Bregman divergence as a stopping criteria");
   args.AddOption(&armijo, "-armijo", "--use-armijo", "-pos-bregman",
                  "--positive-bregman-condition",
                  "Use Armijo condition for step size selection. Otherwise, use "
                  "positivity of bregman divergence");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
   }

   std::unique_ptr<Mesh> mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   std::unique_ptr<VectorCoefficient> vforce_cf;
   std::string prob_name;
   GetElasticityProblem((ElasticityProblem)problem, filter_radius, vol_fraction,
                        mesh, vforce_cf, ess_bdr, ess_bdr_filter, prob_name,
                        seq_ref_levels, par_ref_levels);
   filename_prefix << prob_name << "-" << seq_ref_levels + par_ref_levels;
   int dim = mesh->Dimension();
   const int num_el = mesh->GetNE();
   std::unique_ptr<ParMesh> pmesh(static_cast<ParMesh *>(mesh.release()));

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
      if (Mpi::Root())
      {
         mfem::out << "GLVis for 3D is disabled. Use ParaView" << std::endl;
      }
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
   H1_FECollection state_fec(order, dim);  // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order - 1, dim,
                               BasisType::GaussLobatto); // space for ψ
   ParFiniteElementSpace state_fes(pmesh.get(), &state_fec, dim,
                                   Ordering::byNODES);
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
   HelmholtzFilter filter(filter_fes, filter_radius / (2.0 * sqrt(3.0)),
                          ess_bdr_filter, true);
   PrimalDesignDensity density(control_fes, filter, vol_fraction);

   ConstantCoefficient E_cf(E), nu_cf(nu);
   ParametrizedElasticityEquation elasticity(
      state_fes, density.GetFilteredDensity(), densityProjector, E_cf, nu_cf,
      *vforce_cf, ess_bdr, true);
   TopOptProblem optprob(elasticity.GetLinearForm(), elasticity, density, false,
                         true);

   ParGridFunction &u = dynamic_cast<ParGridFunction &>(optprob.GetState());
   ParGridFunction &rho_filter =
      dynamic_cast<ParGridFunction &>(density.GetFilteredDensity());
   ParGridFunction &grad(dynamic_cast<ParGridFunction &>(optprob.GetGradient()));
   ParGridFunction &rho(
      dynamic_cast<ParGridFunction &>(density.GetGridFunction()));
   rho_filter = density.GetDomainVolume() * vol_fraction;

   std::unique_ptr<ParGridFunction> gradH1_selfload;
   std::unique_ptr<ParLinearForm> projected_grad_selfload;
   std::unique_ptr<ScalarVectorProductCoefficient> self_weight;
   std::unique_ptr<VectorConstantCoefficient> gravity;
   std::unique_ptr<VectorGridFunctionCoefficient> u_cf;
   std::unique_ptr<GridFunctionCoefficient> rho_filter_cf;
   std::unique_ptr<InnerProductCoefficient> ug;
   std::unique_ptr<ProductCoefficient> ug_dfrho;
   std::unique_ptr<SumCoefficient> ug_all;
   if (problem <= ElasticityProblem::MBB_selfloading)
   {
      Vector g(dim);
      g = 0.0;
      g(dim - 1) = -9.81;
      gravity.reset(new VectorConstantCoefficient(g));
      rho_filter_cf.reset(new GridFunctionCoefficient(&rho_filter));
      u_cf.reset(new VectorGridFunctionCoefficient(&u));
      self_weight.reset(
         new ScalarVectorProductCoefficient(*rho_filter_cf, *gravity));
      ug.reset(new InnerProductCoefficient(*gravity, *u_cf));
      ug_dfrho.reset(new ProductCoefficient(
                        *ug, densityProjector.GetDerivative(rho_filter)));
      ug_all.reset(new SumCoefficient(*ug, *ug_dfrho));

      elasticity.GetLinearForm().AddDomainIntegrator(
         new VectorDomainLFIntegrator(*self_weight));
      elasticity.SetLinearFormStationary(false);
      elasticity.GetLinearForm().Assemble();

      gradH1_selfload.reset(new ParGridFunction(&filter_fes));
      *gradH1_selfload = 0.0;

      projected_grad_selfload.reset(new ParLinearForm(&control_fes));
      projected_grad_selfload->AddDomainIntegrator(
         new L2ProjectionLFIntegrator(*gradH1_selfload));
      density.SetVolumeConstraintType(-1);
      density.GetGridFunction() = inv_sigmoid(0.8);
   }
   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sout_frho, sout_r, sout_KKT;
   std::unique_ptr<ParGridFunction> designDensity_gf;
   ParGridFunction KKT_gf(&control_fes);
   if (glvis_visualization)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      designDensity_gf.reset(new ParGridFunction(&filter_fes));
      designDensity_gf->ProjectCoefficient(
         densityProjector.GetPhysicalDensity(density.GetFilteredDensity()));
      sout_frho.open(vishost, visport);
      if (sout_frho.is_open())
      {
         sout_frho << "parallel " << num_procs << " " << myid << "\n";
         sout_frho.precision(8);
         sout_frho << "solution\n"
                   << *pmesh << rho_filter
                   << "window_title 'Filtered density ρ̃ - MMA " << problem << "'\n"
                   << "keys Rjl***************\n"
                   << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }
      sout_r.open(vishost, visport);
      if (sout_r.is_open())
      {
         sout_r << "parallel " << num_procs << " " << myid << "\n";
         sout_r.precision(8);
         sout_r << "solution\n"
                << *pmesh << rho << "window_title 'Raw density ρ - MMA " << problem
                << "'\n"
                << "keys Rjl***************\n"
                << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }
      sout_KKT.open(vishost, visport);
      if (sout_KKT.is_open())
      {
         sout_KKT << "parallel " << num_procs << " " << myid << "\n";
         sout_KKT.precision(8);
         sout_KKT << "solution\n"
                  << *pmesh << KKT_gf << "window_title 'KKT - MMA " << problem
                  << "'\n"
                  << "keys Rjl***************\n"
                  << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }
   }
   std::unique_ptr<ParaViewDataCollection> pd;
   if (paraview)
   {
      pd.reset(new ParaViewDataCollection(filename_prefix.str(), pmesh.get()));
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("state", &u);
      pd->RegisterField("rho", &rho);
      pd->RegisterField("frho", &rho_filter);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(order > 1);
      pd->SetCycle(0);
      pd->SetTime(0);
      pd->Save();
   }
   double KKT0 = -infinity();
   ConstantCoefficient const_cf(1e09);
   Array<int> material_bdr(ess_bdr_filter);
   for (auto &val : material_bdr)
   {
      val = val == 1;
   }

   // 11. Iterate
   ParGridFunction old_grad(&control_fes), old_rho(&control_fes);

   ParLinearForm diff_rho_form(&control_fes);
   std::unique_ptr<Coefficient> diff_rho(optprob.GetDensityDiffCoeff(old_rho));
   diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(*diff_rho));

   ParLinearForm der_sig_form(&control_fes);
   MappedGridFunctionCoefficient der_sig(&rho, [](double x)
   {
      const double rho = sigmoid(x);
      return rho * (1 - rho);
   });
   der_sig_form.AddDomainIntegrator(new DomainLFIntegrator(der_sig));
   ParGridFunction one_gf(&control_fes);
   one_gf = 1.0;
   ParGridFunction tmp_rho_gf(&control_fes);

   if (Mpi::Root())
      mfem::out << "\n"
                << "Initialization Done." << "\n"
                << "Start Projected Mirror Descent Step." << "\n"
                << std::endl;

   double compliance = optprob.Eval();
   optprob.UpdateGradient();
   if (problem <= ElasticityProblem::MBB_selfloading)
   {
      filter.Apply(*ug, *gradH1_selfload);
      projected_grad_selfload->Assemble();
      grad.Add(2.0, *projected_grad_selfload);
   }
   double step_size(0.5),
          volume(density.GetVolume() / density.GetDomainVolume()),
          stationarityError(density.StationarityError(grad));
   int num_reeval(0);
   double old_compliance;
   double stationarityError0(stationarityError);
   double relative_stationarity(1.0), relative_stationarity_bregman(1.0);
   double KKT(0.0);
   TableLogger logger;
   logger.Append(std::string("Volume"), volume);
   logger.Append(std::string("Compliance"), compliance);
   logger.Append(std::string("Stationarity"), stationarityError);
   logger.Append(std::string("Re-evel"), num_reeval);
   logger.Append(std::string("Step Size"), step_size);
   logger.Append(std::string("KKT"), KKT);
   logger.SaveWhenPrint(filename_prefix.str());
   logger.Print();


   NativeMMA *mma;
   {
      double a=0.0;
      double c=1000.0;
      double d=0.0;
      mma = new mfem::NativeMMA(pmesh->GetComm(), 1, grad, &a, &c, &d);
   }
   bool converged = false;
   density.ComputeVolume();
   ParGridFunction lower(rho), upper(rho), ograd(rho), one(&control_fes),
                   dv(&control_fes), old2_rho(&control_fes);
   ParBilinearForm mass(&control_fes);
   mass.AddDomainIntegrator(new MassIntegrator());
   mass.Assemble();
   one = 1.0;
   mass.Mult(one, dv);
   for (int k = 0; k < max_it; k++)
   {
      old_compliance = compliance;
      if (k < 2)
      {
         lower = rho; lower -= 0.5;
         upper = rho; upper += 0.5;
      }
      else
      {
         double gamma;
         for (int i=0; i<rho.Size(); i++)
         {
            const double diff = (rho(i) - old_rho(i))*(old_rho(i) - old2_rho(i));
            if (diff < 0) { gamma = 0.7; }
            else if (diff > 0) { gamma = 1.2; }
            else { gamma = 1.0; }
            lower(i) = std::max(rho(i) - gamma*(old_rho(i)-lower(i)), 0.0);
            upper(i) = std::min(rho(i) + gamma*(upper(i)-old_rho(i)), 1.0);
         }
      }
      lower.Clip(0, 1);
      upper.Clip(0, 1);
      old2_rho = old_rho;
      old_rho = rho;
      double con = density.GetVolume() - density.GetDomainVolume()*vol_fraction;
      mma->Update(rho, grad, &con, &dv, lower, upper);
      volume = density.ComputeVolume();
      compliance = optprob.Eval();
      optprob.UpdateGradient();
      if (problem <= ElasticityProblem::MBB_selfloading)
      {
         filter.Apply(*ug, *gradH1_selfload);
         projected_grad_selfload->Assemble();
         grad.Add(2.0, *projected_grad_selfload);
      }

      // Step 4. Visualization
      if (glvis_visualization)
      {
         if (sout_frho.is_open())
         {
            // designDensity_gf->ProjectCoefficient(densityProjector.GetPhysicalDensity(
            //                                         density.GetFilteredDensity()));
            sout_frho << "parallel " << num_procs << " " << myid << "\n";
            sout_frho << "solution\n" << *pmesh << rho_filter << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
         if (sout_r.is_open())
         {
            sout_r << "parallel " << num_procs << " " << myid << "\n";
            sout_r << "solution\n" << *pmesh << rho << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
         if (sout_KKT.is_open())
         {
            sout_KKT << "parallel " << num_procs << " " << myid << "\n";
            sout_KKT << "solution\n" << *pmesh << KKT_gf << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
      }
      if (paraview)
      {
         pd->SetCycle(k + 1);
         pd->SetTime(k + 1);
         pd->Save();
      }

      // Check convergence
      stationarityError =
         density.StationarityError(grad);

      relative_stationarity = stationarityError / stationarityError0;
      logger.Print();

      if ((use_bregman ? relative_stationarity_bregman : relative_stationarity) <
          tol_stationarity &&
          std::fabs((old_compliance - compliance) / compliance) <
          tol_compliance)
      {
         converged = true;
         if (Mpi::Root())
         {
            mfem::out << "Total number of iteration = " << k + 1 << std::endl;
         }
         break;
      }
      // if (KKT < 1e-04)
      // {
      //    converged = true;
      //    if (Mpi::Root())
      //    {
      //       mfem::out << "Total number of iteration = " << k + 1 << std::endl;
      //    }
      //    break;
      // }
   }
   if (!converged)
   {
      if (Mpi::Root())
      {
         mfem::out << "Total number of iteration = " << max_it << std::endl;
      }
      if (Mpi::Root())
      {
         mfem::out << "Maximum iteration reached." << std::endl;
      }
   }
   if (save)
   {
      ostringstream solfile, solfile2;
      solfile << filename_prefix.str() << "-" << seq_ref_levels << "-"
              << par_ref_levels << "-0." << setfill('0') << setw(6) << myid;
      solfile2 << filename_prefix.str() << "-" << seq_ref_levels << "-"
               << par_ref_levels << "-f." << setfill('0') << setw(6) << myid;
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << rho;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << density.GetFilteredDensity();
   }

   delete mma; // explicitly delete mma before finalize petsc
   mfem::MFEMFinalizePetsc();
   mfem::Mpi::Finalize();
   return 0;
}
