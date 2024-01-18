#include "mfem.hpp"
#include "mtop_coefficients.hpp"

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   int tot_iter = 100;
   int print_level = 1;
   bool visualization = false;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&tot_iter,
                  "-it",
                  "--linear-iterations",
                  "Maximum iterations for the linear solve.");
   args.AddOption(&fradius,
                  "-r",
                  "--radius",
                  "Filter radius");
   args.Parse();


   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();
   if (myrank==0)
   {
      std::cout<<"Dim:"<<dim<<" Sdim:"<<sdim<<std::endl;
   }


   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();


   mfem::RandFieldCoefficient rf(&pmesh,2);
   mfem::UniformDistributionCoefficient uf(&rf,0.0,1.0);
   rf.SetMaternParameter(6.7);

   mfem::H1_FECollection fec(2,pmesh.Dimension());
   mfem::ParFiniteElementSpace fes(&pmesh,&fec,1);
   mfem::ParGridFunction rg10(&fes);
   mfem::ParGridFunction rg05(&fes);
   mfem::ParGridFunction rg02(&fes);

   mfem::ParaViewDataCollection paraview_dc("Diffusion",&pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("rf_1.0",&rg10);
   paraview_dc.RegisterField("rf_0.5",&rg05);
   paraview_dc.RegisterField("rf_0.2",&rg02);


   for (int i=0; i<2; i++)
   {
      paraview_dc.SetCycle(i);
      paraview_dc.SetTime(double(i));

      rf.SetCorrelationLen(0.1);
      rf.Sample();
      rg10.ProjectCoefficient(uf);

      rf.SetCorrelationLen(0.5);
      rf.Sample();
      rg05.ProjectCoefficient(uf);

      rf.SetCorrelationLen(0.2);
      rf.Sample();
      rg02.ProjectCoefficient(uf);

      paraview_dc.Save();
   }

   MPI_Finalize();
   return 0;

}
