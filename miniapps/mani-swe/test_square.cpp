#include "mfem.hpp"
#include "manihyp.hpp"

using namespace mfem;


int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int order = 8;
   int refinement_level = 3;

   Hypre::Init();
   std::unique_ptr<ParMesh> pmesh;
   {
      Mesh mesh("./data/periodic-square-3d.mesh");
      pmesh.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      mesh.Clear();
   }

   const int dim = pmesh->Dimension();
   const int sdim = pmesh->SpaceDimension();


   pmesh->SetCurvature(order, true);
   for (int i=0; i<refinement_level; i++)
   {
      pmesh->UniformRefinement();
   }
   bool visualization = true;
   DG_FECollection dg_fec(order, dim);
   ParFiniteElementSpace pfes(pmesh.get(), &dg_fec);
   ParGridFunction x(&pfes);
   x = 1.0;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x
               << "keys 'mj'"
               << std::flush;
   }
}