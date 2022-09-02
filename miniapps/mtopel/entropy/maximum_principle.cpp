//                                MFEM Obstacle Problem
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


double eps = 1e-3;
// double eps = 1e-2;
double Ramp_BC(const Vector &pt);
double EJ_exact_solution(const Vector &pt);

double lnit(double x)
{
   double tol = 1e-12;
   x = min(max(tol,x),1.0-tol);
   // MFEM_ASSERT(x>0.0, "Argument must be > 0");
   // MFEM_ASSERT(x<1.0, "Argument must be < 1");
   return log(x/(1.0-x));
}

double expit(double x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+exp(-x));
   }
   else
   {
      return exp(x)/(1.0+exp(x));
   }
}

double dexpitdx(double x)
{
   double tmp = expit(-x);
   return tmp - pow(tmp,2);
}

double d2expitdx2(double x)
{
   double tmp = expit(-x);
   return -tmp + 3.0 * pow(tmp,2) - 2.0 * pow(tmp, 3);
}

class LnitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   LnitGridFunctionCoefficient(GridFunction &u_, double min_val_=-1e10, double max_val_=1e10)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExpitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   ExpitGridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dExpitdxGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   dExpitdxGridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class d2Expitdx2GridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   d2Expitdx2GridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;
   int max_it = 5;
   int ref_levels = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   UMFPackSolver umf_solver;

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   // ess_tdof_list.Print();

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   auto func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(2.0*M_PI*x(i));
      }
      // return -1.0;
      return -x.Size() * pow(2.0*M_PI,2) * val / 2.0;
      // return x.Size()*pow(M_PI,2) * val;
   };
   auto perturbation_func = [](const Vector &x)
   {
      double scale = 5e-1;
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return 1.0 + scale * pow(val, 3.0);
   };
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Vector zero_vec(dim);
   zero_vec = 0.0;

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction u(&fespace);
   GridFunction psi(&fespace);
   GridFunction delta_psi(&fespace);
   GridFunction psi_old(&fespace);
   delta_psi = 0.0;

   /////////// Example 1   
   // u = 0.5;
   // VectorConstantCoefficient beta_coeff(zero_vec);
   // FunctionCoefficient f(func);
   // ConstantCoefficient bdry_coef(0.5);
   // u.ProjectBdrCoefficient(bdry_coef, ess_bdr);
   // double alpha0 = 1.0;

   // /////////// Example 2
   // u = 0.5;
   // Vector beta(dim);
   // beta(0) = 0.0;
   // beta(1) = 0.0;
   // // beta(0) = 1.0;
   // // beta(1) = 0.5;
   // beta /= sqrt(1.25);
   // double eps = 1.0;
   // ConstantCoefficient f(0.0);
   // FunctionCoefficient bdry_coef(Ramp_BC);
   // VectorConstantCoefficient beta_coeff(beta);
   // double alpha0 = 0.5;
   // u.ProjectBdrCoefficient(bdry_coef, ess_bdr);

   // /////////// Example 3
   u = 0.5;
   Vector beta(dim);
   beta(0) = 1.0;
   beta(1) = 0.0;
   ConstantCoefficient f(0.0);
   FunctionCoefficient bdry_coef(EJ_exact_solution);
   VectorConstantCoefficient beta_coeff(beta);
   double alpha0 = 0.01;
   // FunctionCoefficient perturbation(perturbation_func);
   // ProductCoefficient IC_coeff(bdry_coef, perturbation);
   // u.ProjectCoefficient(IC_coeff);
   u.ProjectCoefficient(bdry_coef);


   // u.ProjectBdrCoefficient(bdry_coef, ess_bdr);

   for (int i = 0; i < u.Size(); i++)
   {
      if (u(i) > 1.0)
      {
         u(i) = 1.0;
      }
      else if (u(i) < 0.0)
      {
         u(i) = 0.0;
      }
   }

   LnitGridFunctionCoefficient lnit_u(u);
   psi.ProjectCoefficient(lnit_u);
   psi_old = psi;

   OperatorPtr A;
   Vector B, X;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 12. Iterate
   for (int k = 0; k < max_it; k++)
   {
      // double alpha = alpha0 / log(k+2);
      // double alpha = alpha0 / sqrt(k+1);
      // double alpha = alpha0 * sqrt(k+1);
      double alpha = alpha0;
      // alpha *= 2;

      for (int j = 0; j < 3; j++)
      {
         // A. Assembly
         
         // MD
         double c1 = eps;
         double c2 = eps*(1.0 - alpha);
         // double c1 = 1.0;
         // double c2 = (1.0 - eps*alpha);

         // IMD
         // double c1 = 1.0 + alpha;
         // double c2 = 1.0;

         GridFunctionCoefficient psi_cf(&psi);
         GridFunctionCoefficient psi_old_cf(&psi_old);
         dExpitdxGridFunctionCoefficient dexpitdx_psi(psi);
         dExpitdxGridFunctionCoefficient dexpitdx_psi_old(psi_old);
         dExpitdxGridFunctionCoefficient d2expitdx2_psi(psi);
         // dExpitdxGridFunctionCoefficient d2expitdx2_psi_old(psi_old);
         GradientGridFunctionCoefficient grad_psi(&psi);
         GradientGridFunctionCoefficient grad_psi_old(&psi_old);
         ScalarVectorProductCoefficient c1_grad_psi(c1, grad_psi);
         ScalarVectorProductCoefficient c2_grad_psi_old(c2, grad_psi_old);
         ProductCoefficient c1_dexpitdx_psi(c1, dexpitdx_psi);
         ScalarVectorProductCoefficient c1_dexpitdx_psi_grad_psi(dexpitdx_psi, c1_grad_psi);
         ScalarVectorProductCoefficient c1_d2expitdx2_psi_grad_psi(d2expitdx2_psi, c1_grad_psi);
         ScalarVectorProductCoefficient c2_dexpitdx_psi_old_grad_psi_old(dexpitdx_psi_old, c2_grad_psi_old);
         // InnerProductCoefficient beta_dexpitdx_psi(beta_coeff, dexpitdx_psi);



         BilinearForm a(&fespace);
         a.AddDomainIntegrator(new DiffusionIntegrator(c1_dexpitdx_psi));
         a.AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(c1_d2expitdx2_psi_grad_psi)));
         a.AddDomainIntegrator(new MassIntegrator(one));
         a.Assemble();

         VectorSumCoefficient gradient_term_RHS(c1_dexpitdx_psi_grad_psi, c2_dexpitdx_psi_old_grad_psi_old, -1.0, 1.0);
         SumCoefficient mass_term_RHS(psi_cf, psi_old_cf, -1.0, 1.0);
         ProductCoefficient alpha_f(alpha, f);
         ScalarVectorProductCoefficient dexpitdx_psi_old_grad_psi_old(dexpitdx_psi_old, grad_psi_old);
         ScalarVectorProductCoefficient minus_alpha_beta(-alpha, beta_coeff);
         InnerProductCoefficient convection_coef(minus_alpha_beta, dexpitdx_psi_old_grad_psi_old);

         LinearForm b(&fespace);
         b.AddDomainIntegrator(new DomainLFGradIntegrator(gradient_term_RHS));
         b.AddDomainIntegrator(new DomainLFIntegrator(mass_term_RHS));
         b.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
         b.AddDomainIntegrator(new DomainLFIntegrator(convection_coef));
         b.Assemble();

         // B. Solve state equation
         a.FormLinearSystem(ess_tdof_list, delta_psi, b, A, X, B);
         // GSSmoother S((SparseMatrix&)(*A));
         // GMRES(*A, S, B, X, 0, 20000, 100, 1e-8, 1e-8);
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);

         // C. Recover state variable
         a.RecoverFEMSolution(X, b, delta_psi);

         double gamma = 1.0;
         delta_psi *= gamma;
         psi += delta_psi;
      }
      psi_old = psi;

      // 14. Send the solution by socket to a GLVis server.
      ExpitGridFunctionCoefficient expit_psi(psi);
      u.ProjectCoefficient(expit_psi);
      // sol_sock << "solution\n" << mesh << psi << "window_title 'Discrete solution'" << flush;
      sol_sock << "solution\n" << mesh << u << "window_title 'Discrete solution'" << flush;
   }

   // // 14. Exact solution.
   // if (visualization)
   // {
   //    socketstream err_sock(vishost, visport);
   //    err_sock.precision(8);
   //    FunctionCoefficient exact_coef(exact_solution);

   //    GridFunction error(&fespace);
   //    error = 0.0;
   //    error.ProjectCoefficient(exact_coef);
   //    error -= u;

   //    err_sock << "solution\n" << mesh << error << "window_title 'Error'"  << flush;
   // }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}

double LnitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, lnit(val)));
}

double ExpitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, expit(val)));
}

double dExpitdxGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, dexpitdx(val)));
}

double d2Expitdx2GridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, d2expitdx2(val)));
}

double Ramp_BC(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double tol = 1e-10;
   double eps = 0.05;

   if (  (abs(y) < tol && x >= 0.2)
      || (abs(x-1.0) < tol)
      || (abs(y-1.0) < tol) )
   {
      return 0.0;
   }
   else if (  (abs(x) < tol && y <= 1.0 - eps)
           || (abs(y) < tol && x <= 0.2 - eps) )
   {
      return 1.0;
   }
   else if (x >= (0.2 - eps) && abs(y) < tol)
   {
      return (0.2 - x)/eps;
   }
   else if (y >= (1.0 - eps) && abs(x) < tol)
   {
      return  (1.0 - y)/eps;
   }
   else
   {
      return 0.5;
   }
}

double EJ_exact_solution(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double lambda = M_PI*M_PI*eps;
   double r1 = (1.0 + sqrt(1.0 + 4.0 * eps * lambda))/(2*eps);
   double r2 = (1.0 - sqrt(1.0 + 4.0 * eps * lambda))/(2*eps);

   double num = exp(r2 * (x - 1.0)) - exp(r1 * (x-1.0));
   double denom = exp(-r2) - exp(-r1);
   // double denom = r1 * exp(-r2) - r2 * exp(-r1);

   double scale = 0.5;
   // double scale = (r1 * exp(-r2) - r2 * exp(-r1)) / (exp(-r2) - exp(-r1));

   return scale * num / denom * cos(M_PI * y) + 0.5;
   
}