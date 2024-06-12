//                  MFEM Example 18 - Serial/Parallel Shared Code
//                  (Implementation of Time-dependent DG Operator)
//
// This code provide example problems for the Euler equations and implements
// the time-dependent DG operator given by the equation:
//
//            (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u, n), [[v]]>_F = 0.
//
// This operator is designed for explicit time stepping methods. Specifically,
// the function IGRDGHyperbolicConservationLaws::Mult implements the following
// transformation:
//
//                             u ↦ M⁻¹(-DF(u) + NF(u))
//
// where M is the mass matrix, DF is the weak divergence of flux, and NF is the
// interface flux. The inverse of the mass matrix is computed element-wise by
// leveraging the block-diagonal structure of the DG mass matrix. Additionally,
// the flux-related terms are computed using the HyperbolicFormIntegrator.
//
// The maximum characteristic speed is determined for each time step. For more
// details, refer to the documentation of IGRDGHyperbolicConservationLaws::Mult.
//

#include "mfem.hpp"

namespace mfem
{
// real_t proper_mod(const real_t x, const real_t divisor)
// {
//    const real_t quotient = std::ceil(x / divisor);
//    const real_t remainder = divisor * (1 + x / divisor - quotient);
//    return remainder == divisor ? 0.0 : remainder;
// }

void SnapToSphere(Mesh &mesh, real_t r)
{
   GridFunction &nodes = *mesh.GetNodes();
   Vector node(mesh.SpaceDimension());
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
   {
      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
      }

      node *= r / (node*node);

      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
      }
   }
   if (mesh.Nonconforming())
   {
      // Snap hanging nodes to the master side.
      Vector tnodes;
      nodes.GetTrueDofs(tnodes);
      nodes.SetFromTrueDofs(tnodes);
   }
}

Mesh* MakeSphereMesh(int elem_type, real_t r)
{
   // 2. Generate an initial high-order (surface) mesh on the unit sphere. The
   //    Mesh object represents a 2D mesh in 3 spatial dimensions. We first add
   //    the elements and the vertices of the mesh, and then make it high-order
   //    by specifying a finite element space for its nodes.
   int Nvert = 8, Nelem = 6;
   if (elem_type == 0)
   {
      Nvert = 6;
      Nelem = 8;
   }
   Mesh *mesh = new Mesh(2, Nvert, Nelem, 0, 3);

   if (elem_type == 0) // inscribed octahedron
   {
      const real_t tri_v[6][3] =
      {
         { 1,  0,  0}, { 0,  1,  0}, {-1,  0,  0},
         { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}
      };
      const int tri_e[8][3] =
      {
         {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
         {1, 0, 5}, {2, 1, 5}, {3, 2, 5}, {0, 3, 5}
      };

      for (int j = 0; j < Nvert; j++)
      {
         mesh->AddVertex(tri_v[j]);
      }
      for (int j = 0; j < Nelem; j++)
      {
         int attribute = j + 1;
         mesh->AddTriangle(tri_e[j], attribute);
      }
      mesh->FinalizeTriMesh(1, 1, true);
   }
   else // inscribed cube
   {
      const real_t quad_v[8][3] =
      {
         {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
         {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
      };
      const int quad_e[6][4] =
      {
         {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
         {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}
      };

      for (int j = 0; j < Nvert; j++)
      {
         mesh->AddVertex(quad_v[j]);
      }
      for (int j = 0; j < Nelem; j++)
      {
         int attribute = j + 1;
         mesh->AddQuad(quad_e[j], attribute);
      }
      mesh->FinalizeQuadMesh(1, 1, true);
   }

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(2, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);
   SnapToSphere(*mesh, r);
   return mesh;
}

// Elliptic Bilinear Solver
class EllipticSolver
{
protected:
   BilinearForm &a;      // LHS
   LinearForm &b;        // RHS
   Array2D<int> ess_bdr; // Component-wise essential boundary marker
   Array<int> ess_tdof_list;
   bool symmetric;
   bool iterative_mode;
   int maxit=10000;
#ifdef MFEM_USE_MPI
   bool parallel; // Flag for ParFiniteElementSpace
   MPI_Comm comm;
#endif
public:
   /// @brief Linear solver for elliptic problem with given Essential BC
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Essential boundary marker for boundary attributes]
   EllipticSolver(BilinearForm &a, LinearForm &b, Array<int> &ess_bdr_);
   /// @brief Linear solver for elliptic problem with given component-wise
   /// essential BC ess_bdr[0,:] - All components, ess_bdr[i,:] - ith-direction
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Component-wise essential boundary marker for boundary
   /// attributes, [Row0: all, Row1: x, ...]
   EllipticSolver(BilinearForm &a, LinearForm &b, Array2D<int> &ess_bdr);

   /// @brief Solve linear system and return FEM solution in x.
   /// @param x FEM solution
   /// @param A_assembled If true, skip assembly of LHS (bilinearform)
   /// @param b_Assembled If true, skip assembly of RHS (linearform)
   /// @return convergence flag
   bool Solve(GridFunction &x, bool A_assembled = false,
              bool b_Assembled = false);
   bool SolveTranspose(GridFunction &x, LinearForm &f, bool A_assembled = false,
                       bool b_Assembled = false);
#ifdef MFEM_USE_MPI
   bool isParallel() { return parallel; }
#endif
   bool isSymmetric() { return symmetric; }
   void SetIterativeMode(bool flag = true) { iterative_mode = flag; };
   void SetMaxIt(int max_it) {maxit = max_it;}

protected:
   /// @brief Get true dofs related to the boundaries in @ess_bdr
   /// @return True dof list
   void GetEssentialTrueDofs();

private:
};

class IGRFluxFunction : public FluxFunction
{
private:
protected:
   FluxFunction &org_flux;
   mutable GridFunctionCoefficient sigma_cf;
public:
   IGRFluxFunction(FluxFunction &fluxfun, GridFunction &sigma)
      : FluxFunction(fluxfun.num_equations, fluxfun.dim), org_flux(fluxfun),
        sigma_cf(&sigma) {}
   /**
    * @brief Compute flux F(u, x) for given state u and physical point x
    *
    * @param[in] state value of state at the current integration point
    * @param[in] Tr element information
    * @param[out] flux F(u, x)
    * @return real_t maximum characteristic speed
    *
    * @note One can put assertion in here to detect non-physical solution
    */
   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &Fu) const override
   {
      real_t org_speed = org_flux.ComputeFlux(state, Tr, Fu);
      real_t sigma_val = sigma_cf.Eval(Tr, Tr.GetIntPoint());
      for (int i = 0; i < dim; i++)
      {
         Fu(i + 1, i) += sigma_val;
      }
      return org_speed + std::fabs(sigma_val);
   }
   /**
    * @brief Compute normal flux. Optionally overloaded in the
    * derived class to avoid creating full dense matrix for flux.
    *
    * @param[in] state state at the current integration point
    * @param[in] normal normal vector, @see CalcOrtho
    * @param[in] Tr face information
    * @param[out] fluxDotN normal flux from the given element at the current
    * integration point
    * @return real_t maximum (normal) characteristic velocity
    */
   real_t ComputeFluxDotN(const Vector &state, const Vector &normal,
                          ElementTransformation &Tr,
                          Vector &fluxDotN) const override
   {
      real_t org_speed = org_flux.ComputeFluxDotN(state, normal, Tr, fluxDotN);
      real_t sigma_val = sigma_cf.Eval(Tr, Tr.GetIntPoint());
      for (int i = 0; i < dim; i++)
      {
         fluxDotN[i + 1] += sigma_val * normal[i];
      }
      return org_speed + std::fabs(sigma_val);
   }
};

class IGRSourceCoeff : public Coefficient
{
public:
   IGRSourceCoeff(Coefficient &alpha, GridFunction &rho, GridFunction &mom)
      : vdim(mom.VectorDim()), alpha(alpha), rho(rho), mom(mom)
   {
#ifndef MFEM_THREAD_SAFE
      Drho.SetSize(vdim);
      Du2.SetSize(vdim);
      Du.SetSize(vdim);
      mom_val.SetSize(vdim);
#endif
   }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
#ifdef MFEM_THREAD_SAFE
      DenseMatrix Du(vdim);
      Vector Drho(vdim), mom_val(vdim);
#endif
      const real_t rho_val = rho.GetValue(T, ip);
      mom.GetVectorValue(T, ip, mom_val);

      rho.GetGradient(T, Drho);
      mom.GetVectorGradient(T, Du);
      Du *= 1.0 / rho_val;
      AddMult_a_VWt(-1.0 / std::pow(rho_val, 2.0), Drho, mom_val, Du);

      const real_t divu = Du.Trace();
    //   MultAtB(Du, Du, Du2);
      Mult(Du, Du, Du2);
      const real_t trDu2 = Du2.Trace();
      // NOTE: TEST SOME AD-HOC
      return alpha.Eval(T, ip) * (std::pow(divu, 2.0) + trDu2);
    //   real_t divu_neg = std::min(0.0, divu);
    //   return 2*alpha.Eval(T, ip) * (divu*divu_neg);
   }

protected:
   int vdim;
   Coefficient &alpha;
   GridFunction &rho, &mom;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix Du, Du2;
   Vector Drho, mom_val;
#endif
};

/// @brief Time dependent DG operator for hyperbolic conservation laws
class IGRDGHyperbolicConservationLaws : public TimeDependentOperator
{
private:
   const int num_equations; // the number of equations
   const int dim;
   FiniteElementSpace &vfes; // vector finite element space
   // Element integration form. Should contain ComputeFlux
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator;
   // Base Nonlinear Form
   std::unique_ptr<NonlinearForm> nonlinearForm;
   // element-wise inverse mass matrix
   std::vector<DenseMatrix> invmass; // local scalar inverse mass
   std::vector<DenseMatrix> weakdiv; // local weak divergence (trial space ByDim)
   // global maximum characteristic speed. Updated by form integrators
   mutable real_t max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;
   VectorCoefficient *dirichlet_cf;

   GridFunction &sigma;
   GridFunctionCoefficient rho_cf;
   std::unique_ptr<RatioCoefficient> sigmaMCoeff;
   std::unique_ptr<RatioCoefficient> sigmaDCoeff;
   std::unique_ptr<IGRSourceCoeff> sigmaFCoeff;
   std::unique_ptr<BilinearForm> sigmaLHS;
   std::unique_ptr<LinearForm> sigmaRHS;
   std::unique_ptr<EllipticSolver> sigmaSolver;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();
   // Compute element-wise weak-divergence matrix
   void ComputeWeakDivergence();

public:
   /**
    * @brief Construct a new IGRDGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    * @param preassembleWeakDivergence preassemble weak divergence for faster
    *                                  assembly
    */
   IGRDGHyperbolicConservationLaws(
      FiniteElementSpace &vfes_,
      Coefficient &alpha, GridFunction &rho, GridFunction &mom, GridFunction &sigma,
      HyperbolicFormIntegrator *formIntegrator_,
      bool preassembleWeakDivergence = true);
   void SetTime(real_t t_) override
   {
      TimeDependentOperator::SetTime(t_);
      if (dirichlet_cf)
      {
         dirichlet_cf->SetTime(t_);
      }
   }
   void SetDirichletBC(VectorCoefficient &cf, Array<int> &ess_bdr)
   {
      dirichlet_cf = &cf;
      formIntegrator->SetDirichletBC(cf, ess_bdr);
      if (formIntegrator.get())
      {
         nonlinearForm->AddBdrFaceIntegrator(formIntegrator.get(), ess_bdr);
      }
   }

   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   void Mult(const Vector &x, Vector &y) const override;
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   real_t GetMaxCharSpeed() { return max_char_speed; }
   void Update();
   HyperbolicFormIntegrator &GetHyperbolicFormIntegrator()
   {
      return *formIntegrator;
   }
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class IGRDGHyperbolicConservationLaws
IGRDGHyperbolicConservationLaws::IGRDGHyperbolicConservationLaws(
   FiniteElementSpace &vfes_,
   Coefficient &alpha, GridFunction &rho, GridFunction &mom, GridFunction &sigma,
   HyperbolicFormIntegrator *formIntegrator_,
   bool preassembleWeakDivergence)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     num_equations(formIntegrator_->num_equations),
     dim(vfes_.GetMesh()->Dimension()), vfes(vfes_),
     formIntegrator(formIntegrator_), z(vfes_.GetTrueVSize()),
     dirichlet_cf(nullptr), sigma(sigma)
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();
#ifndef MFEM_USE_MPI
   nonlinearForm.reset(new NonlinearForm(&vfes));
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(&vfes);
   if (pvfes)
   {
      nonlinearForm.reset(new ParNonlinearForm(pvfes));
   }
   else
   {
      nonlinearForm.reset(new NonlinearForm(&vfes));
   }
#endif
   if (preassembleWeakDivergence)
   {
      ComputeWeakDivergence();
   }
   else
   {
      nonlinearForm->AddDomainIntegrator(formIntegrator.get());
   }
   nonlinearForm->AddInteriorFaceIntegrator(formIntegrator.get());
   nonlinearForm->UseExternalIntegrators();

   FiniteElementSpace &fes_sig = *sigma.FESpace();
#ifndef MFEM_USE_MPI
   sigmaLHS.reset(new BilinearForm(&fes_sig));
   sigmaRHS.reset(new LinearForm(&fes_sig));
#else
   ParFiniteElementSpace *pfes_sig = dynamic_cast<ParFiniteElementSpace*>
                                     (&fes_sig);
   if (pfes_sig)
   {
      sigmaLHS.reset(new ParBilinearForm(pfes_sig));
      sigmaRHS.reset(new ParLinearForm(pfes_sig));
   }
   else
   {
      sigmaLHS.reset(new BilinearForm(&fes_sig));
      sigmaRHS.reset(new LinearForm(&fes_sig));
   }
#endif
   rho_cf.SetGridFunction(&rho);
   sigmaMCoeff.reset(new RatioCoefficient(1.0, rho_cf));
   sigmaDCoeff.reset(new RatioCoefficient(alpha, rho_cf));
   sigmaFCoeff.reset(new IGRSourceCoeff(alpha, rho, mom));
   sigmaLHS->AddDomainIntegrator(new MassIntegrator(*sigmaMCoeff));
//    sigmaLHS->AddDomainIntegrator(new DiffusionIntegrator(*sigmaDCoeff));
   sigmaRHS->AddDomainIntegrator(new DomainLFIntegrator(*sigmaFCoeff));
   Array<int> ess_bdr(0);
   if (pfes_sig)
   {
      if (pfes_sig->GetParMesh()->bdr_attributes.Size())
      {
         ess_bdr.SetSize(pfes_sig->GetParMesh()->bdr_attributes.Max());
         ess_bdr = 0;
      }
   }
   else
   {
      if (fes_sig.GetMesh()->bdr_attributes.Size())
      {
         ess_bdr.SetSize(fes_sig.GetMesh()->bdr_attributes.Max());
         ess_bdr = 0;
      }
   }

   sigmaSolver.reset(new EllipticSolver(*sigmaLHS, *sigmaRHS, ess_bdr));
   sigmaSolver->SetIterativeMode(true);
   // sigmaSolver->SetMaxIt(10);
}

void IGRDGHyperbolicConservationLaws::ComputeInvMass()
{
   InverseIntegrator inv_mass(new MassIntegrator());

   invmass.resize(vfes.GetNE());
   for (int i = 0; i < vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      invmass[i].SetSize(dof);
      inv_mass.AssembleElementMatrix(
         *vfes.GetFE(i), *vfes.GetElementTransformation(i), invmass[i]);
   }
}

void IGRDGHyperbolicConservationLaws::ComputeWeakDivergence()
{
   TransposeIntegrator weak_div(new GradientIntegrator());
   DenseMatrix weakdiv_bynodes;

   weakdiv.resize(vfes.GetNE());
   for (int i = 0; i < vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      weakdiv_bynodes.SetSize(dof, dof * dim);
      weak_div.AssembleElementMatrix2(*vfes.GetFE(i), *vfes.GetFE(i),
                                      *vfes.GetElementTransformation(i),
                                      weakdiv_bynodes);
      weakdiv[i].SetSize(dof, dof * dim);
      // Reorder so that trial space is ByDim.
      // This makes applying weak divergence to flux value simpler.
      for (int j = 0; j < dof; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            weakdiv[i].SetCol(j * dim + d, weakdiv_bynodes.GetColumn(d * dof + j));
         }
      }
   }
}

void IGRDGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator->ResetMaxCharSpeed();
   sigmaSolver->Solve(sigma, false, false);
   // 1. Apply Nonlinear form to obtain an auxiliary result
   //         z = - <F̂(u_h,n), [[v]]>_e
   //    If weak-divergence is not preassembled, we also have weak-divergence
   //         z = - <F̂(u_h,n), [[v]]>_e + (F(u_h), ∇v)
#ifdef MFEM_USE_MPI
   ParGridFunction *psigma = dynamic_cast<ParGridFunction*>(&sigma);
   if (psigma) { psigma->ExchangeFaceNbrData(); }
#endif
   nonlinearForm->Mult(x, z);
   if (!weakdiv.empty()) // if weak divergence is pre-assembled
   {
      // Apply weak divergence to F(u_h), and inverse mass to z_loc + weakdiv_loc
      Vector current_state;     // view of current state at a node
      DenseMatrix current_flux; // flux of current state
      DenseMatrix flux; // element flux value. Whose column is ordered by dim.
      DenseMatrix
      current_xmat; // view of current states in an element, dof x num_eq
      DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
      DenseMatrix current_ymat; // view of element result, dof x num_eq
      const FluxFunction &fluxFunction = formIntegrator->GetFluxFunction();
      Array<int> vdofs;
      Vector xval, zval;
      for (int i = 0; i < vfes.GetNE(); i++)
      {
         ElementTransformation *Tr = vfes.GetElementTransformation(i);
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);
         x.GetSubVector(vdofs, xval);
         current_xmat.UseExternalData(xval.GetData(), dof, num_equations);
         flux.SetSize(num_equations, dim * dof);
         for (int j = 0; j < dof; j++) // compute flux for all nodes in the element
         {
            current_xmat.GetRow(j, current_state);
            current_flux.UseExternalData(flux.GetData() + num_equations * dim * j,
                                         num_equations, dof);
            fluxFunction.ComputeFlux(current_state, *Tr, current_flux);
         }
         // Compute weak-divergence and add it to auxiliary result, z
         // Recalling that weakdiv is reordered by dim, we can apply
         // weak-divergence to the transpose of flux.
         z.GetSubVector(vdofs, zval);
         current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
         mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, current_zmat);
         // Apply inverse mass to auxiliary result to obtain the final result
         current_ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], current_zmat, current_ymat);
         y.SetSubVector(vdofs, current_ymat.GetData());
      }
   }
   else
   {
      // Apply block inverse mass
      Vector zval; // z_loc, dof*num_eq

      DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
      DenseMatrix current_ymat; // view of element result, dof x num_eq
      Array<int> vdofs;
      for (int i = 0; i < vfes.GetNE(); i++)
      {
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);
         z.GetSubVector(vdofs, zval);
         current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
         current_ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], current_zmat, current_ymat);
         y.SetSubVector(vdofs, current_ymat.GetData());
      }
   }
   max_char_speed = formIntegrator->GetMaxCharSpeed();
}

void IGRDGHyperbolicConservationLaws::Update()
{
   nonlinearForm->Update();
   height = nonlinearForm->Height();
   width = height;
   z.SetSize(height);

   ComputeInvMass();
   if (!weakdiv.empty())
   {
      ComputeWeakDivergence();
   }
}

Mesh SWEMesh(const int problem)
{
   switch (problem)
   {
      case 1:
      {
         Mesh mesh("../data/periodic-square.mesh");
         mesh.Transform([](const Vector &x, Vector &y)
         {
            y = x; y *= 20;
         });
         return mesh;
      }
      case 2:
      {
         return Mesh::MakeCartesian1D(4, 2000.0);
      }
      case 3:
      {
          Mesh mesh = Mesh("../data/periodic-segment.mesh");
          mesh.Transform([](const Vector &x, Vector &y){
            y = x;
            y -= 0.5;
            y *= 4000.0;
          });
          return mesh;
      }
      case 4:
      {
          Mesh mesh = Mesh("../data/periodic-segment.mesh");
          mesh.Transform([](const Vector &x, Vector &y){
            y = x;
            y -= 0.5;
            y *= 4000.0;
          });
          return mesh;
      }
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

// Initial condition
VectorFunctionCoefficient SWEInitialCondition(const int problem)
{
   switch (problem)
   {
      case 1: // circular dam break
         return VectorFunctionCoefficient(3, [](const Vector &x, real_t t, Vector &u)
         {
            const real_t sigma = 5;
            const real_t h_min = 6.00;
            const real_t h_max = 10.0;
            u = 0.0;
            u(0) = h_min + (h_max - h_min)*std::exp(-(x*x) / (sigma * sigma));
         });
      case 2: //
         return VectorFunctionCoefficient(2, [](const Vector &x, real_t t, Vector &u)
         {
            const real_t h_L = 10.0;
            const real_t h_R = 5.0;
            u = 0.0;
            u[0] = x[0] < 1000.0 ? h_L : h_R;
         });
      case 3: //
         return VectorFunctionCoefficient(2, [](const Vector &x, real_t t, Vector &u)
         {
            const real_t h_L = 10.0;
            const real_t h_R = 5.0;
            u = 0.0;
            u[0] = std::fabs(x[0]) < 1000.0 ? h_L : h_R;
         });
      case 4: //
         return VectorFunctionCoefficient(2, [](const Vector &x, real_t t, Vector &u)
         {
                    const real_t h_min = 6.0;
           const real_t h_add = 4.0;
           const real_t sigma = 3.0;

           const real_t alpha=5e-2;
           const real_t hl = 10;
           const real_t hr = 0.5 * hl;
           const real_t x0 = x[0];
           u(0) = (hl-hr) * std::exp(-(x0-1000)*alpha)/(1+std::exp(-(x0-1000)*alpha))  * std::exp((x0+1000)*alpha)/(1+std::exp((x0+1000)*alpha)) + hr;

         });
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b,
                               Array<int> &ess_bdr_list)
   : a(a), b(b), ess_bdr(1, ess_bdr_list.Size()), ess_tdof_list(0),
     symmetric(false)
{
   for (int i = 0; i < ess_bdr_list.Size(); i++)
   {
      ess_bdr(0, i) = ess_bdr_list[i];
   }
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace *>(a.FESpace());
   if (pfes)
   {
      parallel = true;
      comm = pfes->GetComm();
   }
   else
   {
      parallel = false;
   }
#endif
   GetEssentialTrueDofs();
}

EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b,
                               Array2D<int> &ess_bdr)
   : a(a), b(b), ess_bdr(ess_bdr), ess_tdof_list(0), symmetric(false),
     iterative_mode(false)
{
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace *>(a.FESpace());
   if (pfes)
   {
      parallel = true;
      comm = pfes->GetComm();
   }
   else
   {
      parallel = false;
   }
#endif
   GetEssentialTrueDofs();
}

void EllipticSolver::GetEssentialTrueDofs()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto pfes = dynamic_cast<ParFiniteElementSpace *>(a.FESpace());
      if (ess_bdr.NumRows() == 1)
      {
         if (ess_bdr.NumCols())
         {
            Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
            pfes->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list);
         }
      }
      else
      {
         if (ess_bdr.NumCols())
         {
            Array<int> ess_tdof_list_comp, ess_bdr_list;
            ess_bdr_list.MakeRef(ess_bdr.GetRow(0), ess_bdr.NumCols());
            pfes->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list_comp, -1);
            ess_tdof_list.Append(ess_tdof_list_comp);

            for (int i = 1; i < ess_bdr.NumRows(); i++)
            {
               ess_bdr_list.MakeRef(ess_bdr.GetRow(i), ess_bdr.NumCols());
               pfes->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list_comp, i - 1);
               ess_tdof_list.Append(ess_tdof_list_comp);
            }
         }
      }
   }
   else
   {

      if (ess_bdr.NumRows() == 1)
      {
         if (ess_bdr.NumCols())
         {
            Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
            a.FESpace()->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list);
         }
      }
      else
      {
         if (ess_bdr.NumCols())
         {
            Array<int> ess_tdof_list_comp, ess_bdr_list;
            ess_bdr_list.MakeRef(ess_bdr.GetRow(0), ess_bdr.NumCols());
            a.FESpace()->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list_comp, -1);
            ess_tdof_list.Append(ess_tdof_list_comp);

            for (int i = 1; i < ess_bdr.NumRows(); i++)
            {
               ess_bdr_list.MakeRef(ess_bdr.GetRow(i), ess_bdr.NumCols());
               a.FESpace()->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list_comp,
                                                 i - 1);
               ess_tdof_list.Append(ess_tdof_list_comp);
            }
         }
      }
   }
#else
   if (ess_bdr.NumRows() == 1)
   {
      if (ess_bdr.NumCols())
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list);
      }
   }
   else
   {
      if (ess_bdr.NumCols())
      {
         Array<int> ess_tdof_list_comp, ess_bdr_list;
         ess_bdr_list.MakeRef(ess_bdr.GetRow(0), ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list_comp, -1);
         ess_tdof_list.Append(ess_tdof_list_comp);

         for (int i = 1; i < ess_bdr.NumRows(); i++)
         {
            ess_bdr_list.MakeRef(ess_bdr.GetRow(i), ess_bdr.NumCols());
            a.FESpace()->GetEssentialTrueDofs(ess_bdr_list, ess_tdof_list_comp,
                                              i - 1);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
      }
   }
#endif
}

bool EllipticSolver::Solve(GridFunction &x, bool A_assembled,
                           bool b_Assembled)
{
   OperatorPtr A;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      A.Reset(new HypreParMatrix);
   }
#endif
   Vector B, X;
   if (!A_assembled)
   {
      a.Update();
      a.Assemble();
   }
   if (!b_Assembled)
   {
      b.Assemble();
   }

   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
   bool converged = true;
#else
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> M;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto M_ptr = new HypreBoomerAMG(static_cast<HypreParMatrix &>(*A));
      M_ptr->SetPrintLevel(0);

      M.reset(M_ptr);
      cg.reset(new CGSolver(comm));
   }
   else
   {
      M.reset(new HypreBoomerAMG(static_cast<HypreParMatrix &>(*A)));
      cg.reset(new CGSolver);
   }
#else
   M.reset(new GSSmoother(static_cast<SparseMatrix &>(*A)));
   cg.reset(new CGSolver);
#endif
   cg->SetRelTol(1e-14);
   cg->SetMaxIter(maxit);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->iterative_mode = iterative_mode;
   cg->Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
   bool converged = cg->GetConverged();
#endif

   return converged;
}

bool EllipticSolver::SolveTranspose(GridFunction &x, LinearForm &f,
                                    bool A_assembled, bool f_Assembled)
{
   OperatorPtr A;
   Vector B, X;

   if (!A_assembled)
   {
      a.Assemble();
   }
   if (!f_Assembled)
   {
      f.Assemble();
   }

   a.FormLinearSystem(ess_tdof_list, x, f, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
   a.RecoverFEMSolution(X, *f, x);
   bool converged = true;
#else
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> M;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M.reset(new HypreBoomerAMG);
      auto M_ptr = new HypreBoomerAMG;
      M_ptr->SetPrintLevel(0);

      M.reset(M_ptr);
      cg.reset(new CGSolver(
                  (dynamic_cast<ParFiniteElementSpace *>(a.FESpace()))->GetComm()));
   }
   else
   {
      M.reset(new GSSmoother((SparseMatrix &)(*A)));
      cg.reset(new CGSolver);
   }
#else
   M.reset(new GSSmoother((SparseMatrix &)(*A)));
   cg.reset(new CGSolver);
#endif
   cg->SetRelTol(1e-14);
   cg->SetMaxIter(maxit);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->iterative_mode = iterative_mode;
   cg->Mult(B, X);
   a.RecoverFEMSolution(X, f, x);
   bool converged = cg->GetConverged();
#endif

   return converged;
}

} // namespace mfem
