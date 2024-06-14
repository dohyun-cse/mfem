//                  MFEM Example 18 - Serial/Parallel Shared Code
//                      (Implementation of Time-dependent DG Operator)
//
// This code provide example problems for the Euler equations and implements
// the time-dependent DG operator given by the equation:
//
//            (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u, n), [[v]]>_F = 0.
//
// This operator is designed for explicit time stepping methods. Specifically,
// the function DGHyperbolicConservationLaws::Mult implements the following
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
// details, refer to the documentation of DGHyperbolicConservationLaws::Mult.
//

#include <functional>
#include "mfem.hpp"

namespace mfem
{

class ManifoldFlux : public FluxFunction
{
private:
   FluxFunction &fluxFunction;
   mutable DenseMatrix FU_tan;
   mutable DenseMatrix jacob2ortho;
   mutable DenseMatrix ortho2jacob;
   mutable Vector state_tan;
   const int offset;

public:
   ManifoldFlux(FluxFunction &fluxFunction, const int sdim, const int offset)
      :FluxFunction(fluxFunction.num_equations, fluxFunction.dim, fluxFunction.sdim),
       fluxFunction(fluxFunction), FU_tan(num_equations, dim), offset(offset),
       jacob2ortho(dim)
   {
      MFEM_ASSERT(sdim == 2 &&
                  dim == 3, "Currently, it only supports 2D surface in 3D");
   }

   FluxFunction &GetFluxFunction() {return fluxFunction;}

   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override
   {
      // Update coordinate
      Jacobian2OrthoCoord(Tr);
      // state in orthogonal coordinate
      state_tan = state;
      Vector vec_Ortho, vec_Jacob;
      vec_Ortho.SetDataAndSize(state_tan.GetData() + offset, sdim);
      vec_Jacob.SetDataAndSize(state.GetData() + offset, sdim);
      jacob2ortho.Mult(vec_Jacob, vec_Ortho);
      // Compute flux in orthogonal coordinate
      real_t speed = fluxFunction.ComputeFlux(state_tan, Tr, FU_tan);
      // revert it back to jacobian coordinate
      MultABt(FU_tan, ortho2jacob, flux);
      return speed;
   }

   void Jacobian2OrthoCoord(ElementTransformation &Tr,
                            DenseMatrix *result = nullptr,
                            DenseMatrix *inv_result = nullptr) const
   {
      // obtain local coordinate
      DenseMatrix jacob = Tr.Jacobian();
      Vector dir1, dir2;
      jacob.GetColumnReference(0, dir1);
      jacob.GetColumnReference(1, dir2);
      // orthogonalize
      real_t dir1_mag_squared = dir1*dir1;
      real_t dir2_mag_squared = dir2*dir2;
      Vector t2(dir2); // second tangent direction
      real_t d1_coeff = -(dir1*dir2)/dir1_mag_squared;
      t2.Add(d1_coeff, dir1);
      real_t t2_norm = std::sqrt(t2*t2);

      jacob2ortho(0,0) = 1.0 / std::sqrt(dir1_mag_squared);
      jacob2ortho(1,0) = d1_coeff / t2_norm;
      jacob2ortho(1,1) = 1.0 / t2_norm;

      // from jacobian coordinate to orthogonal coordinate
      ortho2jacob = jacob2ortho;
      // from orthogonal coordinate to jacobian coordinate
      ortho2jacob.Invert();

      if (result) { result = &jacob2ortho; }
      if (inv_result) { inv_result = &ortho2jacob; }
   }
};


class ManifoldRiemannSolver : public RiemannSolver
{
private:
   RiemannSolver &rsolver;
   const ManifoldFlux &maniflux;
   const int sdim;
   const int offset;
#ifndef MFEM_THREAD_SAFE
   mutable Vector tangent;
   mutable Vector n_L, n_R;
   mutable DenseMatrix rot_L, rot_R;
   mutable Vector state_L, state_R;
   mutable Vector vec_T;
   mutable Vector vec_L, vec_R;
#endif

public:
   ManifoldRiemannSolver(const ManifoldFlux &maniflux, int offset,
                         RiemannSolver &rsolver)
      : RiemannSolver(maniflux), rsolver(rsolver), maniflux(maniflux),
        sdim(maniflux.sdim), offset(offset), vec_T(maniflux.dim)
   {
#ifndef MFEM_THREAD_SAFE
      tangent.SetSize(sdim);
      n_L.SetSize(sdim);
      n_R.SetSize(sdim);
      state_L.SetSize(fluxFunction.num_equations);
      state_R.SetSize(fluxFunction.num_equations);
      rot_L.SetSize(maniflux.dim);
      rot_R.SetSize(maniflux.dim);
      vec_L.SetSize(fluxFunction.dim);
      vec_R.SetSize(fluxFunction.dim);
#endif
   }
   virtual real_t Eval(const Vector &state1, const Vector &state2,
                       const Vector &nor, FaceElementTransformations &Tr,
                       Vector &flux) const

   {

      return 0.0;
   }
};


/// @brief Time dependent DG operator for hyperbolic conservation laws
class DGHyperbolicConservationLaws : public TimeDependentOperator
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

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();
   // Compute element-wise weak-divergence matrix
   void ComputeWeakDivergence();

public:
   /**
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    * @param preassembleWeakDivergence preassemble weak divergence for faster
    *                                  assembly
    */
   DGHyperbolicConservationLaws(
      FiniteElementSpace &vfes_,
      std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
      bool preassembleWeakDivergence=true);
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

};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace &vfes_,
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
   bool preassembleWeakDivergence)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     num_equations(formIntegrator_->num_equations),
     dim(vfes_.GetMesh()->SpaceDimension()),
     vfes(vfes_),
     formIntegrator(std::move(formIntegrator_)),
     z(vfes_.GetTrueVSize())
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

}

void DGHyperbolicConservationLaws::ComputeInvMass()
{
   InverseIntegrator inv_mass(new MassIntegrator());

   invmass.resize(vfes.GetNE());
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      invmass[i].SetSize(dof);
      inv_mass.AssembleElementMatrix(*vfes.GetFE(i),
                                     *vfes.GetElementTransformation(i),
                                     invmass[i]);
   }
}

void DGHyperbolicConservationLaws::ComputeWeakDivergence()
{
   TransposeIntegrator weak_div(new GradientIntegrator());
   DenseMatrix weakdiv_bynodes;

   weakdiv.resize(vfes.GetNE());
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      weakdiv_bynodes.SetSize(dof, dof*dim);
      weak_div.AssembleElementMatrix2(*vfes.GetFE(i), *vfes.GetFE(i),
                                      *vfes.GetElementTransformation(i),
                                      weakdiv_bynodes);
      weakdiv[i].SetSize(dof, dof*dim);
      // Reorder so that trial space is ByDim.
      // This makes applying weak divergence to flux value simpler.
      for (int j=0; j<dof; j++)
      {
         for (int d=0; d<dim; d++)
         {
            weakdiv[i].SetCol(j*dim + d, weakdiv_bynodes.GetColumn(d*dof + j));
         }
      }

   }
}


void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator->ResetMaxCharSpeed();
   // 1. Apply Nonlinear form to obtain an auxiliary result
   //         z = - <F̂(u_h,n), [[v]]>_e
   //    If weak-divergence is not preassembled, we also have weak-divergence
   //         z = - <F̂(u_h,n), [[v]]>_e + (F(u_h), ∇v)
   nonlinearForm->Mult(x, z);
   if (!weakdiv.empty()) // if weak divergence is pre-assembled
   {
      // Apply weak divergence to F(u_h), and inverse mass to z_loc + weakdiv_loc
      Vector current_state; // view of current state at a node
      DenseMatrix current_flux; // flux of current state
      DenseMatrix flux; // element flux value. Whose column is ordered by dim.
      DenseMatrix current_xmat; // view of current states in an element, dof x num_eq
      DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
      DenseMatrix current_ymat; // view of element result, dof x num_eq
      const FluxFunction &fluxFunction = formIntegrator->GetFluxFunction();
      Array<int> vdofs;
      Vector xval, zval;
      for (int i=0; i<vfes.GetNE(); i++)
      {
         ElementTransformation* Tr = vfes.GetElementTransformation(i);
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);
         x.GetSubVector(vdofs, xval);
         current_xmat.UseExternalData(xval.GetData(), dof, num_equations);
         flux.SetSize(num_equations, dim*dof);
         for (int j=0; j<dof; j++) // compute flux for all nodes in the element
         {
            current_xmat.GetRow(j, current_state);
            current_flux.UseExternalData(flux.GetData() + num_equations*dim*j,
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
      for (int i=0; i<vfes.GetNE(); i++)
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

void DGHyperbolicConservationLaws::Update()
{
   nonlinearForm->Update();
   height = nonlinearForm->Height();
   width = height;
   z.SetSize(height);

   ComputeInvMass();
   if (!weakdiv.empty()) {ComputeWeakDivergence();}
}

Mesh SWEMesh(const int problem)
{
   switch (problem)
   {
      case 1:
      {
         Mesh mesh("../data/square-sphere.mesh");
         mesh.Transform([](const Vector &x, Vector &y)
         {
            y = x; y *= 20;
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
      case 1: // fast moving vortex
         return VectorFunctionCoefficient(3, [](const Vector &x, Vector &u)
         {
            const real_t sigma = 5;
            const real_t h_min = 6.00;
            const real_t h_max = 10.0;
            Vector northPole({0.0, 0.0, 20.0});
            u = 0.0;
            u(0) = h_min + (h_max - h_min)*std::exp(-x.DistanceSquaredTo(northPole) /
                                                    (sigma * sigma));
         });
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

} // namespace mfem
