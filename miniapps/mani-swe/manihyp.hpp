#include "mfem.hpp"

namespace mfem
{

void sphere(const Vector &x, Vector &y, const real_t r=1.0);

/**
 * @brief Extract orthogornal vector from B not belonging to A.
 *
 * @param A sub-subspace
 * @param B subspace
 * @param n a unit vector in B orthogornal to column space of A.
 */
void CalcOrtho(const DenseMatrix &A, const DenseMatrix &B, Vector &n);

class ManifoldCoord
{
   // attributes
private:
   ElementTransformation *curr_el;
   mutable DenseMatrix mani_vec_state;
   mutable DenseMatrix phys_vec_state;
   mutable Vector normal_comp;
   mutable Vector phys_vec;
protected:
public:
   const int dim;
   const int sdim;

   // methods
private:
protected:
public:
   ManifoldCoord(const int dim, const int sdim):dim(dim), sdim(sdim) {}

   /**
    * @brief Convert manifold state to physical state
    *
    * @param el Target element
    * @param state  Current state value
    * @param phys_state Current Physical state value
    */
   void convertElemState(ElementTransformation &Tr,
                         const int nrScalar, const int nrVector,
                         const Vector &state, Vector &phys_state) const;

   /**
    * @brief Convert left and right states to physical states
    *
    * In physical space, scalar spaces are the same as manifold state
    * However, vector states are translated in local coordinates
    * Basic conversion is v -> Jv
    * To incoporate discontinuity of local coordinates along the interface,
    * we convert state from one element (left) to another element (right) by
    * v -> J1v -> J1v + (n1 dot Jv) (n2 - n1)
    * where J1, n1 are Jacobian and normal vector from one element,
    * and n2 is the normal vector from another element.
    *
    * @param Tr Interface transformation
    * @param nrScalar The number of scalar states
    * @param nrVector The number of vector states
    * @param stateL Input left state
    * @param stateR Input right state
    * @param normalL **outward** normal from the left element
    * @param normalR **inward** normal from the right element
    * @param stateL_L **left** state in the __left__ coordinate system
    * @param stateR_L **right** state in the __left__ coordinate system
    * @param stateL_R **left** state in the __right__ coordinate system
    * @param stateR_R **right** state in the __right__ coordinate system
    */
   void convertFaceState(FaceElementTransformations &Tr,
                         const int nrScalar, const int nrVector,
                         const Vector &stateL, const Vector &stateR,
                         Vector &normalL, Vector &normalR,
                         Vector &stateL_L, Vector &stateR_L,
                         Vector &stateL_R, Vector &stateR_R) const;

};

class ManifoldFlux : public FluxFunction
{
   // attributes
private:
   FluxFunction &org_flux;
   const ManifoldCoord &coord;
   int nrScalar;
   int nrVector;
   mutable Vector phys_state;
protected:
public:

   // methods
private:
protected:
public:
   ManifoldFlux(FluxFunction &flux, ManifoldCoord &coord, int nrScalar)
      : FluxFunction(flux.num_equations, flux.dim), org_flux(flux),
        coord(coord), nrScalar(nrScalar)
   {
      nrVector = (org_flux.num_equations - nrScalar)/coord.sdim;
      phys_state.SetSize(nrScalar + nrVector*coord.sdim);
   }

   const ManifoldCoord &GetCoordinate() const {return coord;}

   /**
    * @brief Compute physical flux from manifold state
    *
    * @param state manifold state
    * @param Tr local element transformation
    * @param flux physical state
    * @return maximum characteristic speed
    */

   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override final;

   real_t ComputeFluxDotN(const Vector &state, const Vector &normal,
                          ElementTransformation &Tr,
                          Vector &fluxDotN) const override final
   {
      MFEM_ABORT("Use ComputeNormalFluxes.");
   }

   real_t ComputeNormalFluxes(const Vector &stateL,
                              const Vector &stateR,
                              FaceElementTransformations &Tr,
                              Vector &normalL, Vector &normalR,
                              Vector &stateL_L, Vector &stateR_L,
                              Vector &fluxL_L, Vector &fluxR_L,
                              Vector &stateL_R, Vector &stateR_R,
                              Vector &fluxL_R, Vector &fluxR_R) const;

   int GetNumScalars() const { return nrScalar; }
};


class ManifoldNumericalFlux : public RiemannSolver
{
   // attributes
private:
protected:
   const ManifoldFlux &maniflux;
   mutable Vector fluxL_L, fluxR_L, fluxL_R, fluxR_R;
   mutable Vector stateL_L, stateR_L, stateL_R, stateR_R;
   mutable Vector normalL, normalR;
public:

   // methods
private:
protected:
public:
   ManifoldNumericalFlux(const ManifoldFlux &flux):RiemannSolver(flux),
      maniflux(flux)
   {
      fluxL_L.SetSize(maniflux.num_equations);
      fluxR_L.SetSize(maniflux.num_equations);
      fluxL_R.SetSize(maniflux.num_equations);
      fluxR_R.SetSize(maniflux.num_equations);
      stateL_L.SetSize(maniflux.num_equations);
      stateR_L.SetSize(maniflux.num_equations);
      stateL_R.SetSize(maniflux.num_equations);
      stateR_R.SetSize(maniflux.num_equations);
      normalL.SetSize(maniflux.GetCoordinate().sdim);
      normalR.SetSize(maniflux.GetCoordinate().sdim);
   }
   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const final { MFEM_ABORT("Use the other Eval function") };
   virtual real_t Eval(const Vector &stateL, const Vector &stateR,
                       FaceElementTransformations &Tr,
                       Vector &hatFL, Vector &hatFR) const = 0;

   const ManifoldFlux &GetManifoldFluxFunction() const {return maniflux;}
   const ManifoldCoord &GetCoordinate() const {return maniflux.GetCoordinate();}
};

class ManifoldRusanovFlux : public ManifoldNumericalFlux
{
   // attributes
private:
protected:
public:

   // methods
private:
protected:
public:
   ManifoldRusanovFlux(const ManifoldFlux &flux):ManifoldNumericalFlux(flux) {}
   virtual real_t Eval(const Vector &stateL, const Vector &stateR,
                       FaceElementTransformations &Tr,
                       Vector &hatFL, Vector &hatFR) const override
   {
#ifdef MFEM_THREAD_SAFE
      Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
      const real_t maxE = maniflux.ComputeNormalFluxes(stateL, stateR, Tr,
                                                       normalL, normalR,
                                                       stateL_L, stateR_L, fluxL_L, fluxR_L,
                                                       stateL_R, stateR_R, fluxL_R, fluxR_R);
      // here, std::sqrt(nor*nor) is multiplied to match the scale with fluxN
      const real_t scaledMaxE = maxE*Tr.Weight();
      for (int i=0; i<maniflux.num_equations; i++)
      {
         hatFL[i] = 0.5*(scaledMaxE*(stateL_L[i] - stateR_L[i]) +
                         (fluxL_L[i] + fluxR_L[i]));
         hatFR[i] = 0.5*(scaledMaxE*(stateL_R[i] - stateR_R[i]) +
                         (fluxL_R[i] + fluxR_R[i]));
      }
      return maxE;
   }
};

class ManifoldVectorMassIntegrator : public BilinearFormIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
   DenseMatrix elmat_comp, elmat_comp_weighted;
   DenseMatrix JtJ;
#endif
   // PA extension
   const FiniteElementSpace *fespace;
   int dim, sdim, ne, nq;

public:
   ManifoldVectorMassIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { }

   /** Given a particular Finite Element computes the element mass matrix
       elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);

};
class ManifoldVectorGradientIntegrator : public BilinearFormIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
   DenseMatrix elmat_comp, elmat_comp_weighted;
   DenseMatrix JtJ;
#endif
   // PA extension
   const FiniteElementSpace *fespace;
   int dim, sdim, ne, nq;

public:
   ManifoldVectorGradientIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { }

   /** Given a particular Finite Element computes the element mass matrix
       elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);

};

class ManifoldHyperbolicFormIntegrator : public NonlinearFormIntegrator
{
   // attributes
private:
protected:
   const ManifoldNumericalFlux &numFlux;
   const ManifoldFlux &maniFlux;
   const ManifoldCoord &coord;
   real_t max_char_speed=0.0;
   Vector state, phys_state;
   Vector stateL, stateR;
   Vector phys_stateL, phys_stateR;
   Vector phys_hatFL, phys_hatFR;
   Vector hatFL, hatFR;
   Vector shape;
   Vector shape1, shape2;
   // DenseMatrix adjJ;
   DenseMatrix dshape;
   DenseMatrix gshape, vector_gshape, vector_gshape_comp;
   DenseMatrix hess_shape;
   DenseTensor HessMat;
   DenseMatrix gradJ;
   Vector x_nodes;
   DenseMatrix phys_flux;
   DenseMatrix phys_flux_scalars, phys_flux_vectors;
   const IntegrationRule *intrule;
   Array<int> hess_map;
   DG_FECollection dg_fec;
public:

   // methods
private:
   const int GetElementIntegratioOrder(ElementTransformation &Trans,
                                       const int order)
   {
      return Trans.OrderJ()+Trans.OrderW()+order;
   }

   const int GetFaceIntegratioOrder(FaceElementTransformations &Trans,
                                    const int orderL, const int orderR)
   {
      return std::max(Trans.Elem1->OrderJ(),
                      Trans.Elem2->Order())+Trans.OrderW() + std::max(orderL, orderR);
   }
   const IntegrationRule &GetRule(const FiniteElement &el1,
                                  const FiniteElement &el2, FaceElementTransformations &Tr)
   {
      return IntRules.Get(Tr.GetGeometryType(),
                          el1.GetOrder() + el2.GetOrder() + Tr.Elem1->OrderJ() + Tr.Elem2->OrderJ() +
                          Tr.OrderW());
   }
   const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                  const FiniteElement &test_fe, ElementTransformation &Trans)
   {
      const int order = trial_fe.GetOrder() + trial_fe.GetOrder() + Trans.OrderW() +
                        Trans.OrderJ()*2;
      return IntRules.Get(trial_fe.GetGeomType(), order);
   }
protected:
public:
   /**
    * @brief Integrator of (F(u), grad v) - <\hat{F}(u), [v]> with given numerical flux.
    * numerical flux both implements F(u) and numerical flux \hat{F}(u).
    *
    * @param flux Numerical flux
    * @param ir Optionally chosen integration rule
    */
   ManifoldHyperbolicFormIntegrator(const ManifoldNumericalFlux &flux,
                                    const IntegrationRule *ir=nullptr);

   // Compute (F(u), grad v)
   void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   // Compute -<\hat{F}(u), [v]> with given numerical flux
   void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect) override;
   // Get maximum characteristic speed for each processor.
   // For parallel assembly, you need to use MPI_Allreduce to synchronize.
   real_t GetMaxCharSpeed() { return max_char_speed; }

   // Set max_char_speed to 0
   void ResetMaxCharSpeed() { max_char_speed=0.0;}

};

class ManifoldStateCoefficient : public VectorCoefficient
{
   // attributes
private:
   VectorCoefficient &phys_cf;
   Vector phys_state;
   DenseMatrix mani_vecs, phys_vecs;
   const int nrScalar, nrVector, dim, sdim;
protected:
public:

   // methods
private:
protected:
public:
   ManifoldStateCoefficient(VectorCoefficient &phys_cf,
                            const int nrScalar, const int nrVector, const int dim)
      :VectorCoefficient(nrScalar + nrVector*dim), phys_cf(phys_cf),
       phys_state(phys_cf.GetVDim()), nrScalar(nrScalar), nrVector(nrVector), dim(dim),
       sdim((phys_cf.GetVDim()-nrScalar)/nrVector)
   {}
   virtual void Eval(Vector &mani_state, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      phys_cf.Eval(phys_state, T, ip);
      for (int i=0; i<nrScalar; i++)
      {
         mani_state[i] = phys_state[i];
      }
      const DenseMatrix& invJ = T.InverseJacobian();
      mani_vecs.UseExternalData(mani_state.GetData() + nrScalar, dim, nrVector);
      phys_vecs.UseExternalData(phys_state.GetData() + nrScalar, sdim, nrVector);
      Mult(invJ, phys_vecs, mani_vecs);
   }

};

class ManifoldPhysVectorCoefficient : public VectorCoefficient
{
   // attributes
private:
   GridFunction &gf;
   const int vid, dim, sdim;
   Vector val;
   Vector val_view;
protected:
public:

   // methods
private:
protected:
public:
   ManifoldPhysVectorCoefficient(GridFunction &gf,
                                 const int vid, const int dim, const int sdim)
      :VectorCoefficient(sdim), gf(gf), vid(vid), dim(dim), sdim(sdim)
   {
      val.SetSize(gf.VectorDim());
   }
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      gf.GetVectorValue(T, ip, val);
      val_view.SetDataAndSize(val.GetData() + vid, dim);
      T.Jacobian().Mult(val_view, V);
   }

};


class CoriolisForce : public VectorCoefficient
{
private:
   const real_t omega; // Coriolis parameter
   ManifoldPhysVectorCoefficient &mom_cf;
   Vector mom; // local momentum
   Vector normal; // surface normal
   Vector x;
   Vector V_phys;
   Vector V_mani;
public:
   CoriolisForce(ManifoldPhysVectorCoefficient &mom_cf,
                 const real_t omega):VectorCoefficient(3), omega(omega), mom_cf(mom_cf),
      mom(mom_cf.GetVDim()), normal(mom_cf.GetVDim()), x(mom_cf.GetVDim()),
      V_phys(mom_cf.GetVDim()), V_mani(mom_cf.GetVDim()-1)
   {
      MFEM_ASSERT(mom_cf.GetVDim() == 3, "Momentum should be 3D");
   }
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      V = 0.0;
      T.Transform(T.GetIntPoint(), x);
      const real_t theta = std::acos(x[2] / std::sqrt(x*x));
      const real_t f = 2*omega*std::sin(theta);
      mom_cf.Eval(mom, T, ip);
      CalcOrtho(T.Jacobian(), normal);
      normal /= std::sqrt(normal*normal);
      normal.cross3D(mom, V_phys);
      V_mani.SetData(V.GetData() + 1);
      T.Jacobian().MultTranspose(V_phys, V_mani);
      V_mani *= f;
   }
};

class ManifoldDGHyperbolicConservationLaws : public TimeDependentOperator
{
private:
   const int dim, sdim, nrScalar, nrVector;
   FiniteElementSpace &vfes; // vector finite element space
   // Element integration form. Should contain ComputeFlux
   ManifoldHyperbolicFormIntegrator &formIntegrator;
   // Base Nonlinear Form
   std::unique_ptr<NonlinearForm> nonlinearForm;
   std::unique_ptr<LinearForm> force;
   // element-wise inverse mass matrix
   int int_offset;
   std::vector<DenseMatrix> invmass; // local scalar inverse mass
   std::vector<DenseMatrix> invmass_vec; // local scalar inverse mass
   std::vector<DenseMatrix> weakdiv; // local weak divergence
   std::vector<DenseMatrix> weakdiv_vec; // local weak divergence
   // global maximum characteristic speed. Updated by form integrators
   mutable real_t max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();
   // Compute element-wise weak-divergence matrix
   void ComputeWeakDivergence();
   bool parallel = false;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
public:
   /**
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    * @param preassembleWeakDivergence preassemble weak divergence for faster
    *                                  assembly
    */
   ManifoldDGHyperbolicConservationLaws(
      FiniteElementSpace &vfes,
      ManifoldHyperbolicFormIntegrator &formIntegrator,
      const int nrScalar,
      const int order_offset=3);
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
   void AddForce(LinearFormIntegrator *lfdi)
   {
      if (force)
      {
         force->AddDomainIntegrator(lfdi);
      }
      else
      {
         if (parallel)
         {
#ifdef MFEM_USE_MPI
            ParFiniteElementSpace *pvfes = static_cast<ParFiniteElementSpace*>(&vfes);
            force.reset(new ParLinearForm(pvfes, z.GetData()));
#endif
         }
         else
         {
            force.reset(new LinearForm(&vfes, z.GetData()));
         }
         force->AddDomainIntegrator(lfdi);
      }
   }
};


} // end of namespace mfem