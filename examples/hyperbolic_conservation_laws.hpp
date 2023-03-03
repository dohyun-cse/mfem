//                  MFEM Example 18 - Serial/Parallel Shared Code

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Problem definition
// extern int problem;

// Maximum characteristic speed (updated by integrators)

// Abstract Numerical flux hat{F}(u-, u+).
// Eval: state (u-, u+), flux (Fu-, Fu+), speed, face normal |-> flux
class NumericalFlux {
 public:
  NumericalFlux(){};
  virtual void Eval(const Vector &state1, const Vector &state2,
                    const Vector &flux1, const Vector &flux2, const double maxE,
                    const Vector &nor, Vector &flux) {
    mfem_error("Not Implemented");
  }
};

/// @brief (F(u), grad v) nonlinear form integrator where the abstract method
/// ComputeFlux: const Vector &state = u -> const DenseMatrix flux = F(u) should
/// be overloaded.
class DGHyperbolicFormIntegrator : public NonlinearFormIntegrator {
 private:
  double max_char_speed;
  int IntOrderOffset;

 protected:
  virtual double ComputeFlux(const Vector &state, DenseMatrix &flux) = 0;

 public:
  DGHyperbolicFormIntegrator(const IntegrationRule *ir = NULL)
      : NonlinearFormIntegrator(ir), IntOrderOffset(0){};
  DGHyperbolicFormIntegrator(const int IntOrderOffset_ = 3)
      : NonlinearFormIntegrator(NULL), IntOrderOffset(IntOrderOffset_){};

  const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                 const FiniteElement &test_fe) {
    int order;
    order = trial_fe.GetOrder() + test_fe.GetOrder() + IntOrderOffset;
    return IntRules.Get(trial_fe.GetGeomType(), order);
  }

  void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, Vector &elvect);
  inline double getMaxCharSpeed() { return max_char_speed; }
  inline void resetMaxCharSpeed() { max_char_speed = 0.0; }
};

// Interior face term: <hat{F}.n,[w]>
// where hat{F}.n is determined by NumericalFlux rsolver.
class FaceIntegrator : public NonlinearFormIntegrator {
 private:
  const int num_equations;
  double max_char_speed;
  const int IntOrderOffset;
  NumericalFlux *rsolver;
  Vector shape1;
  Vector shape2;
  Vector funval1;
  Vector funval2;
  Vector flux1;
  Vector flux2;
  Vector nor;
  Vector fluxN;

 protected:
  virtual double ComputeFluxDotN(const Vector &state, const Vector &nor,
                                 Vector &flux) = 0;

 public:
  FaceIntegrator(NumericalFlux *rsolver_, const int dim,
                 const int num_equations_, const int IntOrderOffset_ = 3)
      : NonlinearFormIntegrator(),
        num_equations(num_equations_),
        max_char_speed(0.0),
        IntOrderOffset(IntOrderOffset_),
        rsolver(rsolver_),
        funval1(num_equations_),
        funval2(num_equations_),
        flux1(num_equations_),
        flux2(num_equations_),
        nor(dim),
        fluxN(num_equations_){};
  FaceIntegrator(NumericalFlux *rsolver_, const int dim,
                 const int num_equations_, const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir),
        num_equations(num_equations_),
        max_char_speed(0.0),
        IntOrderOffset(0),
        rsolver(rsolver_),
        funval1(num_equations_),
        funval2(num_equations_),
        flux1(num_equations_),
        flux2(num_equations_),
        nor(dim),
        fluxN(num_equations_){};

  const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                 const FiniteElement &test_fe) {
    int order;
    order = trial_fe.GetOrder() + test_fe.GetOrder() + IntOrderOffset;
    return IntRules.Get(trial_fe.GetGeomType(), order);
  }
  inline double getMaxCharSpeed() { return max_char_speed; }
  inline void resetMaxCharSpeed() { max_char_speed = 0.0; }

  virtual void AssembleFaceVector(const FiniteElement &el1,
                                  const FiniteElement &el2,
                                  FaceElementTransformations &Tr,
                                  const Vector &elfun, Vector &elvect);
  virtual ~FaceIntegrator() {}
};

// Base Hyperbolic conservation law class.
// This contains all methods needed except the flux function.
class DGHyperbolicConservationLaws : public TimeDependentOperator {
 private:
  const int dim;
  const int num_equations;

  FiniteElementSpace
      &vfes;  // Vector finite element space containing conserved variables
  FaceIntegrator &faceIntegrator;  // Face integration form. Should contain
                                   // ComputeFluxDotN and Riemann Solver
  NonlinearForm faceForm;
  MixedBilinearForm &divA;  // Element integration form, (u, grad V) where u is
                            // scalar, V is vector
  std::vector<DenseMatrix> Me_inv;  // element-wise inverse mass matrix

  // auxiliary variables
  mutable Vector state;
  mutable DenseMatrix f;
  mutable DenseTensor flux;
  mutable Vector z;
  mutable double max_char_speed;

  // Get flux value for given states for all elements
  void GetFlux(const DenseMatrix &state_, DenseTensor &flux_) const;
  // Compute element-wise inverse mass matrix
  void ComputeInvMass();

 protected:
  // Compute flux for given states at a node
  // WARNING: This should be implemented in the sub-class
  virtual double ComputeFlux(const Vector &state, const int dim,
                             DenseMatrix &flux) const = 0;

 public:
  // Constructor
  DGHyperbolicConservationLaws(FiniteElementSpace &vfes_, MixedBilinearForm &divA,
                             FaceIntegrator &faceForm_,
                             const int num_equations_);
  // Apply M\(DIV F(U) + JUMP HAT{F}(U))
  virtual void Mult(const Vector &x, Vector &y) const;
  // Update operators when mesh and finite element spaces are updated
  void Update();
  inline double getMaxCharSpeed() { return max_char_speed; };

  virtual ~DGHyperbolicConservationLaws() {}
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
    FiniteElementSpace &vfes_, MixedBilinearForm &divA_,
    FaceIntegrator &faceIntegrator_, const int num_equations_)
    : TimeDependentOperator(vfes_.GetNDofs() * num_equations_),
      dim(vfes_.GetFE(0)->GetDim()),
      num_equations(num_equations_),
      vfes(vfes_),
      faceIntegrator(faceIntegrator_),
      faceForm(&vfes),
      divA(divA_),
      Me_inv(0),
      state(num_equations),
      f(num_equations, dim),
      flux(vfes.GetNDofs(), dim, num_equations),
      z(vfes_.GetNDofs() * num_equations_),
      max_char_speed(0.0) {
  // Standard local assembly and inversion for energy mass matrices.
  divA.Assemble();
  faceForm.AddInteriorFaceIntegrator(&faceIntegrator);
  ComputeInvMass();

  height = z.Size();
  width = z.Size();
}

void DGHyperbolicConservationLaws::ComputeInvMass() {
  DenseMatrix Me;
  MassIntegrator mi;
  Me_inv.resize(vfes.GetNE());
  for (int i = 0; i < vfes.GetNE(); i++) {
    Me.SetSize(vfes.GetFE(i)->GetDof());
    mi.AssembleElementMatrix(*vfes.GetFE(i), *vfes.GetElementTransformation(i),
                             Me);
    DenseMatrixInverse inv(&Me);
    inv.Factor();
    inv.GetInverseMatrix(Me_inv[i]);
  }
}

void DGHyperbolicConservationLaws::Update() {
  faceForm.Update();
  divA.Update();
  divA.Assemble();
  ComputeInvMass();

  width = faceForm.Width();
  height = faceForm.Height();
  flux.SetSize(vfes.GetNDofs(), dim, num_equations);
  z.SetSize(height);
}

void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const {
  // 0. Reset wavespeed computation before operator application.
  max_char_speed = 0.;
  faceIntegrator.resetMaxCharSpeed();
  // 1. Create the vector z with the face terms -<F.n(u), [w]>.
  faceForm.Mult(x, z);

  // 2. Add the element terms.
  // i.  computing the flux approximately as a grid function by interpolating
  //     at the solution nodes.
  // ii. multiplying this grid function by a (constant) mixed bilinear form for
  //     each of the num_equations, computing (F(u), grad(w)) for each equation.

  DenseMatrix xmat(x.GetData(), vfes.GetNDofs(), num_equations);
  GetFlux(xmat, flux);
  max_char_speed = max(max_char_speed, faceIntegrator.getMaxCharSpeed());

  for (int k = 0; k < num_equations; k++) {
    Vector fk(flux(k).GetData(), dim * vfes.GetNDofs());
    Vector zk(z.GetData() + k * vfes.GetNDofs(), vfes.GetNDofs());
    divA.SpMat().AddMult(fk, zk);
  }

  // 3. Multiply element-wise by the inverse mass matrices.
  Vector zval;
  Array<int> vdofs;
  //   const int dof = vfes.GetFE(0)->GetDof();
  DenseMatrix zmat, ymat;

  for (int i = 0; i < vfes.GetNE(); i++) {
    // Return the vdofs ordered byNODES
    vfes.GetElementVDofs(i, vdofs);
    z.GetSubVector(vdofs, zval);
    zmat.UseExternalData(zval.GetData(), vfes.GetFE(i)->GetDof(),
                         num_equations);
    ymat.SetSize(Me_inv[i].Height(), num_equations);
    mfem::Mult(Me_inv[i], zmat, ymat);
    y.SetSubVector(vdofs, ymat.GetData());
  }
}

// Compute the flux at solution nodes.
void DGHyperbolicConservationLaws::GetFlux(const DenseMatrix &x_,
                                         DenseTensor &flux_) const {
  const int flux_dof = flux_.SizeI();
  const int flux_dim = flux_.SizeJ();

  for (int i = 0; i < flux_dof; i++) {
    for (int k = 0; k < num_equations; k++) {
      state(k) = x_(i, k);
    }
    const double mcs = ComputeFlux(state, flux_dim, f);

    for (int d = 0; d < flux_dim; d++) {
      for (int k = 0; k < num_equations; k++) {
        flux_(i, d, k) = f(k, d);
      }
    }

    // Update max char speed
    // const double mcs = ComputeMaxCharSpeed(state, flux_dim);
    if (mcs > max_char_speed) {
      max_char_speed = mcs;
    }
  }
}

void DGHyperbolicFormIntegrator::AssembleElementVector(
    const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
    Vector &elvect) {
  const int dof = el.GetDof();
  const int dim = el.GetDim();
  const int num_equations = elfun.Size() / dof;

  Vector shape(dof);
  Vector state(num_equations);
  DenseMatrix dshape(dof, dim);
  DenseMatrix flux(num_equations, dim);

  const DenseMatrix elfunmat(elfun.GetData(), dof, num_equations);
  DenseMatrix elvectmat(elvect.GetData(), dof, num_equations);

  elvect.SetSize(dof);
  elvect = 0.0;

  const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    Tr.SetIntPoint(&ip);

    el.CalcShape(ip, shape);
    elfunmat.Mult(shape, state);
    const double mcs = ComputeFlux(shape, flux);
    max_char_speed = mcs > max_char_speed ? mcs : max_char_speed;

    el.CalcPhysDShape(Tr, dshape);
    AddMult_a_ABt(ip.weight * Tr.Weight(), dshape, flux, elvectmat);
  }
}

//////////////////////////////////////////////////////////////////
///                       FACE INTEGRATOR                      ///
//////////////////////////////////////////////////////////////////

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect) {
  // Compute the term <F.n(u),[w]> on the interior faces.
  const int dof1 = el1.GetDof();
  const int dof2 = el2.GetDof();

  shape1.SetSize(dof1);
  shape2.SetSize(dof2);

  elvect.SetSize((dof1 + dof2) * num_equations);
  elvect = 0.0;

  DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
  DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
                         num_equations);

  DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
  DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2,
                          num_equations);

  const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el1, el2);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Tr.SetAllIntPoints(&ip);  // set face and element int. points

    // Calculate basis functions on both elements at the face
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
    el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

    // Interpolate elfun at the point
    elfun1_mat.MultTranspose(shape1, funval1);
    elfun2_mat.MultTranspose(shape2, funval2);

    // Get the normal vector and the flux on the face
    if (nor.Size() == 1)
      nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
    else
      CalcOrtho(Tr.Jacobian(), nor);

    const double mcs = max(ComputeFluxDotN(funval1, nor, flux1),
                           ComputeFluxDotN(funval2, nor, flux2));
    rsolver->Eval(funval1, funval2, flux1, flux2, mcs, nor, fluxN);

    // Update max char speed
    if (mcs > max_char_speed) {
      max_char_speed = mcs;
    }

    fluxN *= ip.weight;
    for (int k = 0; k < num_equations; k++) {
      for (int s = 0; s < dof1; s++) {
        elvect1_mat(s, k) -= fluxN(k) * shape1(s);
      }
      for (int s = 0; s < dof2; s++) {
        elvect2_mat(s, k) += fluxN(k) * shape2(s);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////
///                      NUMERICAL FLUXES                      ///
//////////////////////////////////////////////////////////////////

// Rusanov Flux
class RusanovFlux : public NumericalFlux {
 public:
  void Eval(const Vector &state1, const Vector &state2, const Vector &flux1,
            const Vector &flux2, const double maxE, const Vector &nor,
            Vector &flux) {
    // NOTE: nor in general is not a unit normal

    flux = 0.0;
    flux += state1;
    flux -= state2;
    flux *= maxE * sqrt(nor * nor);
    flux += flux1;
    flux += flux2;
    flux *= 0.5;
  }
};

// Upwind Flux, Not Yet Implemented
class UpwindFlux : public NumericalFlux {
 public:
  void Eval(const Vector &state1, const Vector &state2, const Vector &flux1,
            const Vector &flux2, const double maxE, const Vector &nor,
            Vector &flux) {
    // NOTE: nor in general is not a unit normal
    mfem_error("Not Implemented");
  }
};

//////////////////////////////////////////////////////////////////
///                        EULER SYSTEM                        ///
//////////////////////////////////////////////////////////////////

// Euler System main class. Overload ComputeFlux
class EulerSystem : public DGHyperbolicConservationLaws {
 private:
  const double specific_heat_ratio;
  const double gas_constant;
  double ComputeFlux(const Vector &state, const int dim,
                     DenseMatrix &flux) const {
    const double den = state(0);
    const Vector den_vel(state.GetData() + 1, dim);
    const double den_energy = state(1 + dim);

    const double pres = (specific_heat_ratio - 1.0) *
                        (den_energy - 0.5 * (den_vel * den_vel) / den);

    MFEM_ASSERT(den >= 0, "Negative Density");
    MFEM_ASSERT(pres >= 0, "Negative Pressure");
    MFEM_ASSERT(den_energy >= 0, "Negative Energy");

    for (int d = 0; d < dim; d++) {
      flux(0, d) = den_vel(d);
      for (int i = 0; i < dim; i++) {
        flux(1 + i, d) = den_vel(i) * den_vel(d) / den;
      }
      flux(1 + d, d) += pres;
    }

    const double H = (den_energy + pres) / den;
    for (int d = 0; d < dim; d++) {
      flux(1 + dim, d) = den_vel(d) * H;
    }

    const double sound = sqrt(specific_heat_ratio * pres / den);
    const double vel = sqrt(den_vel * den_vel) / den;

    return vel + sound;
  }

 public:
  EulerSystem(FiniteElementSpace &vfes_, MixedBilinearForm &divA_,
              FaceIntegrator &faceForm_,
              const double specific_heat_ratio_ = 1.4,
              const double gas_constant_ = 1.0)
      : DGHyperbolicConservationLaws(vfes_, divA_, faceForm_,
                                   vfes_.GetMesh()->Dimension() + 2),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_){};
};

// Euler System face integration. Overload ComputeFluxDotN
class EulerFaceIntegrator : public FaceIntegrator {
 private:
  const double specific_heat_ratio;
  const double gas_constant;
  double ComputeFluxDotN(const Vector &state, const Vector &nor,
                         Vector &fluxN) {
    // NOTE: nor in general is not a unit normal
    const int dim = nor.Size();
    const double den = state(0);
    const Vector den_vel(state.GetData() + 1, dim);
    const double den_energy = state(1 + dim);

    const double pres = (specific_heat_ratio - 1.0) *
                        (den_energy - 0.5 * (den_vel * den_vel) / den);

    MFEM_ASSERT(den >= 0, "Negative Density");
    MFEM_ASSERT(pres >= 0, "Negative Pressure");
    MFEM_ASSERT(den_energy >= 0, "Negative Energy");

    double den_velN = 0;
    for (int d = 0; d < dim; d++) {
      den_velN += den_vel(d) * nor(d);
    }

    fluxN(0) = den_velN;
    for (int d = 0; d < dim; d++) {
      fluxN(1 + d) = den_velN * den_vel(d) / den + pres * nor(d);
    }

    const double H = (den_energy + pres) / den;
    fluxN(1 + dim) = den_velN * H;

    const double sound = sqrt(specific_heat_ratio * pres / den);
    const double vel = sqrt(den_vel * den_vel) / den;

    return vel + sound;
  }

 public:
  EulerFaceIntegrator(NumericalFlux *rsolver_, const int dim,
                      const double specific_heat_ratio_ = 1.4,
                      const double gas_constant_ = 1.0)
      : FaceIntegrator(rsolver_, dim, dim + 2),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_){};
};

//////////////////////////////////////////////////////////////////
///                      BURGERS EQUATION                      ///
//////////////////////////////////////////////////////////////////

// Burgers equation main class. Overload ComputeFlux
class BurgersEquation : public DGHyperbolicConservationLaws {
 private:
  double ComputeFlux(const Vector &state, const int dim,
                     DenseMatrix &flux) const {
    flux = state * state * 0.5;
    return abs(state(0));
  };

 public:
  BurgersEquation(FiniteElementSpace &vfes_, MixedBilinearForm &divA_,
                  FaceIntegrator &faceForm_)
      : DGHyperbolicConservationLaws(vfes_, divA_, faceForm_, 1){};
};

// Burgers equation face integration. Overload ComputeFluxDotN
class BurgersFaceIntegrator : public FaceIntegrator {
 private:
  double ComputeFluxDotN(const Vector &state, const Vector &nor,
                         Vector &fluxN) {
    fluxN = nor.Sum() * (state * state) * 0.5;
    return abs(state(0));
  };

 public:
  BurgersFaceIntegrator(NumericalFlux *rsolver_, const int dim)
      : FaceIntegrator(rsolver_, dim, 1){};
};

//////////////////////////////////////////////////////////////////
///                        SHALLOW WATER                       ///
//////////////////////////////////////////////////////////////////

// Burgers equation main class. Overload ComputeFlux
class ShallowWater : public DGHyperbolicConservationLaws {
 private:
  const double g;
  double ComputeFlux(const Vector &state, const int dim,
                     DenseMatrix &flux) const {
    const double height = state(0);
    const Vector h_vel(state.GetData() + 1, dim);

    const double energy = 0.5 * g * (height * height);

    MFEM_ASSERT(height >= 0, "Negative Height");

    for (int d = 0; d < dim; d++) {
      flux(0, d) = h_vel(d);
      for (int i = 0; i < dim; i++) {
        flux(1 + i, d) = h_vel(i) * h_vel(d) / height;
      }
      flux(1 + d, d) += energy;
    }

    const double sound = sqrt(g * height);
    const double vel = sqrt(h_vel * h_vel) / height;

    return vel + sound;
  };

 public:
  ShallowWater(FiniteElementSpace &vfes_, MixedBilinearForm &divA_,
               FaceIntegrator &faceForm_, const double g_ = 9.81)
      : DGHyperbolicConservationLaws(vfes_, divA_, faceForm_,
                                   1 + vfes_.GetFE(0)->GetDim()),
        g(g_){};
};

// Burgers equation face integration. Overload ComputeFluxDotN
class ShallowWaterFaceIntegrator : public FaceIntegrator {
 private:
  const double g;
  double ComputeFluxDotN(const Vector &state, const Vector &nor,
                         Vector &fluxN) {
    const int dim = nor.Size();
    const double height = state(0);
    const Vector h_vel(state.GetData() + 1, dim);

    const double energy = 0.5 * g * (height * height);

    MFEM_ASSERT(height >= 0, "Negative Height");
    fluxN(0) = h_vel * nor;
    const double normal_vel = fluxN(0) / height;
    for (int i = 0; i < dim; i++) {
      fluxN(1 + i) = normal_vel * h_vel(i) + energy * nor(i);
    }

    const double sound = sqrt(g * height);
    const double vel = sqrt(h_vel * h_vel) / height;

    return vel + sound;
  };

 public:
  ShallowWaterFaceIntegrator(NumericalFlux *rsolver_, const int dim,
                             const double g_ = 9.81)
      : FaceIntegrator(rsolver_, dim, 1 + dim), g(g_){};
};