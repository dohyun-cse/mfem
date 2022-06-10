#include "mfem.hpp"
#include <random>
#include <vector>

namespace mfem {

// ===========================================================================
// Header interface
// ===========================================================================

/// Fills the vector x with random numbers between a and b.
void FillWithRandomNumbers(std::vector<double>& x, double a = 0.0, 
                           double b = 1.0);

/// This function creates random rotation matrices (3 x 3) and stores them in 
/// the vector. That means, x[0 - 8] is the first rotation matrix. x[9 - 17] is 
/// the second and so forth. Size of the vector determines the number of 
/// rotation that fit into the vector and should be a multiple of 9.
void FillWithRandomRotations(std::vector<double>& x);

/// Construct the second order tensor (matrix coefficient) Theta from the 
/// equation R^T(e1,e2,e3) diag(l1,l2,l3) R (e1,e2,e3).
DenseMatrix ConstructMatrixCoefficient(double l1, double l2, double l3, 
                                       double e1, double e2, double e3, 
                                       double nu, int dim);

/// Construct the normalization coefficient eta of the white noise right hands 
/// side.
double ConstructNormalizationCoefficient(double nu, double l1, double l2, 
                                         double l3, int dim);

// ===========================================================================
// Implementation details
// ===========================================================================

void FillWithRandomNumbers(std::vector<double>& x, double a, double b) {
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis(a, b);
   for(int i = 0; i < x.size(); i++){
      x[i] = dis(gen);
   }
}

void FillWithRandomRotations(std::vector<double>& x){
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis(0, 1);
   for (size_t i = 0; i < x.size(); i += 9) {
      // Get a random rotation matrix via unifrom euler angles.
      double e1 = 2 * M_PI * dis(gen);
      double e2 = 2 * M_PI * dis(gen);
      double e3 = 2 * M_PI * dis(gen);
      const double c1 = cos(e1);
      const double s1 = sin(e1);
      const double c2 = cos(e2);
      const double s2 = sin(e2);
      const double c3 = cos(e3);
      const double s3 = sin(e3);

      // Fill the rotation matrix R with the Euler angles. See for instance 
      // the definition in wikipedia.
      x[i + 0] = c1 * c3 - c2 * s1 * s3;
      x[i + 1] = -c1 * s3 - c2 * c3 * s1;
      x[i + 2] = s1 * s2;
      x[i + 3] = c3 * s1 + c1 * c2 * s3;
      x[i + 4] = c1 * c2 * c3 - s1 * s3;
      x[i + 5] = -c1 * s2;
      x[i + 6] = s2 * s3;
      x[i + 7] = c3 * s2;
      x[i + 8] = c2;
   }
}

DenseMatrix ConstructMatrixCoefficient(double l1, double l2, double l3, 
                                       double e1, double e2, double e3, 
                                       double nu, int dim){

   if (dim == 3){
      // Compute cosine and sine of the angles e1, e2, e3
      const double c1 = cos(e1);
      const double s1 = sin(e1);
      const double c2 = cos(e2);
      const double s2 = sin(e2);
      const double c3 = cos(e3);
      const double s3 = sin(e3);

      // Fill the rotation matrix R with the Euler angles.
      DenseMatrix R(3,3);
      R(0,0) = c1 * c3 - c2 * s1 * s3;
      R(0,1) = -c1 * s3 - c2 * c3 * s1;
      R(0,2) = s1 * s2;
      R(1,0) = c3 * s1 + c1 * c2 * s3;
      R(1,1) = c1 * c2 * c3 - s1 * s3;
      R(1,2) = -c1 * s2;
      R(2,0) = s2 * s3;
      R(2,1) = c3 * s2;
      R(2,2) = c2;

      // Multiply the rotation matrix R with the translation vector.
      Vector l(3);
      l(0) = l1;
      l(1) = l2;
      l(2) = l3;
      l *= (1 / ( 2.0 * nu));
      
      // Compute result = R^t diag(l) R
      DenseMatrix res(3,3);
      R.Transpose();
      MultADBt(R, l, R, res);
      return res;
   } else {
      DenseMatrix res(2,2);
      res(0,0) = l1;
      res(1,1) = l2;
      res(0,1) = 0;
      res(1,0) = 0;
      return res;
   }
}

double ConstructNormalizationCoefficient(double nu, double l1, double l2, 
                                         double l3, int dim) {
   const double det = sqrt(l1*l2*l3);
   const double gamma1 = tgamma(nu + static_cast<double>(dim) / 2.0);
   const double gamma2 = tgamma(nu);
   if (Mpi::Root()){
      mfem::out << "nu = " << nu << ", dim = " << dim << std::endl;
      mfem::out << "gamma1 = " << gamma1 << "\n";
      mfem::out << "gamma2 = " << gamma2 << "\n";
      mfem::out << "det = " << det << "\n";
   }
   return sqrt(pow(2 * M_PI, dim / 2.0) * det * gamma1 /
               (gamma2 * pow(nu, dim / 2.0)));
}

} // namespace mfem