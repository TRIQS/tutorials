#include <triqs/arrays.hpp>
using namespace triqs::arrays;
int main() {

 auto a = matrix<double>(2, 2);                   // Declare a 2x2 matrix of double
 auto b = array<double, 3>(5, 2, 2);              // Declare a 5x2x2 array of double
 auto c = array<double, 2>{{1, 2, 3}, {4, 5, 6}}; // 2x3 array, with initialization

 triqs::clef::placeholder<0> i_;
 triqs::clef::placeholder<1> j_;
 triqs::clef::placeholder<2> k_;

 // Assign values
 a(i_, j_) << i_ + j_;
 b(i_, j_, k_) << i_ * a(k_, j_);

 std::cout << "a = " << a << std::endl; // Printing

 matrix<double> i = inverse(a); // Inverse using LAPACK
 double d = determinant(a);     // Determinant using LAPACK

 auto ac = a;                 // Make a copy (the container is a regular type)
 ac = a * a + 2 * ac;         // Basic operations (uses BLAS for matrix product)
 b(0, range(), range()) = ac; // Assign ac into partial view of b

 // Writing the array into an hdf5 file.
 auto f = h5::file("a_file.h5", 'w');
 h5_write(f, "a", a);

 auto m = max_element(abs(b)); // maximum of the absolute value of the array.

 // A more "functional" example: compute the norm sum_{i,j} |A_{ij}|
 auto lambda = [](double r, double x) { return r + std::abs(x); };
 auto norm = fold(lambda)(a, 0);
}
