#include <iostream>
#include <vector>
#include <cmath>

template <typename T> concept bool Matrix = requires(T m) {
  { m(0, 0) } ->double;
  { dim(m) } ->int;
 };

//- -------------------------------------------------

double trace(Matrix const& m) {
 auto r = m(0, 0);
 int d = dim(m);
 for (int i = 1; i < d; ++i) r += m(i, i);
 return r;
}

//- -------------------------------------------------

class square_matrix {
 int n;
 std::vector<double> _data;

 public:
 square_matrix(int n) : n(n), _data(n * n, 1) {}

 double & operator()(int i, int j) { return _data[i + n * j]; }

 double operator()(int i, int j) const { return _data[i + n * j]; }
 friend int dim(square_matrix const& m) { return m.n; }
};

//- -------------------------------------------------

struct hilbert_matrix {
 int n;
 double operator()(int i, int j) const { return 1.0 / (i + j + 1); }
 friend int dim(hilbert_matrix const& m) { return m.n; }
};

//- -------------------------------------------------

class rank1_matrix {
 std::vector<double> _x, _y; // a place to store the data

 public:
 rank1_matrix(std::vector<double> const& x, std::vector<double> const& y) : _x(x), _y(y) {}

 double operator()(int i, int j) const { return _x[i] * _y[j]; }
 friend int dim(rank1_matrix const& m) { return m._x.size(); }
};

//- ----------------------  lazy addition ----------------------------

template <Matrix A, Matrix B> struct lazy_add {
 A const& a;
 B const& b;
 
 double operator()(int i, int j) const { return a(i, j) + b(i, j); }
 friend int dim(lazy_add const& x) { return dim(x.a); }
};

template <Matrix A, Matrix B> lazy_add<A, B> operator+(A const& a, B const& b) {
 return {a, b};
}

//- ----------------------  non matrix class ----------------------------

struct my_vector {};


//- ----------------------  abs ----------------------------

template <typename F, Matrix A> struct lazy_call {
 F f;
 A const& a;
 
 double operator()(int i, int j) const { return f(a(i, j)); } 
 friend int dim(lazy_call const& x) { return dim(x.a); }
};


template <typename F, Matrix A> lazy_call<F, A> make_lazy_call(F f, A const& a) {
 return {f,a}; 
}

template <Matrix A> auto abs(A const& a) {
 return make_lazy_call( [](auto const &x) { using std::abs; return abs(x);}, a);
}

//- ----------------------  main ----------------------------

int main() {
 auto m1 = square_matrix{4};

 m1(0,0) *=2;

 auto m2 = hilbert_matrix{4};
 auto m3 = rank1_matrix{{0,1,2},{1,1,1}};
 std::cout << trace(m1) <<'\n' << trace(m1 + m2)  <<'\n' << trace(m1 + m2 + m3)  << std::endl;
 //(1 + 1.0/3 + 1.0/5 + 1.0/7)  == 1.67619...
 
 //trace(my_vector{}); // does not compile

 std::cout << "test abs "<< std::endl;
 m1(0,0) *=-1;
 std::cout << trace(m1) << std::endl;
 std::cout << trace(abs(m1)) << std::endl;
}

