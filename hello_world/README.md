hello_world
===========

This repository serves as a mininal skeleton
for writing a TRIQS application.

2) This is a trivial example of an application which contains

 * C++ layer
 * Python interface.
 * All the relevant CMakeLists.

In order to use this skeleton you need to first install TRIQS. The TRIQS website is under http://ipht.cea.fr/triqs. Start there
to learn about TRIQS and how to install it.

This skeleton can be installed like any other TRIQS application.
At the prompt, issue the following series of commands:

$ git clone https://github.com/TRIQS/hello_world.git src
$ mkdir build && cd build
$ cmake ../src
$ make
$ make test
$ make install


The repository contains an example IPython notebook for illustration.
In order to try it, navigate to the example subdirectory of this repository.
Assuming that the bin directory of your TRIQS installation is in the search
path, you can issue the following command to start it:

$ipytriqs_notebook hello_world.ipynb

