FROM --platform=linux/amd64 ubuntu:24.04
ARG LLVM=19
ARG BUILD_ID=18

RUN apt-get update && \
      #apt-get install -y lsb-release wget software-properties-common && \
      #wget -O /tmp/llvm.sh https://apt.llvm.org/llvm.sh && chmod +x /tmp/llvm.sh && /tmp/llvm.sh ${LLVM} && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      make \
      meson \
      cmake \
      g++ \
      gfortran \
      git \
      htop \
      nano \
      vim \
      less \
      curl \
      hdf5-tools \
      openssh-client \
      clang-${LLVM} \
      llvm-${LLVM}-dev \
      libclang-${LLVM}-dev \
      libc++-${LLVM}-dev \
      libc++abi-${LLVM}-dev \
      libomp-${LLVM}-dev \
      lldb-${LLVM} \
      libboost-dev \
      libeigen3-dev \
      libfftw3-dev \
      libgfortran5 \
      libgmp-dev \
      libmpfr-dev \
      libhdf5-dev \
      libopenblas-dev \
      libopenmpi-dev \
      libnfft3-dev \
      libzstd-dev \
      openmpi-bin \
      openmpi-common \
      openmpi-doc \
      python3-clang-${LLVM} \
      python3-dev \
      python3-ipython \
      python3-mako \
      python3-matplotlib \
      python3-mpi4py \
      python3-numpy \
      python3-pandas \
      python3-pytest \
      python3-scipy \
      python3-setuptools \
      python3-skimage \
      python3-tk \
      python3-tomli \
      python3-venv \
      jupyter-notebook \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

ARG NB_USER=triqs
ARG NB_UID=1010
RUN useradd -u ${NB_UID} -m ${NB_USER} -o

USER ${NB_USER}
RUN python3 -m venv --system-site-packages /home/${NB_USER}/.venv
ENV VIRTUAL_ENV=/home/${NB_USER}/.venv \
    PATH=/home/${NB_USER}/.venv/bin:$PATH
RUN pip install jupyterlab==4.3.7 jupyterhub jupyter-archive ipykernel sparse-ir[xprec]
RUN python -m ipykernel install --user --name=python3

ARG BRANCH=unstable
ARG NCORES=10
ARG ARCH=native
ENV SRC=/tmp/src BUILD=/tmp/build INSTALL=/usr \
    CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu/openmpi:/usr/include/hdf5/serial:$CPLUS_INCLUDE_PATH \
    CC=clang-${LLVM} CXX=clang++-${LLVM} CXXFLAGS="-stdlib=libc++ -march=${ARCH}"

USER root
RUN set -ex ; \
  mkdir $BUILD ; cd $BUILD ; \
  git clone https://github.com/flatironinstitute/clair --branch doc --depth 1 $SRC ; \
  CXXFLAGS="" cmake $SRC -DCMAKE_INSTALL_PREFIX=$INSTALL ; \
  make -j$NCORES ; \
  make install ; \
  rm -rf $SRC $BUILD ;
RUN set -ex ; \
  mkdir $BUILD ; cd $BUILD ; \
  git clone https://github.com/TRIQS/triqs --branch modest_updates --depth 1 $SRC ; \
  cmake $SRC -DCMAKE_INSTALL_PREFIX=$INSTALL ; \
  make -j$NCORES ; \
  make install ; \
  rm -rf $SRC $BUILD ;
ARG BRANCH=unstable
RUN set -ex ; \
  for pkg in cthyb ctseg ctint tprf maxent hubbardI hartree_fock dft_tools solid_dmft ; do \
    mkdir $BUILD ; cd $BUILD ; \
    git clone https://github.com/TRIQS/$pkg --branch $BRANCH --depth 1 $SRC ; \
    cmake $SRC -DCMAKE_INSTALL_PREFIX=$INSTALL ; \
    make -j$NCORES ; \
    make install ; \
    rm -rf $SRC $BUILD ; \
  done
RUN set -ex ; \
  mkdir $BUILD ; cd $BUILD ; \
  git clone https://github.com/TRIQS/modest --branch DEV_DOCS --depth 1 $SRC ; \
  cmake $SRC -DCMAKE_INSTALL_PREFIX=$INSTALL -DUpdate_Python_Bindings=ON ; \
  make -j$NCORES ; \
  make install ; \
  rm -rf $SRC $BUILD ;

# SLATE
RUN set -ex ; \
  mkdir $BUILD ; cd $BUILD ; \
  git clone --branch master --depth 1 \
              --recurse-submodules --shallow-submodules \
              https://github.com/icl-utk-edu/slate.git $SRC ; \
  cmake $SRC -DCMAKE_INSTALL_PREFIX=$INSTALL -Dblas=openblas -Dgpu_backend=none -Dbuild_tests=no ; \
  make -j$NCORES ; \
  make install ; \
  rm -rf $SRC $BUILD ;

# CoQui
RUN set -ex ; \
  mkdir $BUILD ; cd $BUILD ; \
  git clone --branch triqs_school_2025 --depth 1 https://github.com/AbInitioQHub/coqui $SRC ; \
  cmake $SRC -DCMAKE_INSTALL_PREFIX=$INSTALL \
             -DCMAKE_CXX_FLAGS="${CXXFLAGS} -fPIC" \
             -DCMAKE_C_FLAGS="-fPIC" \
             -DCMAKE_Fortran_FLAGS="-fPIC" \
             -DBUILD_UNIT_TESTS=OFF \
             -DCOQUI_PYTHON_SUPPORT=ON ; \
  make -j$NCORES ; \
  make install ; \
  rm -rf $SRC $BUILD ;

RUN echo "localhost slots=2" >> /etc/openmpi/openmpi-default-hostfile
ENV PYTHONPATH=/usr/lib/python3.12/site-packages:$PYTHONPATH \
    OMP_NUM_THREADS=2

USER ${NB_USER}
COPY --chown=${NB_USER} setup.sh /
EXPOSE 8888
ENTRYPOINT ["/setup.sh"]
