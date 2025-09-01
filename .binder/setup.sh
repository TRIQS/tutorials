#!/bin/bash

if [[ ! -d ~/home/tutorials ]]
then
  git clone https://github.com/triqs/tutorials --branch unstable --depth 1 ~/home/tutorials & sleep 5
fi
cd ~/home/tutorials
export SHELL=/bin/bash

exec "$@"
