#!/usr/bin/env bash
# make sure 01-activate.sh is run beforehand
if [ -z ${MP_IV_SOURCED+x} ]; then echo "MP_IV_SOURCED environment variable is not set. Have you sourced 01-activate.sh?"; exit 1; fi

# run jupyter notebook and allow to access the notebook from a remote machine
jupyter notebook --ip=0.0.0.0 --allow-root
