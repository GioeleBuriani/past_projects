# this file is meant to be sourced by bash to set environment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced. Please run"
    echo "source $0"
    exit 1
fi

# set variable to announce that we're properly sourced (to be checked in successive scripts)
export MP_IV_SOURCED=

# absolute path of this script file
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"

export MPIV_ROOT_DIR=$THIS_DIR

# use 'source' for TA's (they have the 'source' dir) or 'release' (for students)
if [ -d "${THIS_DIR}/source" ]; then
    export SOURCE_DIR="${THIS_DIR}/source"
elif [ -d "${THIS_DIR}/release" ]; then
    export SOURCE_DIR="${THIS_DIR}/release"
fi

export PRACTICUM1_DATA_DIR="${THIS_DIR}/data"
export PRACTICUM3MP_DATA_DIR="${THIS_DIR}/data"

# check for external conda being available
if ! which conda; then
    # source miniconda if existing
    [ -f $THIS_DIR/miniconda3/bin/activate ] && source $THIS_DIR/miniconda3/bin/activate
fi

echo
echo "Adding $SOURCE_DIR, each practicum dir and common dir to PYTHONPATH"
export PYTHONPATH="$SOURCE_DIR:$PYTHONPATH"
export PYTHONPATH="$SOURCE_DIR/practicum1:$PYTHONPATH"
export PYTHONPATH="$SOURCE_DIR/practicum2:$PYTHONPATH"
export PYTHONPATH="$SOURCE_DIR/practicum3_mp:$PYTHONPATH"
export PYTHONPATH="$SOURCE_DIR/practicum3_iv:$PYTHONPATH"
export PYTHONPATH="$SOURCE_DIR/assignment:$PYTHONPATH"
export PYTHONPATH="$SOURCE_DIR/common:$PYTHONPATH"

# activate conda environment mp-iv
conda activate mp-iv
