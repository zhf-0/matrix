#!/bin/bash
# This is the script to check whether the compressed file has required
# structure

version=${1}
external_id=${2}

xzfile="$external_id-$version.tar.xz"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# move to the root of the external repository
cd $script_dir/..

# start the script
rm ${external_id}-*.tar.xz
echo "Remove previous tar.xz files"
tar -cJf $xzfile ocp/
echo "Create external compressed file from ocp folder"

# move back to the folder where you started
cd -
