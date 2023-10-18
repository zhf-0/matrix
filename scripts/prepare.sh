#!/bin/bash
# You should modify this script to prepare the folder structure for external project source codes
# The source code must be in ocp/external/external_id/version/source folder
# get command line arguments
version=${1}
external_id=${2}

# prepare necessary variables
url="https://github.com/zhf-0/matrix/archive/refs/heads/main.zip"
dir="ocp/external/$external_id/$version/source"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
gz_file="$external_id-v$version.tar.gz"
# move to the root of the external repository
cd $script_dir/..

# start the script
rm -rf *.tar.gz
wget -O $gz_file $url
echo "Download $external_id $version"
rm -rf ocp
echo "Remove previous ocp folder"
mkdir -p "$dir"
echo "Create ocp folder structure"
tar -xzf $gz_file -C $dir --strip-components=1
echo "Extract files to ocp folder"
cp -r configurations "$dir/.."
echo "Copy all configurations to ocp folder"
cp -r cmake "$dir/.."
echo "Copy all cmake files to ocp folder"
cp ocp.yml "$dir/.."
echo "Copy ocp.yml to ocp folder"

# move back to the folder where you started
cd -
