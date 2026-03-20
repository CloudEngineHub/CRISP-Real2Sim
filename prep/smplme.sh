#!/bin/bash
set -euo pipefail

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -s "$path" ]]; then
        echo "$label not found: $path" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "$path" ]]; then
        echo "$label not found: $path" >&2
        exit 1
    fi
}

# SMPL Male and Female model
mkdir -p data/smpl
mkdir -p data/smplx
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)
mkdir -p HMR/inputs/checkpoints/body_models
mkdir -p HMR/inputs/checkpoints/body_models/smpl
mkdir -p HMR/inputs/checkpoints/body_models/smplx

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O './data/smpl/smpl.zip' --no-check-certificate --continue
require_file "./data/smpl/smpl.zip" "Downloaded SMPL archive"
unzip -o data/smpl/smpl.zip -d data/smpl
require_dir "data/smpl/SMPL_python_v.1.1.0/smpl/models" "Extracted SMPL model directory"
cp data/smpl/SMPL_python_v.1.1.0/smpl/models/basicModel_f_lbs_10_207_0_v1.1.0.pkl \
   data/smpl/SMPL_FEMALE.pkl

cp data/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl \
   data/smpl/SMPL_MALE.pkl

cp data/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
    data/smpl/SMPL_NEUTRAL.pkl

cp data/smpl/SMPL_python_v.1.1.0/smpl/models/basicModel_f_lbs_10_207_0_v1.1.0.pkl \
   HMR/inputs/checkpoints/body_models/smpl/SMPL_FEMALE.pkl

cp data/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl \
   HMR/inputs/checkpoints/body_models/smpl/SMPL_MALE.pkl

cp data/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
   HMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl

wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip" -O './data/smplx/smplx.zip' --no-check-certificate --continue
require_file "./data/smplx/smplx.zip" "Downloaded SMPL-X archive"
unzip -o data/smplx/smplx.zip -d data/smplx
require_dir "data/smplx/models/smplx" "Extracted SMPL-X model directory"

cp data/smplx/models/smplx/SMPLX_FEMALE.npz \
   HMR/inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz

cp data/smplx/models/smplx/SMPLX_MALE.npz \
   HMR/inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz

cp data/smplx/models/smplx/SMPLX_NEUTRAL.npz \
   HMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
