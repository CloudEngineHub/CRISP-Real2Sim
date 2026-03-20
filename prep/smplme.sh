#!/bin/bash
set -euo pipefail

DOWNLOAD_URL="https://download.is.tue.mpg.de/download.php"
DOWNLOAD_PAGE_URL="https://smpl.is.tue.mpg.de/download.php"
USER_AGENT="${SMPL_USER_AGENT:-Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36}"

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Required command not found: $cmd" >&2
        exit 1
    fi
}

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

is_zip_file() {
    local path="$1"
    python - "$path" <<'PY'
import sys
import zipfile

path = sys.argv[1]
raise SystemExit(0 if zipfile.is_zipfile(path) else 1)
PY
}

download_with_auth() {
    local domain="$1"
    local sfile="$2"
    local out_path="$3"
    local label="$4"
    local headers_file
    local http_code

    headers_file="$(mktemp)"
    http_code="$(
        curl -sS -L \
            --http1.1 \
            --connect-timeout 20 \
            --max-time 120 \
            -A "$USER_AGENT" \
            -e "$DOWNLOAD_PAGE_URL" \
            -H "Origin: https://smpl.is.tue.mpg.de" \
            -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" \
            -D "$headers_file" \
            -o "$out_path" \
            -w '%{http_code}' \
            --data-urlencode "username=$USERNAME" \
            --data-urlencode "password=$PASSWORD" \
            "$DOWNLOAD_URL?domain=$domain&sfile=$sfile"
    )"

    require_file "$out_path" "$label download"

    if ! is_zip_file "$out_path"; then
        if [[ "$http_code" == "401" ]] || grep -qi "Username/Password wrong" "$out_path"; then
            echo "$label download returned 401 Unauthorized." >&2
            echo "The username/password was not accepted by the SMPL download server." >&2
        elif [[ "$http_code" == "403" ]] || grep -qi "forbidden" "$out_path"; then
            echo "$label download returned 403 Forbidden." >&2
            echo "This usually means the account has not accepted the required SMPL/SMPL-X license for $sfile yet." >&2
            echo "Log in on the official site, accept the relevant license, confirm browser download works, then rerun this script." >&2
        elif grep -qi "sign in" "$out_path"; then
            echo "$label download returned the SMPL sign-in page instead of a zip file." >&2
            echo "The server did not accept this account for $sfile." >&2
        else
            echo "$label download did not return a zip archive." >&2
            echo "HTTP status: $http_code" >&2
            echo "Saved response: $out_path" >&2
        fi
        rm -f "$headers_file"
        exit 1
    fi

    unzip -tqq "$out_path" >/dev/null
    rm -f "$headers_file"
}

require_cmd curl
require_cmd unzip
require_cmd python

mkdir -p data/smpl
mkdir -p data/smplx
mkdir -p HMR/inputs/checkpoints/body_models
mkdir -p HMR/inputs/checkpoints/body_models/smpl
mkdir -p HMR/inputs/checkpoints/body_models/smplx

echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"

USERNAME="${SMPL_USERNAME:-}"
PASSWORD="${SMPL_PASSWORD:-}"

if [[ -z "$USERNAME" ]]; then
    read -r -p "Username (SMPL):" USERNAME
fi

if [[ -z "$PASSWORD" ]]; then
    read -r -s -p "Password (SMPL):" PASSWORD
    echo
fi

download_with_auth "smpl" "SMPL_python_v.1.1.0.zip" "./data/smpl/smpl.zip" "SMPL"
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

download_with_auth "smplx" "models_smplx_v1_1.zip" "./data/smplx/smplx.zip" "SMPL-X"
unzip -o data/smplx/smplx.zip -d data/smplx
require_dir "data/smplx/models/smplx" "Extracted SMPL-X model directory"

cp data/smplx/models/smplx/SMPLX_FEMALE.npz \
   HMR/inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz

cp data/smplx/models/smplx/SMPLX_MALE.npz \
   HMR/inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz

cp data/smplx/models/smplx/SMPLX_NEUTRAL.npz \
   HMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz

echo "SMPL and SMPL-X models downloaded successfully."
