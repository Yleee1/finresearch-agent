#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="${project_dir}/data/raw"
mkdir -p "${raw_dir}"

download_and_verify() {
  local url="$1"
  local output="$2"
  local expected_sha256="$3"

  if [[ ! -f "${output}" ]]; then
    curl --fail --location --retry 3 --output "${output}" "${url}"
  fi

  local actual_sha256
  actual_sha256="$(shasum -a 256 "${output}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Checksum mismatch: ${output}" >&2
    echo "Expected: ${expected_sha256}" >&2
    echo "Actual:   ${actual_sha256}" >&2
    exit 1
  fi
  echo "Verified: ${output}"
}

download_and_verify \
  "https://static.cninfo.com.cn/finalpage/2025-03-25/1222881496.PDF" \
  "${raw_dir}/byd_2024_annual_report.pdf" \
  "e9c2d7fdd088e151ccb6c8ad3d95587b2b014b10f2c9731508d23ce07fde4de3"

download_and_verify \
  "https://static.cninfo.com.cn/finalpage/2025-03-15/1222806982.PDF" \
  "${raw_dir}/catl_2024_annual_report.pdf" \
  "b4f1713d7b821eb076c102711d177fe942ccc2bc8dd171ae5d7a95799a65b0ad"

echo "Reports are available under data/raw/ and remain excluded from Git."
