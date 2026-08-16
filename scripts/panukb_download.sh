#!/usr/bin/env bash
# Download Pan-UKB flat sumstats files for the multi-trait ppb evaluation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKSUMS="$SCRIPT_DIR/panukb_checksums.tsv"
mkdir -p "$SCRIPT_DIR/../data/panukb"
cd "$SCRIPT_DIR/../data/panukb"
BASE="https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files"

md5_of() {
  # Portable MD5 digest: GNU/BSD `md5sum`, macOS `md5 -q`, or stdlib python3.
  # All three print the bare hex digest; callers compare it as a string, so a
  # missing tool yields an empty digest and the validation below fails closed.
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    python3 - "$1" <<'PY'
import hashlib, sys
digest = hashlib.md5()
with open(sys.argv[1], "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
  fi
}

validate() {
  local path=$1 expected_size=$2 expected_md5=$3 actual_size
  [ -f "$path" ] || return 1
  actual_size=$(wc -c < "$path")
  [ "$actual_size" = "$expected_size" ] || return 1
  [ "$(md5_of "$path")" = "$expected_md5" ] || return 1
  gzip -t "$path"
}

for f in \
  continuous-50-both_sexes-irnt.tsv.bgz \
  continuous-21001-both_sexes-irnt.tsv.bgz \
  biomarkers-30780-both_sexes-irnt.tsv.bgz \
  continuous-4080-both_sexes-irnt.tsv.bgz \
  icd10-E11-both_sexes.tsv.bgz \
  icd10-J45-both_sexes.tsv.bgz \
  icd10-F32-both_sexes.tsv.bgz \
  icd10-I25-both_sexes.tsv.bgz \
  icd10-C50-both_sexes.tsv.bgz
do
  row=$(awk -F '\t' -v file="$f" '$1 == file {print $2 " " $3}' "$CHECKSUMS")
  if [ -z "$row" ]; then
    echo "error: no published checksum for $f" >&2
    exit 1
  fi
  read -r expected_size expected_md5 <<< "$row"

  if validate "$f" "$expected_size" "$expected_md5" 2>/dev/null; then
    echo "OK $f $expected_size $expected_md5"
    continue
  fi

  tmp="${f}.part.$$"
  trap 'rm -f "$tmp"' EXIT
  echo "downloading $f ..."
  curl --fail --show-error --location --retry 3 --output "$tmp" "$BASE/$f"
  if ! validate "$tmp" "$expected_size" "$expected_md5"; then
    echo "error: $f does not match the published Pan-UKB size/checksum" >&2
    exit 1
  fi
  mv -f "$tmp" "$f"
  trap - EXIT
  echo "OK $f $expected_size $expected_md5"
done
echo ALL_DONE
