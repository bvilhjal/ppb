#!/usr/bin/env bash
# Download Pan-UKB flat sumstats files for the multi-trait ppb evaluation.
set -euo pipefail
mkdir -p "$(dirname "$0")/../data/panukb"
cd "$(dirname "$0")/../data/panukb"
BASE="https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files"
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
  if [ -s "$f" ] && gzip -t "$f" 2>/dev/null; then
    echo "OK $f $(stat -c%s "$f")"
    continue
  fi

  tmp="${f}.part.$$"
  trap 'rm -f "$tmp"' EXIT
  echo "downloading $f ..."
  curl --fail --show-error --location --retry 3 --output "$tmp" "$BASE/$f"
  gzip -t "$tmp"
  mv -f "$tmp" "$f"
  trap - EXIT
  echo "OK $f $(stat -c%s "$f")"
done
echo ALL_DONE
