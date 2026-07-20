#!/bin/bash
# Download Pan-UKB flat sumstats files for the multi-trait ppb evaluation.
set -e
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
  if [ ! -s "$f" ]; then
    echo "downloading $f ..."
    curl -sL --retry 3 -o "$f" "$BASE/$f"
  fi
  echo "OK $f $(stat -c%s "$f")"
done
echo ALL_DONE
