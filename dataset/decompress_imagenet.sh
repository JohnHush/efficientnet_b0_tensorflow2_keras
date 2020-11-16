#!/bin/bash
set -e

SYNSETS_FILE='imagenet_lsvrc_2015_synsets.txt'
TRAIN_TARBALL='/Users/jh/working_data/imagenet/test/ILSVRC2012_img_test.tar'
OUTPUT_PATH='/Users/jh/working_data/imagenet/test'

while read SYNSET; do
  echo "Processing: ${SYNSET}"

  # Create a directory and delete anything there.
  mkdir -p "${OUTPUT_PATH}/${SYNSET}"
  rm -rf "${OUTPUT_PATH}/${SYNSET}/*"

  # Uncompress into the directory.
  tar xf "${TRAIN_TARBALL}" "${SYNSET}.tar"
  tar xf "${SYNSET}.tar" -C "${OUTPUT_PATH}/${SYNSET}/"
  rm -f "${SYNSET}.tar"

  echo "Finished processing: ${SYNSET}"
done < "${SYNSETS_FILE}"
