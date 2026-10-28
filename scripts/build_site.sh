#!/usr/bin/env bash
# Assemble the static visualizer site into _site/.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf _site
mkdir -p _site/bytepair _site/models
cp web/index.html _site/
cp src/bytepair/__init__.py src/bytepair/base.py src/bytepair/basic.py src/bytepair/regex.py _site/bytepair/
cp models/pride.model _site/models/
echo "built _site/"
