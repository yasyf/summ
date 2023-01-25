#!/bin/bash

git checkout main
hatch version patch
VERSION=$(hatch version)
git add summ/__about__.py
git commit -m "Bump to $VERSION"
git tag "$VERSION"
git push --tags
