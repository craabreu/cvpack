#!/usr/bin/env bash
set -e -v
ruff format cvpack
ruff check --fix cvpack
