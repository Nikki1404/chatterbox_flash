 => => extracting sha256:914a4d8ff4576548e51306b439952aad1d7caecb6741ed11ee5bf6c7ebe93b70                                                              2.0s
 => ERROR [ 2/18] RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-dev python3-venv python3-pip     git curl ca-  38.1s
------
 > [ 2/18] RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-dev python3-venv python3-pip     git curl ca-certificates ffmpeg build-essential ninja-build     && rm -rf /var/lib/apt/lists/*:
31.03 Ign:1 http://security.ubuntu.com/ubuntu noble-security InRelease
31.05 Ign:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64  InRelease
31.05 Ign:3 http://archive.ubuntu.com/ubuntu noble InRelease
31.05 Ign:4 http://archive.ubuntu.com/ubuntu noble-updates InRelease
31.05 Ign:5 http://archive.ubuntu.com/ubuntu noble-backports InRelease
32.03 Ign:1 http://security.ubuntu.com/ubuntu noble-security InRelease
32.05 Ign:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64  InRelease
32.05 Ign:3 http://archive.ubuntu.com/ubuntu noble InRelease
32.05 Ign:4 http://archive.ubuntu.com/ubuntu noble-updates InRelease
32.05 Ign:5 http://archive.ubuntu.com/ubuntu noble-backports InRelease
34.03 Ign:1 http://security.ubuntu.com/ubuntu noble-security InRelease
34.05 Ign:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64  InRelease
34.05 Ign:3 http://archive.ubuntu.com/ubuntu noble InRelease
34.05 Ign:4 http://archive.ubuntu.com/ubuntu noble-updates InRelease
34.05 Ign:5 http://archive.ubuntu.com/ubuntu noble-backports InRelease
38.03 Err:1 http://security.ubuntu.com/ubuntu noble-security InRelease
38.03   Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
38.05 Err:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64  InRelease
38.05   Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
38.05 Err:3 http://archive.ubuntu.com/ubuntu noble InRelease
38.05   Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
38.05 Err:4 http://archive.ubuntu.com/ubuntu noble-updates InRelease
38.05   Unable to connect to 163.116.128.80:8080:
38.05 Err:5 http://archive.ubuntu.com/ubuntu noble-backports InRelease
38.05   Unable to connect to 163.116.128.80:8080:
38.05 Reading package lists...
38.07 W: Failed to fetch https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/InRelease  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
38.07 W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/noble/InRelease  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
38.07 W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/noble-updates/InRelease  Unable to connect to 163.116.128.80:8080:
38.07 W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/noble-backports/InRelease  Unable to connect to 163.116.128.80:8080:
38.07 W: Failed to fetch http://security.ubuntu.com/ubuntu/dists/noble-security/InRelease  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
38.07 W: Some index files failed to download. They have been ignored, or old ones used instead.
38.08 Reading package lists...
38.09 Building dependency tree...
38.09 Reading state information...
38.10 E: Unable to locate package python3
38.10 E: Unable to locate package python3-dev
38.10 E: Unable to locate package python3-venv
38.10 E: Unable to locate package python3-pip
38.10 E: Unable to locate package git
38.10 E: Unable to locate package curl
38.10 E: Unable to locate package ffmpeg
38.10 E: Unable to locate package build-essential
38.10 E: Unable to locate package ninja-build
------
Dockerfile:24
--------------------
  23 |
  24 | >>> RUN apt-get update && apt-get install -y --no-install-recommends \
  25 | >>>     python3 python3-dev python3-venv python3-pip \
  26 | >>>     git curl ca-certificates ffmpeg build-essential ninja-build \
  27 | >>>     && rm -rf /var/lib/apt/lists/*
  28 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c apt-get update && apt-get install -y --no-install-recommends     python3 python3-dev python3-venv python3-pip     git curl ca-certificates ffmpeg build-essential ninja-build     && rm -rf /var/lib/apt/lists/*" did not complete successfully: exit code: 100