o the correct fix is not “remove the proxy everywhere.” Configure Docker so that Docker Hub bypasses the proxy, while your build steps continue using it.

Keep these lines in your Dockerfile:

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ENV HTTP_PROXY="http://163.116.128.80:8080"
ENV HTTPS_PROXY="http://163.116.128.80:8080"

The problem is only the Docker daemon trying to use that proxy for:

registry-1.docker.io
Configure Docker daemon with NO_PROXY

First inspect:

cat /etc/systemd/system/docker.service.d/http-proxy.conf

You probably have something similar to:

[Service]
Environment="HTTP_PROXY=http://163.116.128.80:8080"
Environment="HTTPS_PROXY=http://163.116.128.80:8080"

Change it to:

[Service]
Environment="HTTP_PROXY=http://163.116.128.80:8080"
Environment="HTTPS_PROXY=http://163.116.128.80:8080"
Environment="NO_PROXY=localhost,127.0.0.1,::1,registry-1.docker.io,auth.docker.io,index.docker.io,docker.io,.docker.io"

You can create/replace it directly:

mkdir -p /etc/systemd/system/docker.service.d

cat > /etc/systemd/system/docker.service.d/http-proxy.conf <<'EOF'
[Service]
Environment="HTTP_PROXY=http://163.116.128.80:8080"
Environment="HTTPS_PROXY=http://163.116.128.80:8080"
Environment="NO_PROXY=localhost,127.0.0.1,::1,registry-1.docker.io,auth.docker.io,index.docker.io,docker.io,.docker.io"
EOF

Then:

systemctl daemon-reload
systemctl restart docker

Check:

systemctl show docker --property=Environment

You should see both the proxy and NO_PROXY.

Now test:

docker pull nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04


Run this exactly:

cat > /etc/systemd/system/docker.service.d/http-proxy.conf <<'EOF'
[Service]
Environment="HTTP_PROXY=http://163.116.128.80:8080"
Environment="HTTPS_PROXY=http://163.116.128.80:8080"
Environment="NO_PROXY=localhost,127.0.0.1,169.254.169.254,metadata.google.internal,registry-1.docker.io,auth.docker.io,index.docker.io,docker.io,.docker.io"
EOF

Then reload Docker:

systemctl daemon-reload
systemctl restart docker

Verify:

docker info | grep -i proxy

You should now see something like:

HTTP Proxy: http://163.116.128.80:8080
HTTPS Proxy: http://163.116.128.80:8080
No Proxy: localhost,127.0.0.1,169.254.169.254,metadata.google.internal,registry-1.docker.io,auth.docker.io,index.docker.io,docker.io,.docker.io

Then test the base image pull:

docker pull nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04



sudo mkdir -p /etc/systemd/system/docker.service.d

sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<'EOF'
[Service]
Environment="HTTP_PROXY=http://163.116.128.80:8080"
Environment="HTTPS_PROXY=http://163.116.128.80:8080"
Environment="NO_PROXY=localhost,127.0.0.1,::1,169.254.169.254,metadata.google.internal,registry-1.docker.io,auth.docker.io,index.docker.io,docker.io,.docker.io"
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker

systemctl show docker --property=Environment

docker info | grep -i proxy


 => [ 7/21] RUN uv pip install -e ".[flashinfer]"                                                                                                     73.1s
 => ERROR [ 8/21] RUN python - <<'PY'                                                                                                                 32.6s
------
 > [ 8/21] RUN python - <<'PY':
31.94 Traceback (most recent call last):
31.94   File "<stdin>", line 2, in <module>
31.94   File "/opt/venv/lib/python3.12/site-packages/flashinfer/__init__.py", line 249, in <module>
31.94     from . import mamba as mamba
31.94   File "/opt/venv/lib/python3.12/site-packages/flashinfer/mamba/__init__.py", line 23, in <module>
31.94     from .ssd_combined import SSDCombined
31.94   File "/opt/venv/lib/python3.12/site-packages/flashinfer/mamba/ssd_combined.py", line 35, in <module>
31.94     from ..triton.kernels.ssd_chunk_state import chunk_cumsum_fwd
31.94   File "/opt/venv/lib/python3.12/site-packages/flashinfer/triton/__init__.py", line 28, in <module>
31.94     from . import sm_constraint_gemm  # noqa: F401
31.94     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
31.94   File "/opt/venv/lib/python3.12/site-packages/flashinfer/triton/sm_constraint_gemm.py", line 6, in <module>
31.94     from .kernels.sm_constraint_gemm import (
31.94   File "/opt/venv/lib/python3.12/site-packages/flashinfer/triton/kernels/sm_constraint_gemm.py", line 48, in <module>
31.94     @triton.autotune(
31.94      ^^^^^^^^^^^^^^^^
31.94   File "/opt/venv/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 378, in decorator
31.94     return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
31.94            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
31.94   File "/opt/venv/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 130, in __init__
31.94     self.do_bench = driver.active.get_benchmarker()
31.94                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
31.94   File "/opt/venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 23, in __getattr__
31.94     self._initialize_obj()
31.94   File "/opt/venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 20, in _initialize_obj
31.94     self._obj = self._init_fn()
31.94                 ^^^^^^^^^^^^^^^
31.94   File "/opt/venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 8, in _create_driver
31.94     raise RuntimeError(f"{len(actives)} active drivers ({actives}). There should only be one.")
31.94 RuntimeError: 0 active drivers ([]). There should only be one.
------
Dockerfile:62
--------------------
  61 |
  62 | >>> RUN python - <<'PY'
  63 | >>> import torch
  64 | >>> import flashinfer
  65 | >>>
  66 | >>> print("=" * 70)
  67 | >>> print("PyTorch version      :", torch.__version__)
  68 | >>> print("PyTorch CUDA version :", torch.version.cuda)
  69 | >>> print("FlashInfer version   :", flashinfer.__version__)
  70 | >>> print("=" * 70)
  71 | >>> PY
  72 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c python - <<'PY'\nimport torch\nimport flashinfer\n\nprint(\"=\" * 70)\nprint(\"PyTorch version      :\", torch.__version__)\nprint(\"PyTorch CUDA version :\", torch.version.cuda)\nprint(\"FlashInfer version   :\", flashinfer.__version__)\nprint(\"=\" * 70)\nPY" did not complete successfully: exit code: 1