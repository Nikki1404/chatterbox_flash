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


 => [ 6/18] WORKDIR /opt/chatterbox-flash                                                                                                              0.0s
 => [ 7/18] RUN uv pip install -e ".[flashinfer]"                                                                                                     75.0s
 => ERROR [ 8/18] RUN uv pip install     flashinfer-cubin     flashinfer-jit-cache     --index-url https://flashinfer.ai/whl/cu130                    19.3s
------
 > [ 8/18] RUN uv pip install     flashinfer-cubin     flashinfer-jit-cache     --index-url https://flashinfer.ai/whl/cu130:
1.240 Using Python 3.12.3 environment at: /opt/venv
19.32 error: No solution found when resolving dependencies
19.32   cause: Because flashinfer-cubin was not found in the package registry and you require flashinfer-cubin, we can conclude that your requirements are unsatisfiable.
------
Dockerfile:45
--------------------
  44 |     # CUDA 13.0 FlashInfer precompiled/JIT packages.
  45 | >>> RUN uv pip install \
  46 | >>>     flashinfer-cubin \
  47 | >>>     flashinfer-jit-cache \
  48 | >>>     --index-url https://flashinfer.ai/whl/cu130
  49 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c uv pip install     flashinfer-cubin     flashinfer-jit-cache     --index-url https://flashinfer.ai/whl/cu130" did not complete successfully: exit code: 1