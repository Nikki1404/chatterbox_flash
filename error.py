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


export http_proxy=http://163.116.128.80:8080
export https_proxy=http://163.116.128.80:8080
export HTTP_PROXY=http://163.116.128.80:8080
export HTTPS_PROXY=http://163.116.128.80:8080

apt-get update

apt-get install -y \
  ca-certificates \
  curl \
  gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update

apt-get install -y nvidia-container-toolkit

nvidia-ctk --version
nvidia-ctk runtime configure --runtime=docker

systemctl show docker --property=Environment

docker info | grep -i proxy


(base) root@EC03-E01-AICO3:/home/CORP/re_nikitav/chatterbox_flash# docker run --rm   --gpus all   --ipc=host   --shm-size=8g   chatterbox-flash
python -c "
import torch
import flashinfer

print('GPU:', torch.cuda.get_device_name(0))
print('CUDA available:', torch.cuda.is_available())
print('Torch CUDA:', torch.version.cuda)
print('FlashInfer:', flashinfer.__version__)
"
docker: Error response from daemon: failed to discover GPU vendor from CDI: no known GPU vendor found

Run 'docker run --help' for more information

(base) root@EC03-E01-AICO3:/home/CORP/re_nikitav/chatterbox_flash# nvidia-ctk --version
nvidia-ctk: command not found
(base) root@EC03-E01-AICO3:/home/CORP/re_nikitav/chatterbox_flash# cd ..
(base) root@EC03-E01-AICO3:/home/CORP/re_nikitav# nvidia-ctk --version
nvidia-ctk: command not found
(base) root@EC03-E01-AICO3:/home/CORP/re_nikitav# docker info | grep -i -E "runtime|nvidia|cdi"
 CDI spec directories:
  /etc/cdi
  /var/run/cdi
 Runtimes: io.containerd.runc.v2 runc
 Default Runtime: runc
(base) root@EC03-E01-AICO3:/home/CORP/re_nikitav# ls -la /etc/cdi /var/run/cdi 2>/dev/null
(base) root@EC03-E01-AICO3:/home/CORP/re_nikitav#