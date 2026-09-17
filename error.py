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


(base) root@EC03-E01-AICO3:/home/CORP/re_nikitav/chatterbox_flash# docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p 8000:8000 \
  -v "$(pwd)/reference.wav:/app/reference.wav:ro" \
  chatterbox-flash

==========
== CUDA ==
==========

CUDA Version 13.0.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

INFO:     Started server process [1]
INFO:     Waiting for application startup.
2026-09-17 13:50:20,588 | INFO | GPU: NVIDIA A10G
2026-09-17 13:50:20,588 | INFO | Torch CUDA: 12.6
2026-09-17 13:50:20,753 | INFO | HTTP Request: GET https://huggingface.co/api/agent-harnesses "HTTP/1.1 200 OK"
2026-09-17 13:50:20,809 | INFO | HTTP Request: HEAD https://huggingface.co/ResembleAI/chatterbox-flash/resolve/main/ve.safetensors "HTTP/1.1 302 Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-09-17 13:50:20,809 | WARNING | Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-09-17 13:50:20,866 | INFO | HTTP Request: GET https://huggingface.co/api/models/ResembleAI/chatterbox-flash/xet-read-token/4385507288b8197e6dab8b4e6b1603328d549d9d "HTTP/1.1 200 OK"
ve.safetensors: downloading bytes: ████████████████████████████████████████████████████████████████████████████████████████████████████| 5.44MB,  530kB/s
ve.safetensors: reconstructing file: 100%|████████████████████████████████████████████████████████████████████████████████████| 5.70MB / 5.70MB,  558kB/s
2026-09-17 13:50:21,752 | INFO | HTTP Request: HEAD https://huggingface.co/ResembleAI/chatterbox-flash/resolve/main/t3_flash.safetensors "HTTP/1.1 302 Found"
t3_flash.safetensors: downloading bytes: ██████████████████████████████████████████████████████████████████████████████████████████████| 2.01GB, 45.2MB/s
t3_flash.safetensors: reconstructing file: 100%|██████████████████████████████████████████████████████████████████████████████| 2.13GB / 2.13GB, 98.8MB/s
2026-09-17 13:51:14,668 | INFO | HTTP Request: HEAD https://huggingface.co/ResembleAI/chatterbox-flash/resolve/main/s3gen.safetensors "HTTP/1.1 302 Found"
s3gen.safetensors: downloading bytes: █████████████████████████████████████████████████████████████████████████████████████████████████| 1.00GB, 39.8MB/s
s3gen.safetensors: reconstructing file: 100%|█████████████████████████████████████████████████████████████████████████████████| 1.06GB / 1.06GB, 78.2MB/s
2026-09-17 13:51:43,907 | INFO | HTTP Request: HEAD https://huggingface.co/ResembleAI/chatterbox-flash/resolve/main/tokenizer.json "HTTP/1.1 307 Temporary Redirect"
2026-09-17 13:51:43,920 | INFO | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/ResembleAI/chatterbox-flash/4385507288b8197e6dab8b4e6b1603328d549d9d/tokenizer.json?%2FResembleAI%2Fchatterbox-flash%2Fresolve%2Fmain%2Ftokenizer.json=&etag=%22abd07c710243ba89bf1b21780e7c37ddde92334e%22 "HTTP/1.1 200 OK"
2026-09-17 13:51:43,944 | INFO | HTTP Request: GET https://huggingface.co/api/resolve-cache/models/ResembleAI/chatterbox-flash/4385507288b8197e6dab8b4e6b1603328d549d9d/tokenizer.json?%2FResembleAI%2Fchatterbox-flash%2Fresolve%2Fmain%2Ftokenizer.json=&etag=%22abd07c710243ba89bf1b21780e7c37ddde92334e%22 "HTTP/1.1 200 OK"
tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 25.5k/25.5k [00:00<00:00, 13.6MB/s]
/opt/venv/lib/python3.12/site-packages/diffusers/models/lora.py:393: FutureWarning: `LoRACompatibleLinear` is deprecated and will be removed in version 1.0.0. Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.
  deprecate("LoRACompatibleLinear", "1.0.0", deprecation_message)
2026-09-17 13:51:56,471 | INFO | input frame rate=25
2026-09-17 13:51:57,290 | INFO | Model loaded in 96701.50 ms | sample_rate=24000
/opt/chatterbox-flash/chatterbox_flash/tts.py:185: UserWarning: PySoundFile failed. Trying audioread instead.
  s3gen_ref_wav, _sr = librosa.load(str(wav_fpath), sr=S3GEN_SR)
/opt/venv/lib/python3.12/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
        Deprecated as of librosa version 0.10.0.
        It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
ERROR:    Traceback (most recent call last):
  File "/opt/venv/lib/python3.12/site-packages/librosa/core/audio.py", line 176, in load
    y, sr_native = __soundfile_load(path, offset, duration, dtype)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/librosa/core/audio.py", line 209, in __soundfile_load
    context = sf.SoundFile(path)
              ^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/soundfile.py", line 708, in __init__
    self._file = self._open(file, mode_int, closefd)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/soundfile.py", line 1296, in _open
    raise LibsndfileError(err, prefix=f"Error opening {self.name!r}: ")
soundfile.LibsndfileError: Error opening '/app/reference.wav': Format not recognised.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/venv/lib/python3.12/site-packages/starlette/routing.py", line 694, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/opt/venv/lib/python3.12/site-packages/fastapi/routing.py", line 265, in __aenter__
    await self._router._startup()
  File "/opt/venv/lib/python3.12/site-packages/fastapi/routing.py", line 6375, in _startup
    await handler()
  File "/app/server.py", line 115, in startup
    tts.prepare_conditionals(REFERENCE_AUDIO)
  File "/opt/chatterbox-flash/chatterbox_flash/tts.py", line 185, in prepare_conditionals
    s3gen_ref_wav, _sr = librosa.load(str(wav_fpath), sr=S3GEN_SR)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/librosa/core/audio.py", line 184, in load
    y, sr_native = __audioread_load(path, offset, duration, dtype)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/decorator/__init__.py", line 247, in fun
    return caller(func, *(extras + args), **kw)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/librosa/util/decorators.py", line 63, in __wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/librosa/core/audio.py", line 240, in __audioread_load
    reader = audioread.audio_open(path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/audioread/__init__.py", line 126, in audio_open
    return BackendClass(path)
           ^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/audioread/rawread.py", line 59, in __init__
    self._fh = open(filename, 'rb')
               ^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/app/reference.wav'

ERROR:    Application startup failed. Exiting.