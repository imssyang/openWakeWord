FROM ghcr.io/imssyang/ai:cu126.trt107.torch260.ubuntu22

LABEL org.opencontainers.image.description="imssyang/ai:openwakeword_ubuntu22"

ARG OPENWAKEWORD_HOME="/opt/ai/openWakeWord"

RUN apt-get update && \
    apt-get install -y \
    libspeexdsp-dev \
    python3-pyaudio \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt ${OPENWAKEWORD_HOME}/requirements.txt

RUN pip install -r ${OPENWAKEWORD_HOME}/requirements.txt

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENTRYPOINT ["/bin/bash"]
