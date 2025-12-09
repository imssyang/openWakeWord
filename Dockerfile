FROM ghcr.io/imssyang/ai:cu126.trt107.torch260.ubuntu22

LABEL org.opencontainers.image.description="imssyang/ai:openwakeword"

ARG OPENWAKEWORD_HOME="/opt/ai/openWakeWord"

ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && \
    apt-get install -y \
    alsa-utils \
    libpulse0 \
    libspeexdsp-dev \
    pulseaudio-utils \
    python3-pyaudio && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt ${OPENWAKEWORD_HOME}/requirements.txt

RUN pip install --no-cache-dir -r ${OPENWAKEWORD_HOME}/requirements.txt

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR ${OPENWAKEWORD_HOME}

ENTRYPOINT ["/bin/bash"]
