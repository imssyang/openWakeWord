FROM ghcr.io/imssyang/ai:cu126.trt107.torch260.ubuntu22

LABEL org.opencontainers.image.description="imssyang/ai:openwakeword"

ARG OPENWAKEWORD_HOME="/opt/ai/openWakeWord"

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torchcodec:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -y \
    libasound2-dev \
    libspeexdsp-dev \
    libpulse-dev \
    python3-pyaudio && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/PortAudio/portaudio.git /opt/portaudio && \
    cd /opt/portaudio && \
    ./configure --with-pulseaudio && \
    make && make install && ldconfig

ADD requirements.txt ${OPENWAKEWORD_HOME}/requirements.txt

RUN pip install --no-cache-dir -r ${OPENWAKEWORD_HOME}/requirements.txt

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torchcodec:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

WORKDIR ${OPENWAKEWORD_HOME}

ENTRYPOINT ["/bin/bash"]
