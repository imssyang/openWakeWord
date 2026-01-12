# https://stackoverflow.com/questions/68310978/playing-sound-in-docker-container-on-wsl-in-windows-11
# https://stackoverflow.com/questions/73092750/how-to-show-gui-apps-from-docker-desktop-container-on-windows-11
#
IMAGE := ghcr.io/imssyang/ai:openwakeword
HOSTDIR := /opt/ai/app/openWakeWord
WORKDIR := /opt/ai/openWakeWord

build:
	docker build \
		--progress=plain \
		--tag $(IMAGE) \
		-f Dockerfile .

run:
	docker run -itd --gpus all \
		--privileged \
		--env DISPLAY=${DISPLAY} \
		--env PULSE_SERVER=${PULSE_SERVER} \
		--env PYTHONPATH=$(WORKDIR) \
		--env WAYLAND_DISPLAY=${WAYLAND_DISPLAY} \
		--env XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
		--volume /mnt/wslg:/mnt/wslg \
		--volume /mnt/wslg/.X11-unix:/tmp/.X11-unix \
		--volume $(HOSTDIR):$(WORKDIR) \
		--workdir $(WORKDIR) \
		--entrypoint /bin/bash \
		--name ai.openwakeword \
		$(IMAGE)

run.cpu:
	docker run -itd \
		--privileged \
		--env DISPLAY=${DISPLAY} \
		--env PULSE_SERVER=${PULSE_SERVER} \
		--env PYTHONPATH=$(WORKDIR) \
		--env WAYLAND_DISPLAY=${WAYLAND_DISPLAY} \
		--env XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
		--volume /mnt/wslg:/mnt/wslg \
		--volume /mnt/wslg/.X11-unix:/tmp/.X11-unix \
		--volume $(HOSTDIR):$(WORKDIR) \
		--workdir $(WORKDIR) \
		--entrypoint /bin/bash \
		--name ai.openwakeword \
		$(IMAGE)

exec:
	docker exec -it ai.openwakeword /bin/bash

