build:
	docker build \
		--progress=plain \
		--tag imssyang/ai:openwakeword \
		-f Dockerfile .

run:
	docker run -itd \
		--device /dev/snd \
		--group-add audio \
		--env PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
		--volume ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
		--volume /opt/ai:/opt/ai \
		--workdir /opt/ai \
		--entrypoint /bin/bash \
		--name ai.openwakeword \
		imssyang/ai:openwakeword

exec:
	docker exec -it ai.openwakeword /bin/bash
