ARG VARIANT=bookworm-slim
FROM debian:${VARIANT}
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "deb http://deb.debian.org/debian bookworm-backports main" > /etc/apt/sources.list.d/backports.list

# Arguments
ARG CONTAINER_USER=luisa
ARG CONTAINER_GROUP=luisa
ARG BLENDER_BRANCH=blender-v4.0-release
ARG BLENDER_REPO=https://projects.blender.org/blender/blender.git
ARG LUISACOMPUTE_REPO=https://github.com/LuisaGroup/luisa-compute-rs.git

# Install dependencies
RUN apt-get update \
    && apt-get install -t bookworm-backports -y git pkg-config curl unzip clang cmake ninja-build python3-dev python3-pip uuid-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set users
RUN adduser --disabled-password --gecos "" ${CONTAINER_USER}
USER ${CONTAINER_USER}
WORKDIR /home/${CONTAINER_USER}

# Install rustup
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=${PATH}:/home/${CONTAINER_USER}/.cargo/bin

# Install peer dependencies
RUN git clone --branch ${BLENDER_BRANCH} --depth 1 ${BLENDER_REPO} blender_src
RUN git clone --recurse-submodules --shallow-submodules --depth 1 ${LUISACOMPUTE_REPO} luisa-compute-rs

CMD [ "/bin/bash" ]
