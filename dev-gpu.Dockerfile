# use nvidia cuda/cudnn image
FROM continuumio/miniconda3:latest

RUN apt update \
    && apt install -y \
        nano \
        git \
        make \
	    sed \
        wget \
        zsh

# move into the root user's home directory
WORKDIR /root

# install core Python environment and system packages
COPY ./Makefile ./environment.yml ./
RUN make conda-update

SHELL ["conda", "run", "--no-capture-output", "-n", "training-pipeline", "/bin/bash", "-c"]

# install the core requirements, then remove build files
COPY ./requirements ./requirements
RUN make pip-tools-train && rm -rf ./Makefile ./requirements ./environment.yml


# run all commands inside the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "training-pipeline", "/bin/zsh"]