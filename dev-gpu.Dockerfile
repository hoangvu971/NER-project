# use nvidia cuda/cudnn image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    && apt update \
    && apt install -y \
        git \
        make \
	    sed \
        tmux \
        vim \
        wget \
        zsh

# allow history search in terminal
RUN echo "\"\e[A\": history-search-backward" > $HOME/.inputrc && echo "\"\e[B\": history-search-forward" $HOME/.inputrc

# install miniconda3
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh

RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda init zsh

# Add conda to PATH
ENV PATH="/root/miniconda3/bin:${PATH}"

# move into the root user's home directory
WORKDIR /root

# install core Python environment and system packages
COPY ./Makefile ./environment.yml ./
RUN make conda-update

# switch to a login shell after cleaning up config:
#   removing error-causing line in /root/.profile, see https://www.educative.io/answers/error-mesg-ttyname-failed-inappropriate-ioctl-for-device
#   removing environment-setting in /root/.bashrc
RUN sed -i "s/mesg n || true/tty -s \&\& mesg n/" $HOME/.profile
RUN sed -i "s/conda activate base//" $HOME/.bashrc
SHELL ["conda", "run", "--no-capture-output", "-n", "training-pipeline", "/bin/bash", "-c"]

# install the core requirements, then remove build files
COPY ./requirements ./requirements
RUN make pip-tools && rm -rf ./Makefile ./requirements ./environment.yml
RUN pip install datasets -U

# run all commands inside the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "training-pipeline", "/bin/bash"]