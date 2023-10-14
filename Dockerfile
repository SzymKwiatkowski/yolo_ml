FROM ubuntu:22.04

RUN echo "Europe/Utc" > /etc/timezone
# RUN ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip
ENV SHELL /bin/bash

RUN apt-get update -q && \
	export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends tzdata
RUN dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update -q && \
    apt-get install -y --no-install-recommends apt-utils software-properties-common wget curl rsync netcat mg vim bzip2 zip unzip

RUN apt-get update -q && \
        export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y --no-install-recommends install libgl1-mesa-glx libgl1-mesa-dri && \
    apt-get -y install mesa-utils && \
    rm -rf /var/lib/apt/lists/*

# RUN sed -i 's/--no-generate//g' /usr/share/bash-completion/completions/apt-get && \
#     sed -i 's/--no-generate//g' /usr/share/bash-completion/completions/apt-cache

RUN sed -i "s/#force_color_prompt=yes/force_color_prompt=yes/g" /root/.bashrc

RUN sed -i "s/#force_color_prompt=yes/force_color_prompt=yes/g" /home/$USERNAME/.bashrc

RUN echo 'if [ -f /etc/bash_completion ] && ! shopt -oq posix; then \n\
    . /etc/bash_completion \n\
fi \n\
\n\
export USER=${USERNAME} \n' >> /home/$USERNAME/.bashrc

# Install requirements
COPY requirements.txt ./
RUN pip3 install -r requirements.txt 

# Install formatting provider
RUN pip3 install autopep8 

RUN mkdir /home/ml

WORKDIR /home/ml

RUN export COMETML_KEY=api_key

USER $USERNAME
CMD ["/bin/bash"]⏎     