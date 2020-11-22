FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root

RUN apt-get update 
RUN apt-get install -y python3.6-dev \
                       python3-pip \
                       wget
RUN wget https://github.com/hpcng/singularity/releases/download/v3.6.4/singularity-3.6.4.tar.gz
RUN tar xvf singularity-3.6.4.tar.gz
RUN cd singularity-3.6.4
RUN ./configure --prefix=/usr/local
RUN make
RUN sudo make install
