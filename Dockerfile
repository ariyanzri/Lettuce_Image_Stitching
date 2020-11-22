FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root



RUN wget https://github.com/singularityware/singularity/releases/download/3.6.4/singularity-3.6.4.tar.gz
RUN tar xvf singularity-3.6.4.tar.gz
RUN cd singularity-3.6.4
RUN ./configure --prefix=/usr/local
RUN make
RUN sudo make install
