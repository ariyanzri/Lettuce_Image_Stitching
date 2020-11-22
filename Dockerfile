FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root

RUN apt-get update 
RUN apt-get install -y python3.6-dev \
                       python3-pip \
                       wget \
                       gdal-bin \
                       libgdal-dev \
                       libspatialindex-dev \
                       build-essential \
                       software-properties-common \
                       apt-utils
#RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
#RUN apt-get update
#RUN apt-get install -y libgdal-dev
#RUN pip3 install cython
#RUN pip3 install --upgrade cython
#RUN pip3 install setuptools
#RUN pip3 install GDAL==3.0.4
#RUN pip3 install numpy==1.19.1
#RUN pip3 install matplotlib==3.2.1
#RUN pip3 install pandas==1.0.3
#RUN wget http://download.osgeo.org/libspatialindex/spatialindex-src-1.7.1.tar.gz
#RUN tar -xvf spatialindex-src-1.7.1.tar.gz
#RUN cd spatialindex-src-1.7.1/ && ./configure && make && make install
#RUN ldconfig                       
#RUN add-apt-repository ppa:ubuntugis/ppa
#RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
#RUN export C_INCLUDE_PATH=/usr/include/gdal

#RUN apt-get update
#RUN apt-get install -y libsm6 libxext6 libxrender-dev

#RUN pip3 install ExifRead==2.3.1
#RUN pip3 install gpsphoto==2.2.3
#RUN pip3 install imageio==2.8.0
#RUN pip3 install networkx==2.4
#RUN pip3 install opencv-python==3.4.2.16
#RUN pip3 install opencv-contrib-python==3.4.2.16
#RUN pip3 install piexif==1.1.3
#RUN pip3 install Pillow==7.1.2
#RUN pip3 install pytz==2020.1
#RUN pip3 install PyWavelets==1.1.1
#RUN pip3 install scikit-image==0.17.2
#RUN pip3 install scikit-learn==0.22.2.post1
#RUN pip3 install scipy==1.4.1
#RUN pip3 install six==1.14.0
#RUN pip3 install sklearn==0.0
#RUN pip3 install tifffile==2020.6.3

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    pkg-config

RUN export VERSION=1.11 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz

RUN echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
    echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
    source ~/.bashrc

RUN go get -u github.com/golang/dep/cmd/dep

RUN go get -d github.com/sylabs/singularity

RUN export VERSION=v3.6.4 && \
    cd $GOPATH/src/github.com/sylabs/singularity && \
    git fetch && \
    git checkout $VERSION

RUN export VERSION=3.6.4 && \
    mkdir -p $GOPATH/src/github.com/sylabs && \
    cd $GOPATH/src/github.com/sylabs && \
    wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz && \
    tar -xzf singularity-${VERSION}.tar.gz && \
    cd ./singularity && \
    ./mconfig

RUN ./mconfig && \
    make -C ./builddir && \
    make -C ./builddir install
