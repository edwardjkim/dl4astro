FROM ipython/scipyserver

MAINTAINER Edward Kim <edward.junhyung.kim@gmail.com>

USER root

RUN apt-get update && \
    apt-get -y -q install libfftw3-dev libatlas-base-dev libatlas-base-dev gfortran wget vim && \

    # SExtractor uses the LAPACK functions available in ATLAS, and
    # it won't be able to find LAPACK without the following line.
    update-alternatives --set liblapack.so /usr/lib/atlas-base/atlas/liblapack.so && \

    cd /tmp && \
    # install sextractor
    wget http://www.astromatic.net/download/sextractor/sextractor-2.19.5.tar.gz && \
    tar xvzf sextractor-2.19.5.tar.gz && \
    cd sextractor-2.19.5 && \
    ./configure --with-atlas-incdir=/usr/include/atlas && \
    make && \
    make install && \

    # Python montage-wrapper
    pip2 install --upgrade pip && \
    pip2 install montage-wrapper && \

    # Install Montage
    cd /tmp && \
    wget http://montage.ipac.caltech.edu/download/Montage_v4.0.tar.gz && \
    tar xvzf Montage_v4.0.tar.gz && \
    cd montage && \
    make

ENV PATH $PATH:/tmp/montage/bin

RUN pip2 install --upgrade terminado && \
    pip2 install Theano Lasagne && \
    pip2 install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git
