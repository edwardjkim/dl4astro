FROM ipython/scipyserver

MAINTAINER Edward Kim <edward.junhyung.kim@gmail.com>

RUN apt-get update && \
    apt-get -y -q install libfftw3-dev libatlas-base-dev wget vim && \

    # SExtractor uses the LAPACK functions available in ATLAS, and
    # it won't be able to find LAPACK without the following line.
    update-alternatives --set liblapack.so /usr/lib/atlas-base/atlas/liblapack.so && \
    
    # install sextractor
    cd / && \
    wget http://www.astromatic.net/download/sextractor/sextractor-2.19.5.tar.gz && \
    tar xvzf sextractor-2.19.5.tar.gz && \
    cd sextractor-2.19.5 && \
    ./configure --with-atlas-incdir=/usr/include/atlas && \
    make && \
    make install && \

    # Python montage-wrapper
    pip install --upgrade pip && \
    pip install montage-wrapper && \

    # Install Montage
    cd / && \
    wget http://montage.ipac.caltech.edu/download/Montage_v4.0.tar.gz && \
    tar xvzf Montage_v4.0.tar.gz && \
    cd /montage && \
    make

ENV PATH $PATH:/montage/bin

RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git && \
    pip install --upgrade --no-deps git+git://github.com/Lasagne/Lasagne.git && \
    pip install --upgrade --no-deps git+git://github.com/dnouri/nolearn.git
