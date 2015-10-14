FROM ipython/scipyserver

MAINTAINER Edward Kim <edward.junhyung.kim@gmail.com>

RUN apt-get update && \
    apt-get -y -q install wget sextractor vim && \
    pip install --upgrade pip && \
    pip install pyfits montage-wrapper

RUN cd / && \
    wget http://montage.ipac.caltech.edu/download/Montage_v4.0.tar.gz && \
    tar xvzf Montage_v4.0.tar.gz && \
    cd /montage && \
    make

ENV PATH $PATH:/montage/bin
