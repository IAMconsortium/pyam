FROM gidden/python2-viz

RUN add-apt-repository -y ppa:glasen/freetype2
RUN apt-get update
RUN apt-get install -y freetype2-demos

COPY . /pyam
WORKDIR /
RUN cd /pyam && python2 setup.py install 
