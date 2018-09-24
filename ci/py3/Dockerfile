FROM gidden/python3-viz

RUN add-apt-repository ppa:glasen/freetype2
RUN apt update
RUN apt install -y freetype2-demos

COPY . /pyam
WORKDIR /
RUN cd /pyam && python3 setup.py install 
