FROM gidden/python3-viz

RUN apt-get install --yes pandoc

COPY . /pyam
WORKDIR /
RUN cd /pyam && python3 setup.py install 
