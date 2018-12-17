FROM gidden/python3-viz

COPY . /pyam
WORKDIR /
RUN cd /pyam && python3 setup.py install 
RUN pip install pytest-cov