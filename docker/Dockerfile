# This is the dockerfile to create the docker image to replicate the full stack study environment. 
# If looking to replicate the studies work please use the provided image instance in the README



FROM harrycoppock/ciab:ciab_v3

# metainformation
LABEL version="0.0.4"
LABEL maintainer="Harry Coppock"


RUN  pip install --upgrade torchaudio
ENV PATH /opt/conda/bin:$PATH
