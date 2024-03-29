FROM continuumio/miniconda3
# install libs needed by open3d
RUN apt-get update && apt-get install -y libusb-1.0-0 libgl1-mesa-glx tree && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /mp-iv/
WORKDIR /mp-iv
COPY environment.yml /mp-iv/
RUN bash -c "conda env create -f environment.yml && conda clean --all --quiet --yes"
COPY 01-activate.sh /mp-iv/
# make sure bash is the default docker interpreter in order to source our environment
SHELL ["bash", "--login", "-c"]
CMD source 01-activate.sh && /mp-iv/02-start-notebook.sh
