# syntax=docker/dockerfile:1

FROM python:3.10.12-slim-bookworm

ARG dirname="/code"

RUN apt-get update -y &&\
    apt-get upgrade -y && \
    apt-get install -y vim tmux libgomp1

RUN mkdir $dirname
WORKDIR $dirname

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . ./
RUN pip3 install --upgrade pip && \
    pip3 install -r ./requirements.txt
