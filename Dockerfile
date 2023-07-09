# syntax=docker/dockerfile:1

FROM python:3.10.12-slim-bookworm

ARG dirname="/code"

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y vim tmux libgomp1

RUN mkdir $dirname
WORKDIR $dirname

COPY . ./
RUN python3 -m venv ./venv
RUN . ./venv/bin/activate && pip3 install --upgrade pip && pip3 install -r ./requirements.txt
