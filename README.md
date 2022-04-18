# SPEECH ENHANCEMENT FOR ASR USAGE
## Description

This is a demo for FullSubNet Speech Enhancement for Vietnamese ASR. The Speech Enhancement model was trained on 1k3 hours of speech data with dynamic mixing (mix-on-the-fly). </br>
Here, we found an approach to remedy the problem of SE when adapting as front-end to ASR, which causes the degradation of ASR decoding performance on clean speech. The idea is simple but can work with any SE and ASR models. 

![alt text](images/flow.jpg)

## Usage

Docker build
```
docker build -t demo .
```
Docker run
```
Docker run demo
```
## Web App

![alt text](images/web_interface.jpg)

## API Calls

Check [example.py](examples.py) file for API usage
