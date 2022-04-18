import requests
import numpy as np
import soundfile as sf
import os



def enhance_and_get_enhanced(path, save_to_dir = False, save_dir = ''):
    # Enhance first
    url = "http://127.0.0.1:5000/enhancefile"
    files = {'file': open(path,'rb')}
    r = requests.post(url, files=files)

    # Get enhanced audio
    filename = r.json()["filename"]
    url = "http://127.0.0.1:5000/getenhanced?filename=" + filename
    r = requests.get(url)
    data = r.json()
    
    if save_to_dir:
        wav = data["enhanced"]
        destpath = os.path.join(save_dir, filename)
        destpath = os.path.abspath(destpath)
        destdir = os.path.dirname(destpath)

        if not os.path.exists(destdir):
            os.mkdir(save_dir)
        sf.write(os.path.join(save_dir, filename), wav, 16000)

    return data

def enhance_and_get_mix(path, save_to_dir = False, save_dir = ''):
    # Enhance first
    url = "http://127.0.0.1:5000/enhancefile"
    files = {'file': open(path,'rb')}
    r = requests.post(url, files=files)

    # Get enhanced audio
    filename = r.json()["filename"]
    p = "0.6" # by default

    url = "http://127.0.0.1:5000/getmix?filename=" + filename + "&p=" + p
    r = requests.get(url)
    data = r.json()

    # save to dir if required
    if save_to_dir:
        wav = data["mix"]
        destpath = os.path.join(save_dir, filename)
        destpath = os.path.abspath(destpath)
        destdir = os.path.dirname(destpath)

        if not os.path.exists(destdir):
            os.mkdir(save_dir)
        sf.write(os.path.join(save_dir, filename), wav, 16000)

    return data

enhance_and_get_enhanced(
    path = '/home/khanhld/Desktop/FullSubNet/static/noisy/0a3a0f5609ea015da15c26a8727a58efc97b8553e1361d6fb049fd63.wav',
    save_to_dir = True,
    save_dir = 'enhanced')

enhance_and_get_mix(
    path = '/home/khanhld/Desktop/FullSubNet/static/noisy/0a3a0f5609ea015da15c26a8727a58efc97b8553e1361d6fb049fd63.wav',
    save_to_dir = True,
    save_dir = 'mix')

