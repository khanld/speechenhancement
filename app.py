import os
import librosa
import torch
import argparse
import json

from utils.utils import transcribe_wav, load_md_model, audiowrite, load_se_model, allowed_file
from flask import Flask, request, render_template, redirect, send_file, make_response
from zipfile import ZipFile
from speechbrain.pretrained import EncoderASR


  
# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['NOISY_FOLDER'], filename)
            file.save(filepath)

            # enhance
            noisy, _ = librosa.load(filepath, sr = 16000)
            noisy = torch.tensor(noisy).unsqueeze(0).to(device)
            enhanced = se_model.enhance(noisy)

            # generate mix
            p = 0.6
            use_se = (torch.sigmoid(md_model(noisy)[0]) >= 0.5).item()
            mix = p * enhanced + (1-p) * noisy
            if use_se:
                mix = enhanced

            noisy = noisy.detach().squeeze(0)
            enhanced = enhanced.detach().squeeze(0)
            mix = mix.detach().squeeze(0)

            # ASR decode
            noisy_transcript = transcribe_wav(asr_model, noisy)
            enhanced_transcript = transcribe_wav(asr_model, enhanced)
            mix_transcript = transcribe_wav(asr_model, mix)
            
            # save to file
            enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], filename)
            audiowrite(enhanced_path, enhanced.cpu().numpy())
            mix_path = os.path.join(app.config['MIX_FOLDER'], filename)
            audiowrite(mix_path, mix.cpu().numpy())

            return render_template("index.html", 
                                    noisyname = 'noisy/'+filename, 
                                    enhancedname = 'enhanced/'+filename,
                                    mixname='mix/'+filename,
                                    noisy_transcript = noisy_transcript,
                                    enhanced_transcript = enhanced_transcript,
                                    mix_transcript = mix_transcript)
    return render_template("index.html")

@app.route('/enhancefile', methods=['POST'])
def enhance_file():
    results = {
        "status": "",
        "message": "",
        "filename": ""
    }

    # check if the post request has the file part
    if 'file' not in request.files:
        results["status"] = "failed"
        results["message"] = "key of input should be 'file'"
        return make_response(results, 400)
        
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        results["status"] = "failed"
        results["message"] = "File is empty"
        return make_response(results, 400)

    if file and allowed_file(file.filename):
        try:
            filename = file.filename
            noisy_path = os.path.join(app.config['NOISY_FOLDER'], filename)
            enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], filename)

            # save noisy to file
            file.save(noisy_path)

            # enhanced
            noisy, _ = librosa.load(noisy_path, sr = 16000)
            noisy = torch.tensor(noisy).unsqueeze(0).to(device)
            enhanced = se_model.enhance(noisy)

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            # save enhanced to file
            audiowrite(enhanced_path, enhanced)

            results["status"] = "success"
            results["message"] = ""
            results["filename"] = filename

            return make_response(results, 200)
        except Exception as e:
            results["status"] = "failed"
            results["message"] = str(e)
            return make_response(results, 500)
    else:
        results["status"] = "failed"
        results["message"] = "File not supported"
        return make_response(results, 400)

@app.route('/getenhanced', methods=['GET'])
def get_enhanced():
    results = {
        "status": "",
        "message": ""
    }
    filename = request.args.get("filename")

    # check if filename is empty
    if filename == '' or filename is None:
        results["status"] = "failed"
        results["message"] = "Please provide filename"
        return make_response(results, 400)
    
    # check if enhanced path exists
    enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], filename)
    if not os.path.exists(enhanced_path):
        results["status"] = "failed"
        results["message"] = "File not exists, check your filename"
        return make_response(results, 400)

    enhanced, _ = librosa.load(enhanced_path, sr = 16000)
    results["status"] = "success"
    results["enhanced"] = enhanced.tolist()
    return make_response(results, 200)

    
@app.route('/getmix', methods=['GET'])
def get_mix():
    results = {
        "status": "",
        "message": ""
    }
    p = request.args.get('p')
    filename = request.args.get("filename")

    # check if filename is empty
    if filename == '' or filename is None:
        results["status"] = "failed"
        results["message"] = "Please provide filename"
        return make_response(results, 400)

    # check value of p
    if p is None:
        p = 0.6
    else:
        try:
            p = float(p)
            assert (p >= 0 or p <= 1), "p out of range"
        except:
            results["status"] = "failed"
            results["message"] = "p should be a number in range[0,1]. Default p = 0.6 if not provided"
            return make_response(results, 400)

    # check if enhanced and noisy paths exist
    noisy_path = os.path.join(app.config['NOISY_FOLDER'], filename)
    enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], filename)
    if not os.path.exists(enhanced_path) or not os.path.exists(noisy_path):
        results["status"] = "failed"
        results["message"] = "File not exists, check your filename"
        return make_response(results, 400)

    try:
        enhanced, _ = librosa.load(enhanced_path, sr = 16000)
        noisy, _ = librosa.load(noisy_path, sr = 16000)
        # generate mixture
        mix = p * enhanced + (1-p) * noisy

        # check if audio contains multispeakers
        noisy = torch.tensor(noisy).unsqueeze(0).to(device)
        use_se = (torch.sigmoid(md_model(noisy)[0]) >= 0.5).item()
        if use_se:
            # reassign mixture to enhanced if the audio contains multispeakers
            mix = enhanced
        
        # save mix to file
        mix_path = os.path.join(app.config['MIX_FOLDER'], filename)
        audiowrite(mix_path, mix)


        results["status"] = "success"
        results["message"] = ""
        results["mix"] = mix.tolist()
        results["p"] = p
        results["is_multispeaker"] = use_se
        return make_response(results, 200)

    except Exception as e:
        results["status"] = "failed"
        results["message"] = str(e)
        results["mix"] = ""
        results["p"] = p
        results["is_multispeaker"] = ""
        return make_response(results, 500)
  
# main driver function
if __name__ == '__main__':

    device =  torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    
    args = argparse.ArgumentParser(description='FullSubNet Demo')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')           
    
    args = args.parse_args()
    f = open(args.config)
    config = json.loads(f.read())
    f.close()
    # print(config["se_model"]["args"])
    se_model = load_se_model(**config["se_model"]["args"], device = device)
    md_model = load_md_model(**config["md_model"]["args"], device = device)
    asr_model = EncoderASR.from_hparams(source="dragonSwing/wav2vec2-base-vn-270h", run_opts={"device":device})


    app.config['UPLOAD_FOLDER'] = config["STATIC_FOLDER"]
    app.config['NOISY_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'noisy')
    app.config['ENHANCED_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced')
    app.config['MIX_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'mix')


    if not os.path.exists(app.config['NOISY_FOLDER']):
        os.makedirs(app.config['NOISY_FOLDER'])
    if not os.path.exists(app.config['ENHANCED_FOLDER']):
        os.makedirs(app.config['ENHANCED_FOLDER'])
    if not os.path.exists(app.config['MIX_FOLDER']):
        os.makedirs(app.config['MIX_FOLDER'])

    app.run(host="0.0.0.0")
