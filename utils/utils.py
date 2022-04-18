import soundfile as sf
import os
import torch 

from inferencer.se_model import Model as SE_Model
from inferencer.md_model import Model as MD_Model



def transcribe_wav(asr_model, waveform):
        """Transcribes the given audiofile into a sequence of words.
        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.
        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = asr_model.transcribe_batch(
            batch, rel_length
        )
        return predicted_words[0]


def load_se_model(
            checkpoint_path, 
            sb_num_neighbors,
            fb_num_neighbors,
            num_freqs,
            look_ahead,
            sequence_model,
            fb_output_activate_function,
            sb_output_activate_function,
            fb_model_hidden_size,
            sb_model_hidden_size,
            weight_init,
            norm_type,
            num_groups_in_drop_band,
            device = "cpu"):
    # Speech Enhancement parameters, em giữ nguyên các tham số mặc định của FullSubnet
    model = SE_Model(
                num_freqs,
                look_ahead,
                sequence_model,
                fb_num_neighbors,
                sb_num_neighbors,
                fb_output_activate_function,
                sb_output_activate_function,
                fb_model_hidden_size,
                sb_model_hidden_size,
                norm_type,
                num_groups_in_drop_band,
                weight_init,
                device)
    model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_static_dict = model_checkpoint["model"]
    epoch = model_checkpoint["epoch"]
    print(f"Loading model checkpoint (epoch == {epoch})...")

    model_static_dict = {key.replace("module.", ""): value for key, value in model_static_dict.items()}

    model.load_state_dict(model_static_dict, strict = False)
    model.to(device)
    model.eval()
    return model



def load_md_model(
                checkpoint_path, 
                num_hidden_layers,
                num_attention_heads,
                intermediate_size,
                hidden_size,
                device = 'cpu'):
    model = MD_Model(num_hidden_layers, num_attention_heads, intermediate_size, hidden_size)
    model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_static_dict = model_checkpoint["model"]
    epoch = model_checkpoint["epoch"]
    print(f"Loading model checkpoint (epoch == {epoch})...")

    model_static_dict = {key.replace("module.", ""): value for key, value in model_static_dict.items()}

    model.load_state_dict(model_static_dict, strict = False)
    model.to(device)
    model.eval()
    return model

def audiowrite(destpath, audio, sample_rate=16000):
    '''Function to write audio'''
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', "wav"}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
