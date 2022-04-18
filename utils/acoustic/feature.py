import os

import librosa
import numpy as np
import torch


def stft(y, n_fft, hop_length, win_length, device):
    """
    Args:
        y: [B, F, T]
        n_fft:
        hop_length:
        win_length:
        device:
    Returns:
        [B, F, T], **complex-valued** STFT coefficients
    """
    assert y.dim() == 2
    return torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(device),
        return_complex=True
    )


def istft(complex_tensor, n_fft, hop_length, win_length, device, length=None, use_mag_phase=False):
    """
    Wrapper for the official torch.istft
    Args:
        complex_tensor: [B, F, T, 2] or (mag: [B, F, T], phase: [B, F, T])
        n_fft:
        hop_length:
        win_length:
        device:
        length:
        use_mag_phase:
    Returns:
        [B, T]
    """
    if use_mag_phase:
        # (mag, phase) or [mag, phase]
        assert isinstance(complex_tensor, tuple) or isinstance(complex_tensor, list)
        mag, phase = complex_tensor
        complex_tensor = torch.stack([
            mag * torch.cos(phase),
            mag * torch.sin(phase)
        ], dim=-1)

    return torch.istft(
        complex_tensor,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(device),
        length=length
    )


def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(y, clipping_threshold=0.999):
    return any(np.abs(y) > clipping_threshold)


def load_wav(file, sr=16000, is_music = False):
    if len(file) == 2:
        return file[-1]
    elif is_music == True:
        return librosa.load(os.path.abspath(os.path.expanduser(file)), mono=True, sr=sr)[0]
    else:
        return librosa.load(os.path.abspath(os.path.expanduser(file)), mono=False, sr=sr)[0]


def aligned_subsample(data_a, data_b, sub_sample_length):
    """
    Start from a random position and take a fixed-length segment from two speech samples
    Notes
        Only support one-dimensional speech signal (T,) and two-dimensional spectrogram signal (F, T)
    """
    assert data_a.shape == data_b.shape, "Inconsistent dataset size."

    dim = np.ndim(data_a)
    assert dim == 1 or dim == 2, "Only support 1D or 2D."

    if data_a.shape[-1] > sub_sample_length:
        length = data_a.shape[-1]
        start = np.random.randint(length - sub_sample_length + 1)
        end = start + sub_sample_length
        if dim == 1:
            return data_a[start:end], data_b[start:end]
        else:
            return data_a[:, start:end], data_b[:, start:end]
    elif data_a.shape[-1] == sub_sample_length:
        return data_a, data_b
    else:
        length = data_a.shape[-1]
        if dim == 1:
            return (
                np.append(data_a, np.zeros(sub_sample_length - length, dtype=np.float32)),
                np.append(data_b, np.zeros(sub_sample_length - length, dtype=np.float32))
            )
        else:
            return (
                np.append(data_a, np.zeros(shape=(data_a.shape[0], sub_sample_length - length), dtype=np.float32), axis=-1),
                np.append(data_b, np.zeros(shape=(data_a.shape[0], sub_sample_length - length), dtype=np.float32), axis=-1)
            )


def subsample(data, sub_sample_length):
    """
    从随机位置开始采样出指定长度的数据
    Notes
        仅支持 1D 数据
    """
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        start = np.random.randint(length - sub_sample_length)
        end = start + sub_sample_length
        data = data[start:end]
        assert len(data) == sub_sample_length
        return data
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
        return data
    else:
        return data

def subsample_with_length_return(data, sub_sample_length):
    """
    从随机位置开始采样出指定长度的数据
    Notes
        仅支持 1D 数据
    """
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        start = np.random.randint(length - sub_sample_length)
        end = start + sub_sample_length
        data = data[start:end]
        assert len(data) == sub_sample_length
        return data, 1.0
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
        return data, length/sub_sample_length
    else:
        return data, 1.0

def subsample_music(clean, mixture, sub_sample_length):
    """
    从随机位置开始采样出指定长度的数据
    Notes
        仅支持 1D 数据
    """
    assert np.ndim(clean) == 1, f"Only support 1D data. The dim is {np.ndim(clean)}"
    assert np.ndim(mixture) == 1, f"Only support 1D data. The dim is {np.ndim(mixture)}"
    assert len(clean) == len(mixture), f"clean not equal mixture in music"
    length = len(clean)

    if length > sub_sample_length:
        start = np.random.randint(length - sub_sample_length)
        end = start + sub_sample_length
        clean = clean[start:end]
        mixture = mixture[start:end]
        assert len(clean) == sub_sample_length
        assert len(mixture) == sub_sample_length
        return clean, mixture
    elif length < sub_sample_length:
        clean = np.append(clean, np.zeros(sub_sample_length - length, dtype=np.float32))
        mixture = np.append(mixture, np.zeros(sub_sample_length - length, dtype=np.float32))
        return clean, mixture
    else:
        return clean, mixture

def overlap_cat(chunk_list, dim=-1):
    """
    按照 50% 的 overlap 沿着最后一个维度对 chunk_list 进行拼接
    Args:
        dim: 需要拼接的维度
        chunk_list(list): [[B, T], [B, T], ...]
    Returns:
        overlap 拼接后
    """
    overlap_output = []
    for i, chunk in enumerate(chunk_list):
        first_half, last_half = torch.split(chunk, chunk.size(-1) // 2, dim=dim)
        if i == 0:
            overlap_output += [first_half, last_half]
        else:
            overlap_output[-1] = (overlap_output[-1] + first_half) / 2
            overlap_output.append(last_half)

    overlap_output = torch.cat(overlap_output, dim=dim)
    return overlap_output


def activity_detector(audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=1e-6):
    """
    Return the percentage of the time the audio signal is above an energy threshold
    Args:
        audio:
        fs:
        activity_threshold:
        target_level:
        eps:
    Returns:
    """
    audio, _, _ = tailor_dB_FS(audio, target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win ** 2) + eps)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the subband part in the FullSubNet model.
    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert batch_size > num_groups, f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., :(num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)


if __name__ == '__main__':
    ipt = torch.rand(70, 1, 257, 200)
    y = torch.rand(1, 16000 * 3)
    # print(drop_band(ipt, 8).shape)
    print(stft(y, 512, 256, 512, 'cpu').shape)