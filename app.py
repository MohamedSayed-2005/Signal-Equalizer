"""
Signal Equalizer
Flask backend — dual-domain equalization: FFT + optimal wavelet per mode + AI.
Supports: CSV, DAT, WAV, MP3
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import json, os, io, wave, struct, subprocess, tempfile
from scipy import signal as scipy_signal
from scipy.io import wavfile as scipy_wavfile

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

# ── AI Model Imports (ECG) ──
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
    from tensorflow.keras.models import Model
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── AI Model Imports (Voices) ──
try:
    import torch
    import torchaudio
    import librosa
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from speechbrain.inference.separation import SepformerSeparation
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False

# ── AI Model Imports (Music / Demucs) ──
try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model as demucs_get_model
    HAS_DEMUCS = True
except ImportError:
    HAS_DEMUCS = False

app = Flask(__name__)
store = {
    'ecg': {},
    'music': {},
    'voices': {},
    'generic': {}
}

# ── Optimal wavelet per mode ───────────────────────────────────────────────────
OPTIMAL_WAVELET = {
    'ecg':    'sym5',   
    'music':  'db4',    
    'voices': 'haar',
    'generic': 'db4'
}

# ── AI Model Definition ───────────────────────────────────────────────────────
def build_fcn_dae():
    inputs = Input(shape=(None, 1))
    
    x = Conv1D(32, 9, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 9, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(16, 9, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 9, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 9, activation='linear', padding='same')(x)

    return Model(inputs, decoded)

voice_separator_model = None

def separate_and_label_voices(signal_arr, sr):
    global voice_separator_model
    if not HAS_SPEECHBRAIN or not HAS_TORCH:
        raise RuntimeError("SpeechBrain and torch/torchaudio/librosa are required. Install with: pip install torch torchaudio speechbrain librosa")

    if voice_separator_model is None:
        import shutil, pathlib

        # ── Windows fix: OSError 1314 (symlink privilege not held) ──────────
        # SpeechBrain's fetcher calls Path.symlink_to() to link cached model
        # files into the savedir.  On Windows this requires the rarely-granted
        # "Create symbolic links" privilege.  We monkey-patch pathlib.Path so
        # that symlink_to() silently falls back to a plain file copy instead.
        _orig_symlink_to = pathlib.Path.symlink_to

        def _safe_symlink_to(self, target, target_is_directory=False):
            try:
                _orig_symlink_to(self, target, target_is_directory)
            except OSError:
                # Symlink failed (Windows privilege issue) — copy the file.
                src = pathlib.Path(target)
                if src.is_dir():
                    if self.exists():
                        shutil.rmtree(self)
                    shutil.copytree(src, self)
                else:
                    self.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, self)

        pathlib.Path.symlink_to = _safe_symlink_to
        # ─────────────────────────────────────────────────────────────────────

        savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pretrained_models", "sepformer")
        os.makedirs(savedir, exist_ok=True)

        try:
            voice_separator_model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir=savedir,
                run_opts={"device": "cpu"},
            )
        finally:
            # Always restore the original symlink_to so nothing else is affected.
            pathlib.Path.symlink_to = _orig_symlink_to

    sig_tensor = torch.tensor(signal_arr).unsqueeze(0).float()

    # Sepformer requires 8000 Hz
    if sr != 8000:
        resampler = torchaudio.transforms.Resample(orig_freq=int(sr), new_freq=8000)
        sig_tensor = resampler(sig_tensor)

    est_sources = voice_separator_model.separate_batch(sig_tensor)
    spk1 = est_sources[0, :, 0].numpy()
    spk2 = est_sources[0, :, 1].numpy()

    # Resample back to original sr
    if sr != 8000:
        spk1 = librosa.resample(spk1, orig_sr=8000, target_sr=int(sr))
        spk2 = librosa.resample(spk2, orig_sr=8000, target_sr=int(sr))

    # Normalize each speaker to original signal scale
    max_in = np.max(np.abs(signal_arr)) + 1e-10
    for spk in [spk1, spk2]:
        mx = np.max(np.abs(spk))
        if mx > 0:
            spk[:] = spk * (max_in / mx)

    # Classify gender by median pitch (lower = male, higher = female)
    def get_pitch(y, target_sr):
        try:
            f0 = librosa.yin(y, fmin=60, fmax=300, sr=int(target_sr))
            f0 = f0[f0 > 0]
            return float(np.median(f0)) if len(f0) > 0 else 0.0
        except Exception:
            return 0.0

    pitch1 = get_pitch(spk1, sr)
    pitch2 = get_pitch(spk2, sr)

    if pitch1 > pitch2:
        return spk2, spk1  # male (lower), female (higher)
    else:
        return spk1, spk2  # male (lower), female (higher)


# ── Music stem separator (Demucs htdemucs) ────────────────────────────────────
# Demucs produces 4 raw stems; we remap them to music-EQ-style labels and
# add a 5th "other" stem that is the residual (input minus sum of the 4).
#
#  Demucs name  →  our key          display label
#  vocals       →  vocals           Vocals
#  drums        →  bass_kick        Bass / Kick Drum
#  bass         →  guitar           Guitar / Strings
#  other        →  piano            Piano / Keys
#  (computed)   →  other            Other  (residual)
#
DEMUCS_TO_KEY = {
    'vocals': 'vocals',
    'drums':  'bass_kick',
    'bass':   'guitar',
    'other':  'piano',
}
MUSIC_STEMS      = ['vocals', 'bass_kick', 'guitar', 'piano', 'other']
MUSIC_STEM_COLORS = ['#ff4081', '#ff6d00', '#ffea00', '#00e5ff', '#00e676']

demucs_model = None

def separate_music_stems(signal_arr, sr):
    """Separate a mono audio signal into 5 display stems.
    Demucs gives 4 raw stems which are remapped to music-EQ-style labels,
    and a 5th 'other' residual stem is computed as input minus the sum of the 4.
    Falls back to a scipy band-pass mock when Demucs is not installed."""
    n = len(signal_arr)

    if not HAS_DEMUCS or not HAS_TORCH:
        # ── Mock fallback: carve stems out with band-pass filters ──────────
        stems = {}
        # vocals  ~ 300 Hz – 3400 Hz
        b, a = scipy_signal.butter(4, [300, 3400], btype='bandpass', fs=sr)
        stems['vocals'] = scipy_signal.filtfilt(b, a, signal_arr)
        # bass_kick ~ 20 Hz – 200 Hz  (kick + bass body)
        b, a = scipy_signal.butter(4, [20, 200], btype='bandpass', fs=sr)
        stems['bass_kick'] = scipy_signal.filtfilt(b, a, signal_arr)
        # guitar ~ 200 Hz – 2000 Hz
        b, a = scipy_signal.butter(4, [200, 2000], btype='bandpass', fs=sr)
        stems['guitar'] = scipy_signal.filtfilt(b, a, signal_arr)
        # piano ~ 2000 Hz – 8000 Hz
        b, a = scipy_signal.butter(4, [2000, min(8000, sr / 2 - 1)], btype='bandpass', fs=sr)
        stems['piano'] = scipy_signal.filtfilt(b, a, signal_arr)
        # other = residual
        combined = stems['vocals'] + stems['bass_kick'] + stems['guitar'] + stems['piano']
        stems['other'] = signal_arr - combined
        return stems

    global demucs_model
    if demucs_model is None:
        demucs_model = demucs_get_model('htdemucs')
        demucs_model.eval()

    # Demucs wants float32 stereo tensor: (1, 2, T)
    mono   = torch.tensor(signal_arr, dtype=torch.float32)
    stereo = mono.unsqueeze(0).repeat(2, 1)   # (2, T)
    batch  = stereo.unsqueeze(0)              # (1, 2, T)

    # Demucs native sr is 44100; resample if needed
    demucs_sr = demucs_model.samplerate
    if int(sr) != demucs_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=int(sr), new_freq=demucs_sr)
        batch = resampler(batch)

    with torch.no_grad():
        sources = apply_model(demucs_model, batch, device='cpu', progress=False)
    # sources shape: (1, n_stems, 2, T)

    raw_stems = {}
    for i, demucs_name in enumerate(demucs_model.sources):
        if demucs_name not in DEMUCS_TO_KEY:
            continue
        our_key   = DEMUCS_TO_KEY[demucs_name]
        stem_mono = sources[0, i].mean(dim=0).numpy()   # (T,)
        if int(sr) != demucs_sr:
            stem_mono = librosa.resample(stem_mono, orig_sr=demucs_sr, target_sr=int(sr))
        stem_mono = stem_mono[:n] if len(stem_mono) >= n else np.pad(stem_mono, (0, n - len(stem_mono)))
        raw_stems[our_key] = stem_mono.astype(np.float64)

    # Ensure all 4 mapped stems exist
    for key in DEMUCS_TO_KEY.values():
        if key not in raw_stems:
            raw_stems[key] = np.zeros(n)

    # 5th stem: residual (input minus sum of 4 Demucs stems)
    combined = sum(raw_stems[k] for k in DEMUCS_TO_KEY.values())
    raw_stems['other'] = signal_arr - combined

    return raw_stems


def apply_ai_denoising(signal_arr, sr):
    USE_MOCK = not HAS_TF 
    
    if USE_MOCK:
        b, a = scipy_signal.butter(4, [0.5, 40], btype='bandpass', fs=sr)
        out_signal = scipy_signal.filtfilt(b, a, signal_arr)
    else:
        orig_len = len(signal_arr)
        pad_len = (4 - (orig_len % 4)) % 4
        padded = np.pad(signal_arr, (0, pad_len), 'constant')
        x = padded.reshape(1, -1, 1)
        
        model = build_fcn_dae()
        y = model.predict(x)
        out_signal = y[0, :, 0][:orig_len] 
    
    max_in = np.max(np.abs(signal_arr))
    max_out = np.max(np.abs(out_signal))
    if max_out > 0:
        out_signal = out_signal * (max_in / max_out)
        
    return out_signal


# ── Signal readers ─────────────────────────────────────────────────────────────
def read_csv_signal(text):
    lines = text.strip().split('\n')
    values, sr = [], 500.0
    for line in lines:
        parts = line.strip().split(',')
        try:
            vals = [float(p) for p in parts]
            values.append(vals[0] if len(vals) == 1 else vals[1])
        except ValueError:
            if 'sr' in line.lower() or 'sample' in line.lower():
                for p in parts:
                    try: sr = float(p)
                    except: pass
    return np.array(values, dtype=np.float64), sr

def read_dat_signal(raw_bytes):
    sig = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
    sig = sig / (np.max(np.abs(sig)) + 1e-10)
    return sig, 500.0

def read_wav_signal(raw_bytes):
    buf = io.BytesIO(raw_bytes)
    sr, data = scipy_wavfile.read(buf)
    dtype_map = {
        'int16':   lambda d: d.astype(np.float64) / 32768.0,
        'int32':   lambda d: d.astype(np.float64) / 2147483648.0,
        'uint8':   lambda d: (d.astype(np.float64) - 128.0) / 128.0,
        'float32': lambda d: d.astype(np.float64),
        'float64': lambda d: d.astype(np.float64),
    }
    conv = dtype_map.get(data.dtype.name, lambda d: d.astype(np.float64) / (np.max(np.abs(d)) + 1e-10))
    sig = conv(data)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    return sig, float(sr)

def find_ffmpeg():
    import shutil
    path = shutil.which('ffmpeg')
    if path: return path
    for p in [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        os.path.join(os.environ.get('LOCALAPPDATA',''), 'ffmpeg','bin','ffmpeg.exe'),
        os.path.join(os.environ.get('USERPROFILE',''),  'ffmpeg','bin','ffmpeg.exe'),
    ]:
        if p and os.path.isfile(p): return p
    return None

def read_mp3_signal(raw_bytes):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install it and add to PATH: https://ffmpeg.org/download.html")
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tmp.write(raw_bytes); tmp_in = tmp.name
    tmp_out = tmp_in.replace('.mp3', '_out.wav')
    try:
        r = subprocess.run([ffmpeg,'-y','-i',tmp_in,'-ac','1','-ar','44100','-f','wav',tmp_out],
                           capture_output=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg: {r.stderr.decode('utf-8','ignore')}")
        with open(tmp_out,'rb') as f: wav = f.read()
        return read_wav_signal(wav)
    finally:
        for p in [tmp_in, tmp_out]:
            try: os.unlink(p)
            except: pass

# ── FFT equalization ───────────────────────────────────────────────────────────
def apply_fft_equalization(signal, sr, sliders, gains):
    n = len(signal)
    fft_data = np.fft.rfft(signal)
    freqs    = np.fft.rfftfreq(n, d=1.0/sr)
    gain_arr = np.ones(len(freqs))
    for i, s in enumerate(sliders):
        g = gains[i] if i < len(gains) else 1.0
        if g == 1.0:
            continue  
        for lo, hi in s.get('ranges', []):
            mask = (freqs >= lo) & (freqs <= hi)
            gain_arr[mask] *= g
    fft_eq = fft_data * gain_arr
    output = np.fft.irfft(fft_eq, n=n)
    return output, freqs, np.abs(fft_data), np.abs(fft_eq)

# ── Wavelet equalization ───────────────────────────────────────────────────────
def get_level_freq_range(lv, level, sr):
    if lv == 0:
        return 0.0, sr / (2 ** level)
    actual = level - lv + 1
    return sr / (2 ** (actual + 1)), sr / (2 ** actual)

def apply_wavelet_equalization(signal, sr, sliders, gains, wavelet):
    if not HAS_PYWT:
        return signal, [], [], [], []

    level = min(pywt.dwt_max_level(len(signal), wavelet), 8)
    coeffs    = pywt.wavedec(signal, wavelet, level=level)
    eq_coeffs = [c.copy() for c in coeffs]
    input_mags = [float(np.sqrt(np.mean(c**2))) for c in coeffs]

    component_level_map = []
    for i, s in enumerate(sliders):
        g = gains[i] if i < len(gains) else 1.0
        target_levels = s.get('wavelet_levels', [])
        for lv in target_levels:
            if 0 <= lv < len(eq_coeffs):
                eq_coeffs[lv] = eq_coeffs[lv] * g
        component_level_map.append({
            'color':  s.get('color', '#00e5ff'),
            'name':   s.get('name', f'Component {i+1}'),
            'levels': [lv for lv in target_levels if 0 <= lv < len(coeffs)]
        })

    output_mags = [float(np.sqrt(np.mean(c**2))) for c in eq_coeffs]

    level_labels = []
    for lv in range(len(coeffs)):
        lo, hi = get_level_freq_range(lv, level, sr)
        if lv == 0:
            level_labels.append(f"cA {lo:.1f}–{hi:.1f}Hz")
        else:
            actual = level - lv + 1
            level_labels.append(f"cD{actual} {lo:.1f}–{hi:.1f}Hz")

    output = pywt.waverec(eq_coeffs, wavelet)[:len(signal)]
    return output, level_labels, input_mags, output_mags, component_level_map

def compute_spectrogram(signal, sr, nperseg=256):
    nperseg = min(nperseg, len(signal))
    f, t, Sxx = scipy_signal.spectrogram(signal, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    return f.tolist(), t.tolist(), (10*np.log10(Sxx+1e-10)).tolist()

def signal_to_wav_bytes(signal, sr):
    s = signal / (np.max(np.abs(signal)) + 1e-10)
    s = (s * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(int(sr)); wf.writeframes(s.tobytes())
    buf.seek(0)
    return buf

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/synthetic', methods=['POST'])
def generate_synthetic():
    """Generates a perfect synthetic signal testing 125, 250, 500, 1k, 2k, 4k, 8k Hz."""
    try:
        data = request.get_json()
        mode = data.get('mode', 'generic')
        sr = 44100.0
        duration = 3.0 
        
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freqs = [125, 250, 500, 1000, 2000, 4000, 8000]
        sig = np.zeros_like(t)
        
        for f in freqs:
            sig += np.sin(2 * np.pi * f * t)
            
        sig = sig / (np.max(np.abs(sig)) + 1e-10) 
        
        if mode not in store:
            store[mode] = {}
            
        store[mode].update(signal=sig, sr=sr, duration=duration,
                           n_samples=len(sig), time=t, 
                           fft_output=sig.copy(), wav_output=sig.copy(),
                           ai_output=sig.copy(), ai_noise=np.zeros_like(sig))
                           
        return jsonify(success=True, sr=sr, duration=duration,
                       n_samples=len(sig), signal=sig.tolist(), time=t.tolist())
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))

@app.route('/api/upload', methods=['POST'])
def upload_file():
    f = request.files.get('file')
    mode = request.form.get('mode', 'ecg')
    if not f: return jsonify(success=False, error='No file provided')
    fname = f.filename.lower()
    try:
        if   fname.endswith('.csv'): sig, sr = read_csv_signal(f.read().decode('utf-8','ignore'))
        elif fname.endswith('.dat'): sig, sr = read_dat_signal(f.read())
        elif fname.endswith('.wav'): sig, sr = read_wav_signal(f.read())
        elif fname.endswith('.mp3'): sig, sr = read_mp3_signal(f.read())
        else: return jsonify(success=False, error='Unsupported file type.')
        if len(sig) == 0: return jsonify(success=False, error='Empty signal')
        MAX = int(sr * 60)
        if len(sig) > MAX: sig = sig[:MAX]
        t = np.arange(len(sig)) / sr
        
        store[mode].update(signal=sig, sr=sr, duration=len(sig)/sr,
                           n_samples=len(sig), time=t, 
                           fft_output=sig.copy(), wav_output=sig.copy(),
                           ai_output=sig.copy(), ai_noise=np.zeros_like(sig))
                           
        return jsonify(success=True, sr=sr, duration=store[mode]['duration'],
                       n_samples=len(sig), signal=sig.tolist(), time=t.tolist())
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))

@app.route('/api/equalize', methods=['POST'])
def equalize():
    data    = request.get_json()
    mode    = data.get('mode', 'ecg')
    if 'signal' not in store[mode]:
        return jsonify(success=False, error='No signal loaded')
        
    gains   = data.get('gains', [])
    sliders = data.get('sliders', [])
    sig, sr = store[mode]['signal'], store[mode]['sr']
    try:
        output, freqs, in_mag, out_mag = apply_fft_equalization(sig, sr, sliders, gains)
        store[mode]['fft_output'] = output
        
        step = max(1, len(freqs)//2000)
        return jsonify(
            success=True,
            output=output.tolist(),
            frequencies=freqs[::step].tolist(),
            input_magnitude=in_mag[::step].tolist(),
            output_magnitude=out_mag[::step].tolist()
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))

@app.route('/api/wavelet_equalize', methods=['POST'])
def wavelet_equalize():
    data    = request.get_json()
    mode    = data.get('mode', 'ecg')
    
    # Generic Mode does not use wavelets, return early
    if mode == 'generic':
        return jsonify(success=True, output=store[mode].get('signal', []).tolist() if 'signal' in store[mode] else [])

    if 'signal' not in store[mode]:
        return jsonify(success=False, error='No signal loaded')
        
    gains   = data.get('gains', [])
    sliders = data.get('sliders', [])
    
    base_sig = store[mode]['signal']
    sr       = store[mode]['sr']
    wavelet  = OPTIMAL_WAVELET.get(mode, 'db4')
    try:
        output, level_labels, in_mags, out_mags, comp_map = \
            apply_wavelet_equalization(base_sig, sr, sliders, gains, wavelet)
        store[mode]['wav_output'] = output
        
        n = len(base_sig)
        freqs = np.fft.rfftfreq(n, d=1.0/sr)
        fft_in = np.abs(np.fft.rfft(base_sig))
        fft_out = np.abs(np.fft.rfft(output))
        step = max(1, len(freqs)//2000)

        return jsonify(
            success=True,
            output=output.tolist(),
            wavelet=wavelet,
            level_labels=level_labels,
            input_magnitude=in_mags,
            output_magnitude=out_mags,
            component_map=comp_map,
            frequencies=freqs[::step].tolist(),
            fft_in_mag=fft_in[::step].tolist(),
            fft_out_mag=fft_out[::step].tolist()
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))

@app.route('/api/ai_process', methods=['POST'])
def ai_process():
    data = request.get_json()
    mode = data.get('mode', 'ecg')
    if 'signal' not in store.get(mode, {}): 
        return jsonify(success=False, error='No signal loaded')

    sig = store[mode]['signal']
    sr = store[mode]['sr']

    try:
        if mode == 'voices':
            male_sig, female_sig = separate_and_label_voices(sig, sr)
            store[mode]['ai_male'] = male_sig
            store[mode]['ai_female'] = female_sig

            # Truncate/pad separated signals to match original length
            n = len(sig)
            male_sig = male_sig[:n] if len(male_sig) >= n else np.pad(male_sig, (0, n - len(male_sig)))
            female_sig = female_sig[:n] if len(female_sig) >= n else np.pad(female_sig, (0, n - len(female_sig)))

            step = max(1, n // 5000)
            return jsonify(
                success=True,
                is_voices=True,
                male=male_sig[::step].tolist(),
                female=female_sig[::step].tolist()
            )
        elif mode == 'music':
            stems = separate_music_stems(sig, sr)
            n = len(sig)
            step = max(1, n // 5000)

            # Store full-resolution stems for audio playback
            for stem_name, stem_arr in stems.items():
                store[mode][f'ai_{stem_name}'] = stem_arr

            return jsonify(
                success=True,
                is_music=True,
                stems={name: stems[name][::step].tolist() for name in MUSIC_STEMS},
                stem_colors={name: color for name, color in zip(MUSIC_STEMS, MUSIC_STEM_COLORS)},
            )
        else:
            output = apply_ai_denoising(sig, sr)
            store[mode]['ai_output'] = output
            removed_noise = sig - output
            store[mode]['ai_noise'] = removed_noise

            n = len(sig)
            fft_in = np.abs(np.fft.rfft(sig))
            fft_out = np.abs(np.fft.rfft(output))
            freqs = np.fft.rfftfreq(n, d=1.0/sr)
            step = max(1, len(freqs) // 2000)

            return jsonify(
                success=True,
                is_voices=False,
                is_music=False,
                output=output.tolist(),
                removed_noise=removed_noise.tolist(),
                frequencies=freqs[::step].tolist(),
                input_magnitude=fft_in[::step].tolist(),
                output_magnitude=fft_out[::step].tolist()
            )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))

@app.route('/api/spectrogram', methods=['GET'])
def get_spectrogram():
    which = request.args.get('which','input')
    mode  = request.args.get('mode', 'ecg')
    if 'signal' not in store[mode]: return jsonify(success=False, error='No signal loaded')
    
    if which == 'fft':
        sig = store[mode].get('fft_output', store[mode]['signal'])
    elif which == 'wav':
        sig = store[mode].get('wav_output', store[mode]['signal'])
    else:
        sig = store[mode]['signal']
        
    try:
        f, t, Sxx = compute_spectrogram(sig, store[mode]['sr'])
        return jsonify(success=True, frequencies=f, times=t, magnitudes=Sxx)
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/api/scalogram', methods=['GET'])
def get_scalogram():
    mode = request.args.get('mode', 'ecg')
    if 'signal' not in store[mode]: 
        return jsonify(success=False, error='No signal loaded')
    
    sig = store[mode].get('wav_output', store[mode]['signal'])
    sr = store[mode]['sr']
    duration = store[mode]['duration']
    
    try:
        max_len = 1000
        if len(sig) > max_len:
            sig = scipy_signal.resample(sig, max_len)
            time_arr = np.linspace(0, duration, max_len)
        else:
            time_arr = np.linspace(0, duration, len(sig))
            
        scales = np.arange(1, 64) 
        wavelet_type = 'cmor1.5-1.0' 
        
        coef, freqs = pywt.cwt(sig, scales, wavelet_type, sampling_period=1/sr)
        magnitudes = np.abs(coef)
        magnitudes_db = 10 * np.log10(magnitudes + 1e-10)
        
        return jsonify(
            success=True, 
            frequencies=freqs.tolist(), 
            times=time_arr.tolist(), 
            magnitudes=magnitudes_db.tolist()
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))

@app.route('/api/audio', methods=['GET'])
def get_audio():
    which = request.args.get('which','fft')
    mode  = request.args.get('mode', 'ecg')
    if 'signal' not in store[mode]: return jsonify(success=False, error='No signal loaded'), 400
    
    if which == 'fft':
        sig = store[mode].get('fft_output', store[mode]['signal'])
    elif which == 'wav':
        sig = store[mode].get('wav_output', store[mode]['signal'])
    elif which == 'ai':
        sig = store[mode].get('ai_output', store[mode]['signal'])
    elif which == 'male':
        sig = store[mode].get('ai_male', store[mode]['signal'])
    elif which == 'female':
        sig = store[mode].get('ai_female', store[mode]['signal'])
    elif which in MUSIC_STEMS:
        sig = store[mode].get(f'ai_{which}', store[mode]['signal'])
    else:
        sig = store[mode]['signal']
        
    return send_file(signal_to_wav_bytes(sig, store[mode]['sr']), mimetype='audio/wav',
                     as_attachment=False, download_name=f'{which}_signal.wav')

@app.route('/api/settings/save', methods=['POST'])
def save_settings():
    data = request.get_json()
    os.makedirs('settings', exist_ok=True)
    path = os.path.join('settings', f"{data.get('name','custom')}.json")
    with open(path,'w') as f: json.dump(data.get('settings',{}), f, indent=2)
    return jsonify(success=True, path=path)

@app.route('/api/settings/load', methods=['POST'])
def load_settings():
    f = request.files.get('file')
    if not f: return jsonify(success=False, error='No file')
    try: return jsonify(success=True, settings=json.load(f))
    except Exception as e: return jsonify(success=False, error=str(e))

@app.route('/api/settings/default', methods=['GET'])
def default_settings():
    path = os.path.join('settings','ecg_mode.json')
    if os.path.exists(path):
        with open(path,'r') as f: return jsonify(success=True, settings=json.load(f))
    return jsonify(success=False, error='Default settings not found')

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)