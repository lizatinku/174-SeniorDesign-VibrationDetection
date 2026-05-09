"""
Vibration Detection Dataset — Playback Assembly Script
=======================================================
Generates a single long WAV file for physical playback through a speaker.
Your partner plays this file while the vibration sensor records on the Pi.
Afterward, use the generated timestamps.csv to chop the sensor recording
into labeled segments.

Structure of output WAV:
  [1s beep][3s silence][45s clip][1s beep][3s silence][45s clip] ...

Mount your Google Drive in Colab first:
  from google.colab import drive
  drive.mount('/content/drive')

Then run this script.
"""

# ─── INSTALL DEPENDENCIES (run this cell first in Colab) ───────────────────
# !pip install pydub numpy scipy

import os
import re
import csv
import random
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
from scipy.signal import butter, sosfilt, resample_poly
from math import gcd

# ─── CONFIGURATION ─────────────────────────────────────────────────────────

# Update these paths to match your Google Drive structure
DRIVE_BASE     = "/content/drive/MyDrive/Senior_Design"
RAVDESS_DIR    = f"{DRIVE_BASE}/RAVDESS"
DEMAND_DIR     = f"{DRIVE_BASE}/DEMAND"
OUTPUT_DIR     = f"{DRIVE_BASE}/Dataset"

# Clip parameters
CLIP_DURATION_MS  = 45_000   # 45 seconds in milliseconds
GAP_DURATION_MS   = 3_000    # 3 second silence between clips
BEEP_DURATION_MS  = 1_000    # 1 second beep before each clip
BEEP_FREQ_HZ      = 300     # 300Hz tone — clearly visible in PRAAT
SAMPLE_RATE       = 8_000   # Hz — Downsampling down to 8000
TARGET_DBFS       = -20.0    # Target loudness for normalization

# Dataset parameters
CLIPS_PER_CLASS        = 60
CLIPS_PER_ENV_CLASS    = 10  # 60 clips / 6 environments = 10 per env per class

#Feasibility mode - generates only 10 clips to test Pi pipeline
FEASIBILITY_MODE = False

# SNR range for mixing speech over background (Classes 1 and 2)
SNR_MIN_DB = 5.0
SNR_MAX_DB = 15.0

# RAVDESS emotion codes
# Position 3 in filename (0-indexed: XX-XX-EM-XX-XX-XX-XX.wav)
CALM_EMOTION_CODES    = {"01", "02"}  # neutral, calm
DISTRESS_EMOTION_CODES = {"05", "06"} # angry, fearful

# DEMAND environments and which channel to use
DEMAND_ENVS   = ["DKITCHEN", "DLIVING", "PRESTO", "SPSQUARE", "OHALLWAY", "NPARK"]
DEMAND_CHANNEL = "ch01.wav"  # Use channel 1 from each environment

# Class definitions
CLASSES = {
    0: "background",
    1: "calm_speech",
    2: "distress",
}

random.seed(42)
np.random.seed(42)

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls in CLASSES.values():
        os.makedirs(f"{OUTPUT_DIR}/{cls}", exist_ok=True)

def make_beep():
    """Generate a 1kHz sine beep as a marker tone."""
    beep = Sine(BEEP_FREQ_HZ).to_audio_segment(duration=BEEP_DURATION_MS)
    beep = beep.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    # Fade in/out to avoid clicks
    beep = beep.fade_in(50).fade_out(50)
    return beep

def make_silence(duration_ms):
    return AudioSegment.silent(duration=duration_ms, frame_rate=SAMPLE_RATE)

def normalize(segment, target_dbfs=TARGET_DBFS):
    """Normalize audio to a target dBFS level."""
    if segment.dBFS == float('-inf'):
        return segment  # silent segment — don't amplify noise floor
    delta = target_dbfs - segment.dBFS
    return segment.apply_gain(delta)

def loop_to_duration(segment, target_ms):
    """Loop or trim an audio segment to exactly target_ms milliseconds."""
    if len(segment) >= target_ms:
        return segment[:target_ms]
    # Loop until long enough then trim
    loops_needed = (target_ms // len(segment)) + 2
    looped = segment * loops_needed
    return looped[:target_ms]

def mix_at_snr(speech, noise, snr_db):
    """
    Mix speech into noise at a given SNR (dB).
    SNR = 20*log10(rms_speech / rms_noise)
    Returns the mixed AudioSegment.
    """
    # Convert to numpy for RMS calculation
    speech_samples = np.array(speech.get_array_of_samples()).astype(np.float32)
    noise_samples  = np.array(noise.get_array_of_samples()).astype(np.float32)

    rms_speech = np.sqrt(np.mean(speech_samples ** 2))
    rms_noise  = np.sqrt(np.mean(noise_samples ** 2))

    if rms_speech < 1e-10 or rms_noise < 1e-10:
        return noise

    # Scale speech to achieve desired SNR
    target_rms_speech = rms_noise * (10 ** (snr_db / 20.0))
    scale_factor = target_rms_speech / rms_speech
    gain_db = 20 * np.log10(scale_factor)

    scaled_speech = speech.apply_gain(gain_db)
    mixed = noise.overlay(scaled_speech)
    return mixed

def scan_ravdess(ravdess_dir, emotion_codes):
    """
    Scan RAVDESS directory and return list of WAV files
    matching the given emotion codes.
    RAVDESS filename format: 03-01-EM-IN-ST-RE-AC.wav
    Position index 2 (0-based) is the emotion code.
    """
    matches = []
    for actor_folder in sorted(os.listdir(ravdess_dir)):
        actor_path = os.path.join(ravdess_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) < 7:
                continue
            modality = parts[0]   # 03 = audio only
            emotion  = parts[2]   # emotion code
            if modality == "03" and emotion in emotion_codes:
                matches.append(os.path.join(actor_path, fname))
    return matches

def load_demand(env_name):
    """Load a DEMAND environment channel as AudioSegment."""
    path = os.path.join(DEMAND_DIR, env_name, DEMAND_CHANNEL)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DEMAND file not found: {path}")
    seg = AudioSegment.from_wav(path)
    seg = seg.set_channels(1).set_sample_width(2)
    seg = resample_audio(seg, SAMPLE_RATE)
    return seg

def lowpass_filter(samples, src_rate, cutoff_hz):
    """Apply an 8th-order Butterworth low-pass filter to float32 samples."""
    nyq = src_rate / 2.0
    sos = butter(8, cutoff_hz / nyq, btype='low', output='sos')
    return sosfilt(sos, samples)

def resample_audio(segment, target_rate):
    """Resample the pydub AudioSegment to target_rate using scipy.
    Applies a Butterworth LPF before downsampling to suppress aliasing."""
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    src_rate = segment.frame_rate

    # Only filter when downsampling; cutoff at 90% of target Nyquist
    if target_rate < src_rate:
        cutoff = 0.9 * (target_rate / 2.0)
        samples = lowpass_filter(samples, src_rate, cutoff)

    # Compute up/down factors as a reduced fraction
    g = gcd(target_rate, src_rate)
    up = target_rate // g
    down = src_rate // g

    resampled = resample_poly(samples, up, down)
    peak = np.abs(resampled).max()
    if peak > 32767:
        resampled = resampled * (32767 / peak)  # scale down to prevent clipping
    resampled = resampled.astype(np.int16)

    return AudioSegment(
        data=resampled.tobytes(),
        frame_rate=target_rate,
        sample_width=2,
        channels=1
    )


# ─── CLIP BUILDERS ─────────────────────────────────────────────────────────

def build_background_clip(demand_env):
    """
    Class 0: 45 seconds of DEMAND noise only, no speech.
    """
    noise = load_demand(demand_env)
    # Random offset into the noise file so clips aren't identical
    max_offset = max(0, len(noise) - CLIP_DURATION_MS - 1000)
    offset = random.randint(0, max_offset) if max_offset > 0 else 0
    noise = noise[offset:]
    clip  = loop_to_duration(noise, CLIP_DURATION_MS)
    clip  = normalize(clip)
    return clip

def build_speech_clip(ravdess_files, demand_env, snr_db):
    """
    Classes 1 & 2: RAVDESS speech mixed into DEMAND noise at given SNR.
    Randomly selects and concatenates RAVDESS clips to fill 45 seconds.
    """
    # Build speech track by concatenating random RAVDESS clips
    speech_track = AudioSegment.empty()
    used_files   = []
    while len(speech_track) < CLIP_DURATION_MS:
        chosen = random.choice(ravdess_files)
        seg = AudioSegment.from_wav(chosen)
        seg = seg.set_channels(1).set_sample_width(2)
        seg = resample_audio(seg, SAMPLE_RATE)
        # Add short silence between utterances (natural pause)
        pause = make_silence(random.randint(300, 800))
        speech_track += seg + pause
        used_files.append(os.path.basename(chosen))

    speech_track = speech_track[:CLIP_DURATION_MS]

    # Load and prepare noise
    noise = load_demand(demand_env)
    max_offset = max(0, len(noise) - CLIP_DURATION_MS - 1000)
    offset = random.randint(0, max_offset) if max_offset > 0 else 0
    noise  = noise[offset:]
    noise  = loop_to_duration(noise, CLIP_DURATION_MS)
    noise  = normalize(noise)

    # Mix speech into noise at target SNR
    mixed = mix_at_snr(speech_track, noise, snr_db)
    mixed = normalize(mixed)
    return mixed, used_files

# ─── MAIN ASSEMBLY ──────────────────────────────────────────────────────────

def assemble_dataset():
    ensure_dirs()
    print("=" * 60)
    print("  Vibration Detection — Playback Assembly")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Clips: {CLIPS_PER_CLASS} per class × 3 classes = {CLIPS_PER_CLASS*3} total")
    print("=" * 60)

    # ── Scan RAVDESS ────────────────────────────────────────────
    print("\nScanning RAVDESS files...")
    calm_files     = scan_ravdess(RAVDESS_DIR, CALM_EMOTION_CODES)
    distress_files = scan_ravdess(RAVDESS_DIR, DISTRESS_EMOTION_CODES)
    print(f"  Calm files found     : {len(calm_files)}")
    print(f"  Distress files found : {len(distress_files)}")
    if not calm_files or not distress_files:
        raise RuntimeError("No RAVDESS files found. Check your RAVDESS_DIR path.")

    beep    = make_beep()
    silence = make_silence(GAP_DURATION_MS)
    records = []         # for labels CSV
    all_clips = []       # ordered list of all 180 clips for concatenation
    clip_store = {}      # fname → AudioSegment, avoids re-reading from disk
    clip_idx  = 0

    # ── Generate all clips ──────────────────────────────────────
    for class_label, class_name in CLASSES.items():
        print(f"\n[Class {class_label}: {class_name}]")

        for env in DEMAND_ENVS:
            for i in range(CLIPS_PER_ENV_CLASS):
                #Feasibility cutoff
                if FEASIBILITY_MODE and clip_idx >= 10:
                    break

                clip_idx += 1

                if class_label == 0:
                    # Background only
                    clip = build_background_clip(env)
                    snr_used = None
                    ravdess_used = []
                else:
                    # Speech + noise
                    snr_db = round(random.uniform(SNR_MIN_DB, SNR_MAX_DB), 1)
                    files  = calm_files if class_label == 1 else distress_files
                    clip, ravdess_used = build_speech_clip(files, env, snr_db)
                    snr_used = snr_db

                # Save individual clip WAV
                fname = f"{class_name}_{env}_{i+1:03d}.wav"
                fpath = os.path.join(OUTPUT_DIR, class_name, fname)
                clip.export(fpath, format="wav")

                # Keep in memory to avoid re-reading from disk during assembly
                clip_store[fname] = clip

                # Store for concatenation
                all_clips.append((clip_idx, fname, class_label,
                                  class_name, env, snr_used, ravdess_used))

                print(f"  [{clip_idx:03d}] {fname}"
                      + (f"  SNR={snr_used}dB" if snr_used else ""))

    # ── Shuffle clips for playback ──────────────────────────────
    # Shuffle so sensor doesn't record all of one class consecutively
    random.shuffle(all_clips)

    # ── Build single concatenated playback WAV ──────────────────
    print("\nAssembling full playback WAV...")
    playback    = AudioSegment.empty()
    timestamps  = []  # (playback_position_s, clip_idx, fname, label)
    position_ms = 0

    for order_idx, (clip_idx, fname, class_label,
                    class_name, env, snr_used, ravdess_used) in enumerate(all_clips):

        # Use in-memory clip to avoid a redundant encode/decode cycle
        clip = clip_store[fname]

        # Record timestamp BEFORE appending beep+silence
        # so timestamp points to actual clip start
        clip_start_ms = position_ms + BEEP_DURATION_MS + GAP_DURATION_MS

        playback    += beep + silence + clip
        position_ms += BEEP_DURATION_MS + GAP_DURATION_MS + len(clip)

        timestamps.append({
            "order":            order_idx + 1,
            "original_clip_idx": clip_idx,
            "filename":         fname,
            "class_label":      class_label,
            "class_name":       class_name,
            "demand_env":       env,
            "snr_db":           snr_used if snr_used is not None else "N/A",
            "clip_start_s":     round(clip_start_ms / 1000, 3),
            "clip_end_s":       round((clip_start_ms + CLIP_DURATION_MS) / 1000, 3),
            "beep_start_s":     round((clip_start_ms - GAP_DURATION_MS
                                       - BEEP_DURATION_MS) / 1000, 3),
            "ravdess_files":    "|".join(ravdess_used) if ravdess_used else "N/A",
        })

    # Export full playback WAV
    playback_path = os.path.join(OUTPUT_DIR, "PLAYBACK_FULL.wav")
    print(f"  Exporting {len(playback)/1000/60:.1f} minute WAV...")
    playback.export(playback_path, format="wav")
    print(f"  Saved: {playback_path}")

    # ── Write timestamps CSV ────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "timestamps.csv")
    fieldnames = ["order", "original_clip_idx", "filename", "class_label",
                  "class_name", "demand_env", "snr_db", "beep_start_s",
                  "clip_start_s", "clip_end_s", "ravdess_files"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timestamps)
    print(f"  Timestamps CSV saved: {csv_path}")

    # ── Summary ─────────────────────────────────────────────────
    total_min = len(playback) / 1000 / 60
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Total clips      : {len(all_clips)}")
    print(f"  Playback duration: {total_min:.1f} minutes")
    print(f"  Full WAV         : PLAYBACK_FULL.wav")
    print(f"  Timestamps       : timestamps.csv")
    print(f"  Individual clips : Dataset/class_name/")
    print(f"{'='*60}")
    print(f"\nINSTRUCTIONS FOR PARTNER:")
    print(f"  1. Play PLAYBACK_FULL.wav through the speaker")
    print(f"  2. Start Pi sensor recording at the SAME TIME")
    print(f"  3. Each clip is preceded by a 1kHz beep — visible in PRAAT")
    print(f"  4. Use timestamps.csv to chop sensor recording into segments")

if __name__ == "__main__":
    assemble_dataset()
