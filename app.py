import streamlit as st
import numpy as np
import time
import librosa
import soundfile as sf
from io import BytesIO

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="WB Rhythm Challenge", page_icon="🎵", layout="centered")

# -----------------------------
# Predefined songs (title -> (BPM, description))
# -----------------------------
PRESET_SONGS = {
    "Lo‑fi Chill (100 BPM)": (100, "Smooth lo‑fi groove—steady 4/4 beat."),
    "Upbeat Pop (128 BPM)": (128, "Dance‑floor friendly—classic 128 BPM."),
    "Hip‑Hop Bounce (90 BPM)": (90, "Laid‑back bounce—tap on the downbeats."),
    "Drum & Bass (174 BPM)": (174, "Fast and energetic—tight timing needed."),
    "Afrobeats (110 BPM)": (110, "Syncopated feel—aim for the pulse."),
}

# -----------------------------
# Helpers
# -----------------------------
def estimate_bpm_from_audio(file_bytes: bytes) -> float:
    """Estimate BPM using librosa; returns float BPM or raises Exception."""
    try:
        # Load audio from bytes
        data, sr = sf.read(BytesIO(file_bytes))
        # If stereo, convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        # Librosa onset strength + tempo estimation
        onset_env = librosa.onset.onset_strength(y=data, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        if tempo is None or len(tempo) == 0:
            raise ValueError("Tempo estimation failed.")
        # Use median tempo for stability
        bpm = float(np.median(tempo))
        # Clamp to reasonable range
        if bpm < 40 or bpm > 240:
            raise ValueError(f"Unusual BPM detected: {bpm:.1f}")
        return bpm
    except Exception as e:
        raise RuntimeError(f"Audio BPM estimation error: {e}")

def init_state():
    defaults = {
        "selected_mode": "Preset",
        "selected_song": list(PRESET_SONGS.keys())[0],
        "estimated_bpm": None,
        "challenge_active": False,
        "start_time": None,
        "tap_log": [],
        "beat_times": [],
        "score": 0,
        "streak": 0,
        "round_length_beats": 16,
        "tolerance_ms": 120,  # tap within ±120 ms of beat
        "feedback": "",
        "last_latency_ms": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def generate_beat_schedule(bpm: float, beats: int, start_time: float) -> list:
    """Generate absolute timestamps (seconds since epoch) for each beat."""
    interval = 60.0 / bpm
    return [start_time + i * interval for i in range(beats)]

def evaluate_tap(tap_time: float, beat_times: list, tolerance_ms: int):
    """Find nearest beat and compute latency; return (hit: bool, latency_ms: float)."""
    if not beat_times:
        return False, None
    # Find nearest beat
    diffs = np.array(beat_times) - tap_time
    idx = int(np.argmin(np.abs(diffs)))
    latency_s = abs(diffs[idx])
    latency_ms = latency_s * 1000.0
    hit = latency_ms <= tolerance_ms
    return hit, latency_ms

def start_challenge(bpm: float):
    st.session_state.estimated_bpm = bpm
    st.session_state.start_time = time.time() + 1.0  # 1s countdown
    st.session_state.beat_times = generate_beat_schedule(
        bpm=bpm,
        beats=st.session_state.round_length_beats,
        start_time=st.session_state.start_time
    )
    st.session_state.tap_log = []
    st.session_state.score = 0
    st.session_state.streak = 0
    st.session_state.feedback = ""
    st.session_state.last_latency_ms = None
    st.session_state.challenge_active = True

def end_challenge():
    st.session_state.challenge_active = False
    st.session_state.feedback = "🏁 Round finished! Tap again to start a new round."

# -----------------------------
# Initialize state
# -----------------------------
init_state()

# -----------------------------
# UI
# -----------------------------
st.title("🎵 WB Rhythm Challenge")
st.caption("Tap to the beat—match the rhythm based on the song’s tempo. Upload your own track or pick a preset.")

with st.sidebar:
    st.header("Game settings")
    st.session_state.round_length_beats = st.slider("Round length (beats)", 8, 64, st.session_state.round_length_beats, step=4)
    st.session_state.tolerance_ms = st.slider("Timing tolerance (ms)", 60, 250, st.session_state.tolerance_ms, step=10)
    st.write("Lower tolerance = harder game. Aim for tight timing!")

# Mode selection
mode = st.radio("Select mode", ["Preset", "Upload"], index=0 if st.session_state.selected_mode == "Preset" else 1)
st.session_state.selected_mode = mode

bpm_source = None
audio_bytes = None

if mode == "Preset":
    st.session_state.selected_song = st.selectbox("Choose a song", list(PRESET_SONGS.keys()),
                                                  index=list(PRESET_SONGS.keys()).index(st.session_state.selected_song))
    preset_bpm, desc = PRESET_SONGS[st.session_state.selected_song]
    st.info(f"Selected: {st.session_state.selected_song} • {desc}")
    bpm_source = preset_bpm
else:
    uploaded = st.file_uploader("Upload audio (WAV/FLAC/OGG/MP3*)", type=["wav", "flac", "ogg", "mp3"])
    if uploaded is not None:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes, format="audio/*")
        if st.button("Estimate BPM"):
            with st.spinner("Estimating BPM..."):
                try:
                    bpm_est = estimate_bpm_from_audio(audio_bytes)
                    st.session_state.estimated_bpm = bpm_est
                    st.success(f"Estimated BPM: {bpm_est:.1f}")
                except Exception as e:
                    st.warning(f"Could not estimate BPM: {e}")
    if st.session_state.estimated_bpm:
        bpm_source = st.session_state.estimated_bpm

# Start challenge
colA, colB = st.columns([1, 1])
if colA.button("🚀 Start / Restart Round"):
    if bpm_source is None:
        st.warning("Select a preset or estimate BPM from an upload first.")
    else:
        start_challenge(bpm_source)

# Live status
if st.session_state.challenge_active and st.session_state.estimated_bpm:
    bpm = st.session_state.estimated_bpm if mode == "Upload" else bpm_source
    interval = 60.0 / bpm
    now = time.time()
    countdown = st.session_state.start_time - now
    if countdown > 0:
        st.info(f"Get ready... starting in {countdown:.1f}s")
    else:
        # Show beat progress
        beats_total = len(st.session_state.beat_times)
        beats_elapsed = sum(1 for t in st.session_state.beat_times if t <= now)
        beats_remaining = max(beats_total - beats_elapsed, 0)

        st.subheader(f"BPM: {bpm:.1f} • Interval: {interval*1000:.0f} ms • Beats left: {beats_remaining}")
        st.progress(min(beats_elapsed / beats_total, 1.0))

        # Visual pulse indicator
        pulse_phase = ((now - st.session_state.start_time) % interval) / interval
        pulse_size = 0.6 + 0.4 * np.sin(2 * np.pi * pulse_phase)
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; align-items:center; padding: 12px;">
                <div style="width:{int(120*pulse_size)}px; height:{int(120*pulse_size)}px; border-radius:50%;
                            background:linear-gradient(135deg,#7c4dff,#00e5ff); box-shadow:0 0 20px rgba(0,229,255,0.6);">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tap button
        tapped = colB.button("🎯 Tap")
        if tapped:
            tap_time = time.time()
            st.session_state.tap_log.append(tap_time)
            hit, latency_ms = evaluate_tap(tap_time, st.session_state.beat_times, st.session_state.tolerance_ms)
            st.session_state.last_latency_ms = latency_ms
            if hit:
                st.session_state.score += 10
                st.session_state.streak += 1
                st.session_state.feedback = f"✅ On beat! Latency: {latency_ms:.0f} ms • +10 points"
            else:
                st.session_state.streak = 0
                st.session_state.feedback = f"❌ Off beat. Latency: {latency_ms:.0f} ms (tolerance ±{st.session_state.tolerance_ms} ms)"

        # End round automatically when last beat passes
        if beats_elapsed >= beats_total:
            end_challenge()

# Scoreboard
c1, c2, c3 = st.columns(3)
c1.metric("Score", st.session_state.score)
c2.metric("Streak", st.session_state.streak)
c3.metric("Tolerance (ms)", st.session_state.tolerance_ms)

# Feedback
if st.session_state.feedback:
    if "On beat" in st.session_state.feedback:
        st.success(st.session_state.feedback)
    elif "Off beat" in st.session_state.feedback:
        st.warning(st.session_state.feedback)
    else:
        st.info(st.session_state.feedback)

# Tips
st.divider()
st.caption("Tips: Tap right on the pulse. Lower tolerance for harder rounds. Upload audio to auto‑estimate BPM (requires clear beats).")
st.caption("Built with Streamlit • WB_chatbot games")

# Notes:
# - MP3 decoding via soundfile may vary by environment; WAV/FLAC recommended for best BPM estimation.
# - If BPM estimation fails, try a track with clear drums or use presets.
