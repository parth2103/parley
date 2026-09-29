"""Automated Live WebRTC Session Runner for Parley C0 Baseline (n=18).

Connects over SmallWebRTC to the active local Parley server (http://127.0.0.1:8000),
streams 18 pre-rendered speech turns across policy lookups, RAG coverage queries,
FNOL claim filings, and adjuster callbacks, and parses the resulting server log
for per-stage percentiles (STT, LLM, TTS, True V2V).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave
from typing import Any

from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame
import httpx
import numpy as np

# 18 Test Prompts spanning RAG, Tool Calling, Coverage, and Claim Filing
BASELINE_PROMPTS = [
    "What is the comprehensive deductible on my auto policy POL-4401?",
    "What is the dwelling coverage limit on homeowners policy HOM-302?",
    "Does my homeowners policy cover damage from burst frozen pipes?",
    "What is my scheduled personal property limit on rider PRP-205?",
    "Can you schedule an adjuster to call me tomorrow morning at 10am at 555-839-2001 for policy POL-4401?",
    "A rock cracked my front windshield on the highway. What is my deductible for glass repair on POL-4401?",
    "What is my roadside assistance towing limit on auto policy POL-4401?",
    "Is sewer backup damage covered under my homeowners policy HOM-302?",
    "I need to open a claim on POL-4401 for an accident that happened on 2026-02-15 on 5th Ave.",
    "What is my rental car reimbursement benefit on policy POL-4401?",
    "Does my homeowners policy cover damage from floods or earthquakes?",
    "How many days do I have to submit a formal claim notice?",
    "Can you schedule a callback on Friday afternoon at 2pm at 555-019-4820 regarding HOM-302?",
    "My basement pipes burst and caused water damage on 2026-01-20. Can you open a claim on HOM-302?",
    "What is the deductible for collision coverage on my auto policy POL-4401?",
    "What is the deductible for scheduled electronics on property rider PRP-205?",
    "Is veterinary care for an injured dog covered under my auto policy?",
    "What is the total maximum payout for rental car reimbursement on POL-4401?",
]


def generate_turn_audio(text: str, wav_path: Path) -> list[AudioFrame]:
    """Generate 16kHz mono audio via macOS say, and chunk into 20ms (320-sample) AudioFrames."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["say", "-o", str(wav_path), "--data-format=LEI16@16000", text],
        check=True,
    )
    with wave.open(str(wav_path), "rb") as w:
        rate = w.getframerate()
        n_frames = w.getnframes()
        pcm_bytes = w.readframes(n_frames)

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    chunk_size = int(rate * 0.020)  # 320 samples for 20ms at 16kHz
    audio_frames = []

    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")
        frame = AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = rate
        audio_frames.append(frame)

    return audio_frames


class ControlledAudioTrack(AudioStreamTrack):
    """AudioStreamTrack that queues speech turns followed by silence."""

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.queue: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self._silent_chunk = np.zeros(int(sample_rate * 0.020), dtype=np.int16)

    def enqueue_turn(self, speech_frames: list[AudioFrame], silence_duration_s: float = 1.0):
        for f in speech_frames:
            self.queue.put_nowait(f)
        # Append silence frames so Silero VAD triggers user_speech_stopped
        silence_frame_count = int(silence_duration_s / 0.020)
        for _ in range(silence_frame_count):
            frame = AudioFrame.from_ndarray(self._silent_chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
            self.queue.put_nowait(frame)

    async def recv(self) -> AudioFrame:
        pts, time_base = await self.next_timestamp()
        try:
            frame = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            # Continuous background silence frame
            frame = AudioFrame.from_ndarray(self._silent_chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
        frame.pts = pts
        frame.time_base = time_base
        return frame


async def run_live_webrtc_session(server_url: str = "http://127.0.0.1:8000") -> None:
    print("=" * 75)
    print(f"PARLEY LIVE VOICE RE-BASELINE (n={len(BASELINE_PROMPTS)} turns)")
    print(f"Server: {server_url} | Model: Groq openai/gpt-oss-120b | TTS: Cartesia sonic-3.5")
    print("=" * 75)

    tmp_dir = Path("eval/results/webrtc_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Pre-synthesizing turn audio files...")
    turn_audio_map = []
    for idx, prompt in enumerate(BASELINE_PROMPTS, 1):
        wav_p = tmp_dir / f"turn_{idx:02d}.wav"
        frames = generate_turn_audio(prompt, wav_p)
        turn_audio_map.append((prompt, frames))
        print(f"  [Turn {idx:02d}] Generated {len(frames)} frames ({len(frames)*20}ms): \"{prompt[:50]}...\"")

    print("\n2. Initializing WebRTC Peer Connection...")
    pc = RTCPeerConnection()
    channel = pc.createDataChannel("parley")
    audio_track = ControlledAudioTrack(sample_rate=16000)
    pc.addTrack(audio_track)

    # Monitor incoming agent audio
    agent_audio_received = asyncio.Event()
    last_agent_audio_time = [time.perf_counter()]

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            async def read_agent_audio():
                while True:
                    try:
                        frame = await track.recv()
                        last_agent_audio_time[0] = time.perf_counter()
                        agent_audio_received.set()
                    except Exception:
                        break
            asyncio.create_task(read_agent_audio())

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    print("3. Connecting to Parley SmallWebRTC endpoint /api/offer...")
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        resp = await http_client.post(
            f"{server_url}/api/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Server rejected WebRTC offer: {resp.status_code} - {resp.text}")
        answer_data = resp.json()
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
        )
        print("   Connected! WebRTC session active.")

        # Heartbeat loop over data channel
        async def send_heartbeat():
            while True:
                await asyncio.sleep(1.0)
                if channel.readyState == "open":
                    try:
                        channel.send("heartbeat")
                    except Exception:
                        pass
        heartbeat_task = asyncio.create_task(send_heartbeat())

        # Give connection 1 second to stabilize
        await asyncio.sleep(1.0)

        # 4. Stream turns sequentially
        print("\n4. Streaming turns...")
        for turn_idx, (prompt, frames) in enumerate(turn_audio_map, 1):
            turn_start_wall = time.perf_counter()
            agent_audio_received.clear()
            print(f"\n>> [Turn {turn_idx:02d}/{len(BASELINE_PROMPTS)}] Speaking: \"{prompt}\"")

            # Stream user utterance + 1.2s trailing silence to trigger VAD user_speech_stopped
            audio_track.enqueue_turn(frames, silence_duration_s=1.2)

            # Wait for agent audio to begin arriving
            try:
                await asyncio.wait_for(agent_audio_received.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                print(f"   [Turn {turn_idx:02d}] Timeout waiting for agent audio response.")
                continue

            # Wait for agent to finish speaking (silence > 1.8s)
            while True:
                await asyncio.sleep(0.3)
                if time.perf_counter() - last_agent_audio_time[0] > 1.8:
                    break

            turn_elapsed = time.perf_counter() - turn_start_wall
            print(f"   [Turn {turn_idx:02d}] Agent finished speaking (turn duration: {turn_elapsed:.2f}s). Pausing...")
            await asyncio.sleep(1.2)  # Natural inter-turn pause

        # 5. Clean teardown
        print("\n5. All turns completed. Closing session...")
        heartbeat_task.cancel()
        await pc.close()


def parse_latest_session_log(logs_dir: Path | str = "logs") -> dict[str, Any]:
    """Parse the most recent Parley server log for True V2V and per-stage latency."""
    log_files = sorted(Path(logs_dir).glob("parley_*.log"), key=os.path.getmtime, reverse=True)
    if not log_files:
        raise FileNotFoundError("No log files found in logs/ directory.")

    log_path = log_files[0]
    print(f"\nParsing server log: {log_path}")

    stt_ttfb: list[float] = []
    llm_ttfb: list[float] = []
    tts_ttfb: list[float] = []
    tts_ttfa: list[float] = []
    true_v2v: list[float] = []
    turns_data: dict[int, dict[str, float]] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            turn_m = re.search(r"turn=(\d+)", line)
            if not turn_m:
                continue
            t_num = int(turn_m.group(1))
            if t_num not in turns_data:
                turns_data[t_num] = {}

            # True V2V
            v2v_m = re.search(r"stage=TRUE_V2V.*true_v2v_s=([\d\.]+)", line)
            if v2v_m:
                val = float(v2v_m.group(1))
                true_v2v.append(val)
                turns_data[t_num]["true_v2v"] = val

            # STT
            stt_m = re.search(r"stage=STT.*ttfb_s=([\d\.]+)", line)
            if stt_m:
                val = float(stt_m.group(1))
                stt_ttfb.append(val)
                turns_data[t_num]["stt"] = val

            # LLM
            llm_m = re.search(r"stage=LLM.*ttfb_s=([\d\.]+)", line)
            if llm_m:
                val = float(llm_m.group(1))
                llm_ttfb.append(val)
                turns_data[t_num]["llm"] = val

            # TTS
            tts_m = re.search(r"stage=TTS.*ttfb_s=([\d\.]+)\s+ttfa_s=([\d\.]+)", line)
            if tts_m:
                ttfb_val = float(tts_m.group(1))
                ttfa_val = float(tts_m.group(2))
                tts_ttfb.append(ttfb_val)
                tts_ttfa.append(ttfa_val)
                turns_data[t_num]["tts_ttfa"] = ttfa_val

    def calc_pct(arr: list[float]) -> dict[str, float]:
        if not arr:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0, "n": 0}
        s = sorted(arr)
        p50 = s[len(s) // 2]
        p90 = s[int(len(s) * 0.90)]
        p95 = s[int(len(s) * 0.95)]
        return {
            "p50": round(p50, 4),
            "p90": round(p90, 4),
            "p95": round(p95, 4),
            "mean": round(sum(s) / len(s), 4),
            "n": len(s),
        }

    results = {
        "log_path": str(log_path),
        "total_complete_turns": len(true_v2v),
        "stt": calc_pct(stt_ttfb),
        "llm": calc_pct(llm_ttfb),
        "tts_ttfa": calc_pct(tts_ttfa),
        "true_v2v": calc_pct(true_v2v),
        "turns": turns_data,
    }

    print("\n" + "=" * 75)
    print("PARLEY C0 RE-BASELINE SUMMARY METRICS")
    print("=" * 75)
    print(f"Total Complete Turns Captured: {results['total_complete_turns']}")
    print("-" * 75)
    print(f"STT (Nova-3):           p50 = {results['stt']['p50']:.3f} s | p95 = {results['stt']['p95']:.3f} s (n={results['stt']['n']})")
    print(f"LLM (Groq gpt-oss-120b): p50 = {results['llm']['p50']:.3f} s | p95 = {results['llm']['p95']:.3f} s (n={results['llm']['n']})")
    print(f"TTS TTFA (Cartesia):    p50 = {results['tts_ttfa']['p50']:.3f} s | p95 = {results['tts_ttfa']['p95']:.3f} s (n={results['tts_ttfa']['n']})")
    print(f"True V2V (Direct TS):   p50 = {results['true_v2v']['p50']:.3f} s | p95 = {results['true_v2v']['p95']:.3f} s (n={results['true_v2v']['n']})")
    print("=" * 75)

    out_file = Path("eval/results/c0_rebaseline_metrics.json")
    with open(out_file, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)
    print(f"Results exported to {out_file}")

    return results


async def main():
    await run_live_webrtc_session()
    await asyncio.sleep(1.0)
    parse_latest_session_log()


if __name__ == "__main__":
    asyncio.run(main())
