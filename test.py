import soundfile as sf
import numpy as np
import time
import os
try:
    import axengine
    os.environ['AX_INFER'] = "True"
    os.environ["AXMODEL_DIR"] = "../../VoxCPM.Axera/model_convert"
    output_path = "output_streaming_ax.wav"
except:
    output_path = "output_streaming.wav"

os.system(f"rm {output_path}")

from voxcpm import VoxCPM

# os.environ["DEBUG_TIME"] = "true"
model = VoxCPM.from_pretrained("../../VoxCPM-0.5B", zipenhancer_model_id="speech_zipenhancer_ans_multiloss_16k_base")

t1 = time.time()
# Streaming
chunks = []
for chunk in model.generate_streaming(
    text = "Streaming text to speech is easy with VoxCPM!",
    prompt_wav_path="asset/en_man1.mp3",      # optional: path to a prompt speech for voice cloning
    prompt_text="Because he has zero capacity to respond to the two and a half hour",          # optional: reference text
    cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
    normalize=True,           # enable external TN tool
    denoise=True,             # enable external Denoise tool
    retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
    retry_badcase_max_times=3,  # maximum retrying times
    retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech

):
    chunks.append(chunk)
wav = np.concatenate(chunks)

t2 = time.time()
print(f"use time {t2-t1} s")

sf.write(output_path, wav, 16000)
print(f"saved: {output_path}")

os.system(f"chmod 777 {output_path}")