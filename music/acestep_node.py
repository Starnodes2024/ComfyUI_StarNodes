import requests
import json
import time
import os
import tempfile
from pathlib import Path
import folder_paths
import torch
import torchaudio

# Set torchaudio backend to avoid torchcodec dependency
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass  # If setting backend fails, continue anyway




class ACEStepMusicGenerator:
    """
    ComfyUI node for ACE Step 1.5 music generation via local API.
    Supports all major parameters for high-quality music generation.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "http://localhost:8001",
                    "multiline": False,
                }),
                "prompt": ("STRING", {
                    "default": "upbeat pop song with guitar",
                    "multiline": True,
                }),
            },
            "optional": {
                "lyrics": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "thinking": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                }),
                "sample_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                }),
                "sample_query": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "use_format": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                }),
                "vocal_language": (["auto", "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "ar", "hi", "bn", "th", "vi", "id", "tr", "nl", "pl"], {
                    "default": "en",
                }),
                "audio_duration": ("FLOAT", {
                    "default": 30.0,
                    "min": 10.0,
                    "max": 600.0,
                    "step": 1.0,
                }),
                "bpm": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 300,
                    "step": 1,
                }),
                "key_scale": ([
                    "C Major", "C Minor", "C# Major", "C# Minor", "Db Major", "Db Minor",
                    "D Major", "D Minor", "D# Major", "D# Minor", "Eb Major", "Eb Minor",
                    "E Major", "E Minor", "F Major", "F Minor", "F# Major", "F# Minor",
                    "Gb Major", "Gb Minor", "G Major", "G Minor", "G# Major", "G# Minor",
                    "Ab Major", "Ab Minor", "A Major", "A Minor", "A# Major", "A# Minor",
                    "Bb Major", "Bb Minor", "B Major", "B Minor"
                ], {
                    "default": "C Major",
                }),
                "time_signature": (["4", "3", "2", "6"], {
                    "default": "4",
                }),
                "inference_steps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                }),
                "use_random_seed": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                }),
                "audio_format": (["mp3", "wav", "flac"], {
                    "default": "mp3",
                }),
                "model": ([
                    "",  # Empty = use server default
                    "auto (use server default)",
                    "acestep-v15-turbo",
                    "acestep-v15-base",
                    "acestep-v15-sft",
                    "acestep-v15-turbo-shift3",
                    "acestep-v15-xl-base",
                    "acestep-v15-xl-sft",
                    "acestep-v15-xl-turbo",
                ], {
                    "default": "",
                }),
                "lm_temperature": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "lm_cfg_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "lm_top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "use_cot_caption": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                }),
                "use_cot_language": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "poll_interval": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                }),
                "timeout": ("INT", {
                    "default": 600,
                    "min": 60,
                    "max": 3600,
                    "step": 30,
                }),
                "subfolder": ("STRING", {
                    "default": "ACE-Step-1.5",
                    "multiline": False,
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "metadata")
    FUNCTION = "generate_music"
    CATEGORY = "⭐StarNodes/Music"
    OUTPUT_NODE = True
    
    def generate_music(self, api_url, prompt, lyrics="", thinking=True, sample_mode=False, 
                      sample_query="", use_format=False, vocal_language="en", 
                      audio_duration=30.0, bpm=120, key_scale="C Major", 
                      time_signature="4", inference_steps=8, guidance_scale=7.0,
                      batch_size=1, use_random_seed=True, seed=-1, audio_format="mp3",
                      model="", lm_temperature=0.85, lm_cfg_scale=2.5, lm_top_p=0.9,
                      use_cot_caption=True, use_cot_language=True, api_key="",
                      poll_interval=2.0, timeout=600, subfolder="ACE-Step-1.5"):
        """
        Generate music using ACE Step 1.5 API.
        
        Args:
            api_url: Base URL of ACE Step API server
            prompt: Music description/caption
            lyrics: Song lyrics
            thinking: Use LM to generate audio codes (higher quality)
            sample_mode: Auto-generate caption/lyrics from sample_query
            sample_query: Natural language description for sample mode
            use_format: Use LM to enhance caption and lyrics
            vocal_language: Language for vocals
            audio_duration: Duration in seconds (10-600)
            bpm: Beats per minute (30-300)
            key_scale: Musical key and scale
            time_signature: Time signature (2/4, 3/4, 4/4, 6/8)
            inference_steps: Number of diffusion steps
            guidance_scale: Prompt guidance strength
            batch_size: Number of audio files to generate
            use_random_seed: Use random seed
            seed: Fixed seed value (when use_random_seed=False)
            audio_format: Output format (mp3, wav, flac)
            model: Specific model to use (empty for default)
            lm_temperature: LM sampling temperature
            lm_cfg_scale: LM CFG scale
            lm_top_p: LM top-p sampling
            use_cot_caption: Use CoT for caption enhancement
            use_cot_language: Use CoT for language detection
            api_key: API authentication key (if required)
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds
            subfolder: Subfolder name for saving audio files (default: ACE-Step-1.5)
            
        Returns:
            Dictionary containing audio waveform data and metadata
        """
        
        print(f"[ACE Step] Starting music generation...")
        print(f"[ACE Step] API URL: {api_url}")
        print(f"[ACE Step] Prompt: {prompt[:100]}...")
        
        # Build request payload
        payload = {
            "prompt": prompt,
            "lyrics": lyrics,
            "thinking": thinking,
            "sample_mode": sample_mode,
            "use_format": use_format,
            "vocal_language": vocal_language if vocal_language != "auto" else "en",
            "audio_duration": audio_duration,
            "bpm": bpm,
            "key_scale": key_scale,
            "time_signature": time_signature,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "batch_size": batch_size,
            "use_random_seed": use_random_seed,
            "seed": seed,
            "audio_format": audio_format,
            "lm_temperature": lm_temperature,
            "lm_cfg_scale": lm_cfg_scale,
            "lm_top_p": lm_top_p,
            "use_cot_caption": use_cot_caption,
            "use_cot_language": use_cot_language,
        }
        
        # Add optional parameters
        if sample_query:
            payload["sample_query"] = sample_query
            
        # Parse model selection
        # Empty string or "auto" means use server default (don't send model parameter)
        if model and model not in ["", "auto (use server default)"]:
            # Remove " (default)" suffix if present
            model_name = model.replace(" (default)", "")
            payload["model"] = model_name
            
        if api_key:
            payload["ai_token"] = api_key
        
        # Submit task
        try:
            release_url = f"{api_url.rstrip('/')}/release_task"
            print(f"[ACE Step] Submitting task to {release_url}")
            
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                release_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") != 200:
                error_msg = result.get("error", "Unknown error")
                raise Exception(f"API error: {error_msg}")
            
            task_data = result.get("data", {})
            task_id = task_data.get("task_id")
            
            if not task_id:
                raise Exception("No task_id returned from API")
            
            print(f"[ACE Step] Task submitted successfully. Task ID: {task_id}")
            print(f"[ACE Step] Queue position: {task_data.get('queue_position', 'N/A')}")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to submit task: {str(e)}")
        
        # Poll for results
        query_url = f"{api_url.rstrip('/')}/query_result"
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"Task timeout after {timeout} seconds")
            
            try:
                query_response = requests.post(
                    query_url,
                    json={"task_id_list": [task_id]},
                    headers=headers,
                    timeout=30
                )
                query_response.raise_for_status()
                
                query_result = query_response.json()
                
                if query_result.get("code") != 200:
                    error_msg = query_result.get("error", "Unknown error")
                    raise Exception(f"Query error: {error_msg}")
                
                tasks = query_result.get("data", [])
                if not tasks:
                    raise Exception("No task data returned")
                
                task_info = tasks[0]
                status = task_info.get("status")
                
                if status == 1:  # Success
                    print(f"[ACE Step] Task completed successfully!")
                    result_str = task_info.get("result", "[]")
                    result_list = json.loads(result_str)
                    
                    if not result_list:
                        raise Exception("No audio files generated")
                    
                    # Create subfolder if specified
                    if subfolder:
                        save_dir = os.path.join(self.output_dir, subfolder)
                        os.makedirs(save_dir, exist_ok=True)
                    else:
                        save_dir = self.output_dir
                    
                    # Download audio files and load waveforms
                    audio_files = []
                    waveforms = []
                    sample_rates = []
                    
                    for idx, item in enumerate(result_list):
                        file_url = item.get("file", "")
                        if not file_url:
                            continue
                        
                        # Construct full URL
                        if file_url.startswith("/"):
                            full_url = f"{api_url.rstrip('/')}{file_url}"
                        else:
                            full_url = file_url
                        
                        print(f"[ACE Step] Downloading audio {idx + 1}/{len(result_list)}...")
                        
                        audio_response = requests.get(full_url, headers=headers, timeout=60)
                        audio_response.raise_for_status()
                        
                        # Save to output directory
                        filename = f"acestep_{task_id}_{idx}.{audio_format}"
                        filepath = os.path.join(save_dir, filename)
                        
                        with open(filepath, "wb") as f:
                            f.write(audio_response.content)
                        
                        print(f"[ACE Step] Saved: {filepath}")
                        
                        audio_files.append({
                            "filename": filename,
                            "subfolder": subfolder if subfolder else "",
                            "filepath": filepath,
                            "prompt": item.get("prompt", ""),
                            "lyrics": item.get("lyrics", ""),
                            "metas": item.get("metas", {}),
                            "seed": item.get("seed_value", ""),
                            "lm_model": item.get("lm_model", ""),
                            "dit_model": item.get("dit_model", ""),
                        })
                    
                    print(f"[ACE Step] Generation complete! Downloaded {len(audio_files)} file(s)")
                    
                    # Reload audio files to ensure correct format (VHS-style workaround)
                    # This ensures compatibility with ComfyUI audio nodes
                    print(f"[ACE Step] Reloading audio files for proper format...")
                    waveforms_reloaded = []
                    sample_rates_reloaded = []
                    
                    for audio_info in audio_files:
                        filepath = audio_info["filepath"]
                        filename = audio_info["filename"]
                        
                        # Try multiple methods to load audio without torchcodec
                        loaded = False
                        
                        # Method 1: Try soundfile directly (most compatible)
                        try:
                            import soundfile as sf
                            audio_data, sample_rate = sf.read(filepath, dtype='float32')
                            # soundfile returns [Samples, Channels], we need [Channels, Samples]
                            if audio_data.ndim == 1:
                                # Mono audio
                                waveform = torch.from_numpy(audio_data).unsqueeze(0)  # [S] -> [1, S]
                            else:
                                # Stereo or multi-channel
                                waveform = torch.from_numpy(audio_data.T)  # [S, C] -> [C, S]
                            
                            # Add batch dimension: [C, S] -> [1, C, S]
                            waveform = waveform.unsqueeze(0)
                            waveforms_reloaded.append(waveform)
                            sample_rates_reloaded.append(sample_rate)
                            loaded = True
                            print(f"[ACE Step] Loaded with soundfile: {filename}, shape={waveform.shape}, sr={sample_rate}")
                        except Exception as e:
                            print(f"[ACE Step] soundfile method failed for {filename}: {e}")
                        
                        # Method 2: Try torchaudio with explicit format
                        if not loaded:
                            try:
                                waveform, sample_rate = torchaudio.load(filepath, format="mp3")
                                if waveform.ndim == 2:
                                    waveform = waveform.unsqueeze(0)  # [C, S] -> [1, C, S]
                                waveforms_reloaded.append(waveform)
                                sample_rates_reloaded.append(sample_rate)
                                loaded = True
                                print(f"[ACE Step] Loaded with torchaudio: {filename}, shape={waveform.shape}, sr={sample_rate}")
                            except Exception as e:
                                print(f"[ACE Step] torchaudio method failed for {filename}: {e}")
                        
                        # Fallback: Use zeros
                        if not loaded:
                            print(f"[ACE Step] WARNING: Could not load {filename}, using silent audio")
                            waveforms_reloaded.append(torch.zeros(1, 2, 44100))
                            sample_rates_reloaded.append(44100)
                    
                    # Prepare final waveform
                    if waveforms_reloaded:
                        # Check for consistent sample rates
                        if len(set(sample_rates_reloaded)) > 1:
                            print(f"[ACE Step] Warning: Inconsistent sample rates: {set(sample_rates_reloaded)}")
                        
                        # For single file: use as-is [1, C, S]
                        # For multiple files: concatenate along batch dimension
                        if len(waveforms_reloaded) == 1:
                            waveform_output = waveforms_reloaded[0]  # [1, C, S]
                        else:
                            # Concatenate along batch dimension (dim=0)
                            waveform_output = torch.cat(waveforms_reloaded, dim=0)  # [B, C, S]
                        
                        sample_rate = sample_rates_reloaded[0] if sample_rates_reloaded else 44100
                        print(f"[ACE Step] Final waveform shape: {waveform_output.shape}, sample_rate: {sample_rate}")
                    else:
                        waveform_output = torch.zeros(1, 2, 44100)
                        sample_rate = 44100
                    
                    # Prepare metadata as JSON string
                    metadata = json.dumps({
                        "files": audio_files,
                        "task_id": task_id,
                        "batch_size": len(audio_files),
                        "sample_rate": sample_rate,
                    }, indent=2)
                    
                    # Return audio data in ComfyUI format
                    return (
                        {
                            "waveform": waveform_output,  # Already has correct shape [1, C, S] or [B, C, S]
                            "sample_rate": sample_rate,
                        },
                        metadata,
                    )
                    
                elif status == 2:  # Failed
                    raise Exception(f"Task failed: {task_info.get('error', 'Unknown error')}")
                
                else:  # Still processing (status == 0)
                    print(f"[ACE Step] Task in progress... (elapsed: {int(elapsed)}s)")
                    time.sleep(poll_interval)
                    
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to query task status: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "ACEStepMusicGenerator": ACEStepMusicGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACEStepMusicGenerator": "⭐ Star ACE Step Music Gen (Local API)",
}
