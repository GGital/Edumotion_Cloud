import base64
import requests
import json
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from pathlib import Path

class VLMServiceClient:
    """Client for vLLM service that mimics your original inference pattern"""
    
    def __init__(self, base_url: str = "http://localhost:8081", api_key: str = "EMPTY"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def encode_video_to_base64(self, video_path: str, num_frames: int = 12) -> str:
        """
        Encode video to base64 string with frame sampling
        Mimics the num_frames parameter from your original function
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Sample frames evenly across the video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        # Create a simple video-like representation by encoding frames
        encoded_frames = []
        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            encoded_frames.append(frame_base64)
        
        # For vLLM service, we'll send as a JSON array of base64 frames
        return json.dumps(encoded_frames)

def vlm_inference_comp(client: VLMServiceClient, videoA_path: str, videoB_path: str, 
                      sys_prompt: str, threshold: float = 0.5, additional_prompt: str = None,
                      num_frames: int = 12, max_new_tokens: int = 2048) -> List[str]:
    """
    Replaces your original vlm_inference_comp function to use vLLM service
    
    Args:
        client: VLMServiceClient instance (replaces model, processor)
        videoA_path: Path to first video
        videoB_path: Path to second video  
        sys_prompt: System prompt
        threshold: Threshold parameter for build_prompt
        additional_prompt: Additional prompt text
        num_frames: Number of frames to sample (replaces processor num_frames)
        max_new_tokens: Maximum tokens to generate (replaces model max_new_tokens)
        
    Returns:
        List[str]: Generated text output (same format as your original function)
    """
    
    # Build user prompt (you'll need to provide this function or I can create it)
    user_prompt = build_prompt(threshold, additional_prompt)
    
    # Encode videos to base64 (mimicking your video processing)
    try:
        videoA_b64 = client.encode_video_to_base64(videoA_path, num_frames)
        videoB_b64 = client.encode_video_to_base64(videoB_path, num_frames)
    except Exception as e:
        raise ValueError(f"Error processing videos: {e}")
    
    # Create messages in the same structure as your original
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": sys_prompt}
            ]
        },
        {
            "role": "user", 
            "content": [
                {"type": "video", "video": f"data:video/mp4;base64,{videoA_b64}"},
                {"type": "video", "video": f"data:video/mp4;base64,{videoB_b64}"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    
    # Prepare request payload for vLLM service
    payload = {
        "model": "qwen2.5-vl-7b-instruct",
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,  # You can adjust this or make it a parameter
        "stream": False
    }
    
    try:
        # Make request to vLLM service
        response = requests.post(
            f"{client.base_url}/v1/chat/completions",
            headers=client.headers,
            json=payload,
            timeout=300  # 5 minute timeout for video processing
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract generated text (mimicking your output format)
        output_text = [result["choices"][0]["message"]["content"]]
        
        return output_text
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"vLLM service request failed: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format from vLLM service: {e}")

def build_prompt(threshold: float, specific_term: str = None) -> str:
    # Extra line for point #3 if a specific term is provided
    if specific_term:
        focus_line = f"\n   - Pay special attention to differences in **{specific_term}**."
    else:
        focus_line = ""

    User_Prompt = f"""You are given two videos:
- **Learner's Doing**: This shows how the learner performs the task.
- **Teacher's Doing**: This shows the correct or ideal way to perform the task.

Please follow these steps:

1. Describe the motion in **the Learner's Doing** step-by-step. Focus on what the subject does, the direction and intensity of movement, and the action sequence.
2. Do the same for **the Teacher's Doing**, using the same level of detail.
3. Compare the two videos specifically in terms of motion. Highlight:
   - Differences in action types
   - Timing or duration of actions
   - Number of people or moving objects
   - Direction, speed, or style of movement{focus_line}
4. Provide a **similarity score** out of 1 that reflects how similar the two performances are in terms of motion only.
5. You are also given a similarity threshold `T` (a float between 0 and 1). If `Score / 1 < T`, provide clear and supportive **feedback to the learner**, focusing on what they can improve to better match the teacher’s motion. Be detailed, encouraging, and constructive.

**Threshold (T)**: {threshold:.2f}

**Output Format**:
--- Learner's Doing Motion Description ---
[step-by-step motion description]

--- Teacher's Doing Motion Description ---
[step-by-step motion description]

--- Motion Comparison ---
[detailed analysis of key differences in movement, style, duration, direction, etc.]

--- Similarity Score (Motion Only) ---
Score: [X]/1

--- Suggestions for Improvement (if Score < T) ---
[Only include this if Score/1 < T. Give encouraging and actionable suggestions to help the learner adjust their motion to better match the teacher.]
"""
    return User_Prompt