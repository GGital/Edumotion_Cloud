import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

sys_prompt_comp = """ You are a video reasoning assistant specialized in analyzing and comparing motion in videos.
You must reason step by step about the motion occurring in each video, and then compare the motion patterns. Always focus primarily on human and object movement, action sequence, and timing.
You will be provided with a similarity threshold (a floating number between 0 and 1). After assigning a similarity score out of 1, compare it against the threshold. If the score is below the threshold, give **clear, actionable feedback** that helps the person in **Video A** improve their performance to match **Video B**. The feedback should sound like advice for a learner, using an encouraging and instructional tone.
"""


sys_prompt_ask = """ You are a vision-language assistant specialized in analyzing video quality and content clarity.
You will watch a video and extract fine-grained details, such as small object interactions, subtle movements, facial expressions, or precise gestures.
You must assess how clearly these minor details are represented and assign a score based on how well the video conveys them.
Be fair, detailed, and reasoned in your grading.
"""

user_prompt_ask = """ Please analyze the following video for minor or subtle visual details.

1. Identify any small or subtle elements you notice in the video. These may include:
   - Hand gestures, facial expressions, or posture changes
   - Object handling or minor movements
   - Visual cues in the environment (e.g., signs, symbols, props)
2. Explain whether each identified detail is **clear and easy to observe**, or **unclear/ambiguous** due to motion, angle, or visibility.
3. Based on your analysis, assign a score out of 100 for how well the video presents these minor details.

**Output Format**:
--- Minor Detail Observations ---
[List of small or subtle visual elements you identified]

--- Representation Assessment ---
[For each, explain how clearly it is shown and why — including visibility, motion clarity, and framing]

--- Minor Detail Clarity Score ---
Score: [XX]/100
"""


def build_prompt(threshold: float) -> str:
  User_Prompt = f""" You are given two videos: Video A and Video B.

Please follow these steps:

1. Describe the motion in **Video A** step-by-step. Focus on what the subject does, the direction and intensity of movement, and the action sequence.
2. Do the same for **Video B**, using the same level of detail.
3. Compare the two videos specifically in terms of motion. Highlight:
   - Differences in action types
   - Timing or duration of actions
   - Number of people or moving objects
   - Direction, speed, or style of movement
4. Provide a **similarity score** out of 1 that reflects how similar the two videos are in terms of motion only.
5. You are also given a similarity threshold `T` (a float between 0 and 1). If `Score / 1 < T`, provide helpful suggestions **as if you're giving feedback to a learner** who is trying to improve their performance in Video A to better match Video B. Be specific and constructive.

**Threshold (T)**: {threshold:.2f}

**Output Format**:
--- Video A Motion Description ---
[step-by-step motion description]

--- Video B Motion Description ---
[step-by-step motion description]

--- Motion Comparison ---
[detailed analysis of key differences in movement, style, duration, direction, etc.]

--- Similarity Score (Motion Only) ---
Score: [X]/1

--- Suggestions for Improvement (if Score < T) ---
[Only include this if Score/1 < T. Give specific, supportive advice to the learner performing in Video A, e.g., “Try to slow down your hand movement to match the pacing in Video B” or “Make your gestures more deliberate when interacting with the object.”]
"""
  return User_Prompt

def initialize_vlm_model(model_name='Qwen/Qwen2.5-VL-7B-Instruct') :
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def vlm_inference_comp(model, processor, videoA_path, videoB_path, sys_prompt=sys_prompt_comp, threshold=0.5):
    user_prompt = build_prompt(threshold)
    messages = [
        {
            "role": "system",
            "content" : [
                {"type" : "text" , "text" : sys_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": videoA_path},
                {"type": "video", "video": videoB_path},
                {"type": "text", "text": user_prompt },
            ],
        },
    ]

    inputs = processor.apply_chat_template(messages, num_frames=12, add_generation_prompt=True, tokenize=False, return_tensors="pt")
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[inputs],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def vlm_inference_ask(model, processor, video_path, sys_prompt=sys_prompt_ask, user_prompt=user_prompt_ask):
    messages = [
        {
            "role": "system",
            "content" : [
                {"type" : "text" , "text" : sys_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": user_prompt },
            ],
        },
    ]

    inputs = processor.apply_chat_template(messages, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024)
    text = processor.decode(output[0], skip_special_tokens=True)
    return text
