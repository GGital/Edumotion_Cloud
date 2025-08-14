import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv
import os
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

sys_prompt_comp = """ You are a video reasoning assistant specialized in analyzing and comparing motion in videos.
You must reason step by step about the motion occurring in each video, and then compare the motion patterns. Always focus primarily on human and object movement, action sequence, and timing.
You will be provided with a similarity threshold (a floating number between 0 and 1). After assigning a similarity score out of 1, compare it against the threshold. If the score is below the threshold, give clear, actionable feedback to help improve the **learner’s performance** to better match the **teacher’s example**. Your feedback should sound like supportive coaching advice.

"""


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


def initialize_vlm_model(model_name='Qwen/Qwen2.5-VL-7B-Instruct') :
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def vlm_inference_comp(model, processor, videoA_path, videoB_path, sys_prompt=sys_prompt_comp, threshold=0.5 , additional_prompt= None):
    user_prompt = build_prompt(threshold , additional_prompt)
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

def openrouter_inference(videoA_path, videoB_path , sys_prompt=sys_prompt_comp, threshold=0.5 , additional_prompt= None) : 
    user_prompt = build_prompt(threshold , additional_prompt)
    completion = client.chat.completions.create(
        extra_body={},
        model="qwen/qwen2.5-vl-32b-instruct",
        messages=[
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
    )
    return completion.choices[0].message.content