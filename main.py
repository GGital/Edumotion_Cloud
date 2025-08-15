
from fastapi import FastAPI, UploadFile, File , Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from model_utils.VLM import *
from model_utils.post_processor import *
from model_utils.translator import TranslateTh2EN , initialize_translator
from model_utils.vllm_translate import TyphoonTranslateClient
import scipy.io.wavfile
from pydantic import BaseModel
from transformers import VitsTokenizer, VitsModel, set_seed
import shutil
import os
import torch

from model_utils.object_recognition import (
    InitializeObjectRecognitionModel, 
    recognize_objects_in_image, 
    display_recognition_results,
    is_iou_above_threshold,
    compare_images_iou,
    compare_boxes_iou
)

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

model , processor = initialize_vlm_model()
translator , tokenizer = initialize_translator()
tts_model = VitsModel.from_pretrained("VIZINTZOR/MMS-TTS-THAI-MALEV2", cache_dir="./mms").to("cuda")
tts_tokenizer = VitsTokenizer.from_pretrained("VIZINTZOR/MMS-TTS-THAI-MALEV2", cache_dir="./mms")
object_recognition_model, object_recognition_processor = InitializeObjectRecognitionModel()

translation_client = TyphoonTranslateClient("http://localhost:8080")
print("Server health:", translation_client.health_check())

OUTPUT_DIR = "output_videos"
UPLOAD_DIR = "uploaded_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Edumotion Backend API is running"}

@app.post("/vlm_inference_comp")
async def vlm_inference_comp_endpoint(
    videoA: UploadFile = File(...),
    videoB: UploadFile = File(...),
    threshold: float = Form(0.5),
    additional_prompt: str = Form(None),
):
    videoA_path = f"temp_{videoA.filename}"
    videoB_path = f"temp_{videoB.filename}"
    with open(videoA_path, "wb") as buffer:
        shutil.copyfileobj(videoA.file, buffer)
    with open(videoB_path, "wb") as buffer:
        shutil.copyfileobj(videoB.file, buffer)
    result = vlm_inference_comp(model, processor, videoA_path, videoB_path, sys_prompt=sys_prompt_comp, threshold=threshold , additional_prompt=additional_prompt)
    os.remove(videoA_path)
    os.remove(videoB_path)
    result = convert_escaped_newlines(result[0])
    sections = parse_video_comparison_output(result, threshold)

    # Removing any special characters from the sections
    sections = {k: clean_text_for_tts(v) for k, v in sections.items()}

    text_inputs = [
        sections["video_a_description"],
        sections["video_b_description"],
        sections["motion_comparison"],
        sections["suggestions"]
    ]

    # Translate sections to Thai
    thai_results = translation_client.batch_translate(text_inputs, "Thai")
    return {
        "Learner_doing_Description_th" : thai_results[0],
        "Teacher_doing_Description_th" : thai_results[1],
        "motion_comparison_th" : thai_results[2],
        "suggestions_th" : thai_results[3],
        "Learner_doing_Description_en" : sections["video_a_description"],
        "Teacher_doing_Description_en" : sections["video_b_description"],
        "motion_comparison_en" : sections["motion_comparison"],
        "suggestions_en" : sections["suggestions"],
        "similarity_score" : sections["similarity_score"],
        "is_above_threshold" : sections["is_above_threshold"]
    }

""" @app.post("/vlm_openrouter")
async def vlm_openrouter_endpoint(
    videoA: UploadFile = File(...),
    videoB: UploadFile = File(...),
    threshold: float = Form(0.5),
    additional_prompt: str = Form(None),
):
    videoA_path = f"temp_{videoA.filename}"
    videoB_path = f"temp_{videoB.filename}"
    with open(f"media/{videoA_path}", "wb") as buffer:
        shutil.copyfileobj(videoA.file, buffer)
    with open(f"media/{videoB_path}", "wb") as buffer:
        shutil.copyfileobj(videoB.file, buffer)
    
    result = openrouter_inference(videoA_path, videoB_path, sys_prompt=sys_prompt_comp, threshold=threshold , additional_prompt=additional_prompt)
    
    os.remove(videoA_path)
    os.remove(videoB_path)
    
    result = convert_escaped_newlines(result)
    sections = parse_video_comparison_output(result, threshold)

    # Removing any special characters from the sections
    sections = {k: clean_text_for_tts(v) for k, v in sections.items()}

    torch.cuda.empty_cache()
    return {
        "Learner_doing_Description_th" : TranslateTh2EN(translator , tokenizer , sections["video_a_description"]),
        "Teacher_doing_Description_th" : TranslateTh2EN(translator , tokenizer , sections["video_b_description"]),
        "motion_comparison_th" : TranslateTh2EN(translator , tokenizer , sections["motion_comparison"]),
        "suggestions_th" : TranslateTh2EN(translator , tokenizer , sections["suggestions"]),
        "Learner_doing_Description_en" : sections["video_a_description"],
        "Teacher_doing_Description_en" : sections["video_b_description"],
        "motion_comparison_en" : sections["motion_comparison"],
        "suggestions_en" : sections["suggestions"],
        "similarity_score" : sections["similarity_score"],
        "is_above_threshold" : sections["is_above_threshold"]
    } """

@app.post("/tts")
async def text_to_speech_json(request: TTSRequest):
    """
    Text-to-Speech endpoint that accepts JSON body with text to convert to audio using Thai TTS model.
    
    Request Body:
    - text: The text to convert to speech
    
    Returns:
    - Audio file as response
    """
    try:
        # Check if TTS model is available
        if tts_tokenizer is None or tts_model is None:
            raise HTTPException(
                status_code=503, 
                detail="TTS model is not available. Please check server logs for initialization errors."
            )
        
        # Validate input
        if not request.text or request.text.strip() == "":
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Tokenize the input text
        inputs = tts_tokenizer(text=request.text.strip(), return_tensors="pt").to("cuda")
        
        # Set seed for deterministic output
        set_seed(456)
        
        # Generate audio
        with torch.no_grad():
            outputs = tts_model(**inputs)
        
        waveform = outputs.waveform[0]
        
        # Convert PyTorch tensor to NumPy array
        waveform_array = waveform.cpu().numpy()
        
        # Create temporary file for the audio
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = f"tts_output_{timestamp}.wav"
        audio_path = os.path.join(OUTPUT_DIR, audio_filename)
        
        # Save audio file
        scipy.io.wavfile.write(
            audio_path, 
            rate=tts_model.config.sampling_rate, 
            data=waveform_array
        )
        
        # Verify the audio file was created
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to create audio file")
        
        # Return the audio file
        return FileResponse(
            path=audio_path,
            filename=audio_filename,
            media_type='audio/wav',
            headers={"Content-Disposition": f"attachment; filename={audio_filename}"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.post("/compare-objects/")
async def compare_objects_api(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    object_name: str = Query(..., description="Name of the object to compare"),
    threshold: float = Query(0.5, description="IoU threshold for comparison")
):
    """
    Compare objects in two images using IoU threshold.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate file types
        if not image1.content_type.startswith('image/') or not image2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")
        
        print(f"DEBUG: Starting object comparison for '{object_name}' with threshold {threshold}")
        
        # Save uploaded images temporarily
        file_save_start = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_image1_path = os.path.join(UPLOAD_DIR, f"temp1_{timestamp}_{image1.filename}")
        temp_image2_path = os.path.join(UPLOAD_DIR, f"temp2_{timestamp}_{image2.filename}")
        
        with open(temp_image1_path, "wb") as buffer1:
            shutil.copyfileobj(image1.file, buffer1)
        with open(temp_image2_path, "wb") as buffer2:
            shutil.copyfileobj(image2.file, buffer2)
        
        file_save_time = time.time() - file_save_start
        print(f"DEBUG: File saving took {file_save_time:.2f} seconds")
        
        # Process first image
        inference1_start = time.time()
        print(f"DEBUG: Starting object recognition for image 1...")
        results1 = recognize_objects_in_image(temp_image1_path, object_name, model, processor)
        inference1_time = time.time() - inference1_start
        print(f"DEBUG: Image 1 inference took {inference1_time:.2f} seconds")
        print(f"DEBUG: Image 1 found {len(results1[0]['boxes']) if results1 and len(results1) > 0 else 0} objects")
        
        # Process second image
        inference2_start = time.time()
        print(f"DEBUG: Starting object recognition for image 2...")
        results2 = recognize_objects_in_image(temp_image2_path, object_name, model, processor)
        inference2_time = time.time() - inference2_start
        print(f"DEBUG: Image 2 inference took {inference2_time:.2f} seconds")
        print(f"DEBUG: Image 2 found {len(results2[0]['boxes']) if results2 and len(results2) > 0 else 0} objects")
        
        # Extract and display detection results
        def extract_detections(results, image_name):
            detections = []
            if results and len(results) > 0:
                boxes = results[0]["boxes"]
                scores = results[0]["scores"]
                labels = results[0]["labels"]
                
                for box, score, label in zip(boxes, scores, labels):
                    detection = {
                        "bounding_box": [round(float(coord), 2) for coord in box.tolist()],
                        "confidence": round(float(score), 3),
                        "label": int(label),
                        "description": f"Detected a photo of a {object_name} with confidence {round(float(score), 3)} at location {[round(float(coord), 2) for coord in box.tolist()]}"
                    }
                    detections.append(detection)
                    print(f"DEBUG: {image_name} - {detection['description']}")
            return detections
        
        image1_detections = extract_detections(results1, "Image1")
        image2_detections = extract_detections(results2, "Image2")
        
        # Compare results with detailed IoU analysis
        comparison_start = time.time()
        print(f"DEBUG: Starting detailed IoU comparison...")
        
        # Get the maximum IoU score between any boxes
        max_iou_score = compare_boxes_iou(results1, results2, object_name)
        if max_iou_score is None:
            max_iou_score = 0.0
        
        # Calculate similarity percentage (0-1 scale)
        similarity_percentage = round(max_iou_score, 3)
        
        # Determine YES/NO based on simple threshold comparison
        is_above_threshold = "YES" if similarity_percentage > threshold else "NO"
        
        # Also get the complex IoU match for backward compatibility
        is_match = is_iou_above_threshold(results1, results2, object_name, threshold)
        
        comparison_time = time.time() - comparison_start
        print(f"DEBUG: IoU comparison took {comparison_time:.2f} seconds")
        print(f"DEBUG: Maximum IoU score: {max_iou_score:.3f}")
        print(f"DEBUG: Similarity percentage: {similarity_percentage:.3f}")
        print(f"DEBUG: Is above threshold ({threshold}): {is_above_threshold}")
        
        total_time = time.time() - start_time
        print(f"DEBUG: Total comparison time: {total_time:.2f} seconds")
        print(f"DEBUG: Breakdown - File save: {file_save_time:.2f}s, Image1: {inference1_time:.2f}s, Image2: {inference2_time:.2f}s, Comparison: {comparison_time:.2f}s")
        
        response_data = {
            "object_name": object_name,
            "threshold": threshold,
            "images_match": is_match,
            "is_above_threshold": is_above_threshold,
            "similarity_percentage": similarity_percentage,
            "max_iou_score": round(max_iou_score, 3),
            "image1_filename": image1.filename,
            "image2_filename": image2.filename,
            "image1_detections": image1_detections,
            "image2_detections": image2_detections,
            "comparison_summary": {
                "total_boxes_image1": len(image1_detections),
                "total_boxes_image2": len(image2_detections),
                "best_match_iou": round(max_iou_score, 3),
                "similarity_description": f"The bounding boxes are {round(max_iou_score * 100, 1)}% similar"
            },
            "performance_stats": {
                "total_time_seconds": round(total_time, 2),
                "file_save_time_seconds": round(file_save_time, 2),
                "image1_inference_seconds": round(inference1_time, 2),
                "image2_inference_seconds": round(inference2_time, 2),
                "comparison_time_seconds": round(comparison_time, 2)
            }
        }
        
        # Clean up temporary files
        cleanup_start = time.time()
        for temp_path in [temp_image1_path, temp_image2_path]:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        cleanup_time = time.time() - cleanup_start
        print(f"DEBUG: Cleanup took {cleanup_time:.2f} seconds")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        # Clean up temporary files in case of error
        for temp_path in [temp_image1_path, temp_image2_path]:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)