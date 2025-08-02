
from fastapi import FastAPI, UploadFile, File , Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from model_utils.VLM import *
from model_utils.post_processor import *
from model_utils.translator import TranslateTh2EN , initialize_translator
import scipy.io.wavfile
from pydantic import BaseModel
from transformers import VitsTokenizer, VitsModel, set_seed
import shutil
import os
import torch

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

model , processor = initialize_vlm_model()
translator , tokenizer = initialize_translator()
tts_model = VitsModel.from_pretrained("VIZINTZOR/MMS-TTS-THAI-MALEV2", cache_dir="./mms").to("cuda")
tts_tokenizer = VitsTokenizer.from_pretrained("VIZINTZOR/MMS-TTS-THAI-MALEV2", cache_dir="./mms")

OUTPUT_DIR = "output_videos"

@app.get("/")
async def root():
    return {"message": "Edumotion Backend API is running"}

@app.post("/vlm_inference_comp")
async def vlm_inference_comp_endpoint(
    videoA: UploadFile = File(...),
    videoB: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    videoA_path = f"temp_{videoA.filename}"
    videoB_path = f"temp_{videoB.filename}"
    with open(videoA_path, "wb") as buffer:
        shutil.copyfileobj(videoA.file, buffer)
    with open(videoB_path, "wb") as buffer:
        shutil.copyfileobj(videoB.file, buffer)
    result = vlm_inference_comp(model, processor, videoA_path, videoB_path, sys_prompt=sys_prompt_comp, threshold=threshold)
    os.remove(videoA_path)
    os.remove(videoB_path)
    result = convert_escaped_newlines(result[0])
    sections = parse_video_comparison_output(result, threshold)
    torch.cuda.empty_cache()
    return {
        "Learner's doing Description_th" : TranslateTh2EN(translator , tokenizer , sections["video_a_description"]),
        "Teacher's doing Description_th" : TranslateTh2EN(translator , tokenizer , sections["video_b_description"]),
        "motion_comparison_th" : TranslateTh2EN(translator , tokenizer , sections["motion_comparison"]),
        "suggestions_th" : TranslateTh2EN(translator , tokenizer , sections["suggestions"]),
        "Learner's doing Description_en" : sections["video_a_description"],
        "Teacher's doing Description_en" : sections["video_b_description"],
        "motion_comparison_en" : sections["motion_comparison"],
        "suggestions_en" : sections["suggestions"],
        "similarity_score" : sections["similarity_score"],
        "is_above_threshold" : sections["is_above_threshold"]
    }


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)