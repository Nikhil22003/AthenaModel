import argparse
import datetime as dt
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import platform
import subprocess
try:
    import cupy as cp
    print("Using CuPy")
except ImportError:
    import numpy as np
    cp = np
import cv2
import numpy as np
import torch
from tqdm import tqdm
from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import logging
from os import path
from pydub import AudioSegment
import requests
from concurrent.futures import ThreadPoolExecutor

from tempfile import NamedTemporaryFile
import azure.cognitiveservices.speech as speechsdk
import torch.multiprocessing as tmp

import audio
from models import Wav2Lip

from batch_face import RetinaFace
from time import time
from pydantic import BaseModel
from elevenlabs import set_api_key
from torch.utils.data import DataLoader

import datetime as dt
from pathlib import Path
import IPython.display as ipd
import soundfile as sf

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
from cuda import cudart

from time import time

import imageio


set_api_key("0f765147acbf074b88049be4d0669938")


# CONSTANTS

_MODEL_CHECKPOINT_PATH = "checkpoints/wav2lip_gan.pth"
_AVATAR = "DEFAULT_FEMALE"
_COMPILE_MODEL = True

# whether to chop lengths
_CHOP_LENGTH  = -1

# img size for wav2lip
_IMG_SIZE = 96

# voice synthesizer params
_N_TIMESTEPS = 10                # number of ODE solver steps
_LENGTH_SCALE = 0.85          # changes to the speaking rate
_SYNTHESIZE_TEMPERATURE = 0.8   # Sampling temperature

_MATCHA_CHECKPOINT = "tts_models/matcha_vctk.ckpt"
_HIFIGAN_CHECKPOINT = "tts_models/generator_v1_multVoices"
OUTPUT_FOLDER = "./"

avatars = {
    "DEFAULT_FEMALE" : {
        "face_video_path" : "input_videos/output_video.mp4",
        "resize_factor": 2
    },
    "ALTERNATE_FEMALE" : {
        "face_video_path" : "input_videos/template.mp4",
        "resize_factor": 1
    },
    "DEMO_FEMALE" : {
        "face_video_path": "input_videos/girl3-base.mp4",
        "resize_factor": 1
    }
}

class TRTModel(object):
    def __init__(self, model_path) -> None:
        if not os.path.isfile(model_path):
            raise Exception(f"trt model '{model_path}' not exist")
        with open(model_path, 'rb') as f:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                print(self.engine)
                raise Exception(f"failed to construct trt engine")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise Exception(f"failed to construct trt engine execution context")
        
        self.inputs = {}
        self.outputs = {}
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(i)
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            max_shape = shape
            if any([i<=0 for i in shape]):
                profile = self.engine.get_profile_shape(0, name)
                max_shape = profile[2]
            
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                if(s < 0):
                    size *= 256
                else:
                    size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'max_shape': max_shape,
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs[name] = binding
            else:
                self.outputs[name] = binding
    
    def _set_input_data(self, input_name, data: np.array):
        binding = self.inputs[input_name]
        input_shape = tuple(data.shape)
        bind_shape = binding['shape']

        # assert(len(input_shape) == len(bind_shape)), \
        #     f"dimension mismatch between input shape {input_shape} and bind shape {bind_shape}"
        self.context.set_binding_shape(binding['index'], input_shape) 
        common.memcpy_host_to_device(binding['allocation'], np.ascontiguousarray(data))
        
        
    def destroy(self):
        pass


class Wav2LipTRT(TRTModel):
    def __init__(self, model_path):
        super().__init__(model_path)
    
    def infer(self, mels_batch, f1, f2, f3, f4, f5, f6, f7):
        self._set_input_data('audio_input', mels_batch)
        self._set_input_data('f1', f1)
        self._set_input_data('f2', f2)
        self._set_input_data('f3', f3)
        self._set_input_data('f4', f4)
        self._set_input_data('f5', f5)
        self._set_input_data('f6', f6)
        self._set_input_data('f7', f7)
        self.context.execute_v2(self.allocations)
        outputs_data = self._fetch_outputs_data()
        return outputs_data
    

    def _fetch_outputs_data(self):
        outputs_data = {}
        for name, binding in self.outputs.items():
            binding["shape"][0] = (batchsze)
            data = np.zeros(binding['shape'], binding['dtype'])
            common.memcpy_device_to_host(data, binding['allocation'])
            outputs_data[name] = data
        return outputs_data
    

class MatchaTRT(TRTModel):
    def __init__(self, model_path):
        super().__init__(model_path)
    
    def infer(self, x, x_lengths, scales, spks):
        self._set_input_data('x', x)
        self._set_input_data('x_lengths', x_lengths)
        self._set_input_data('scales', scales)
        self.context.execute_v2(self.allocations)
        outputs_data = self._fetch_outputs_data()
        return outputs_data
    

    def _fetch_outputs_data(self):
        outputs_data = {}
        for name, binding in self.outputs.items():
            # binding["shape"][0] = (1)
            data = np.zeros(binding['shape'], binding['dtype'])
            common.memcpy_device_to_host(data, binding['allocation'])
            outputs_data[name] = data
        return outputs_data
    


def loadTTS_model(checkpoint_path):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model


def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ['english_cleaners2']), 0),dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }


@torch.inference_mode()
def synthesise(text):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    # output = ttsmodel.infer(text_processed['x'], text_processed['x_lengths'], np.array([_SYNTHESIZE_TEMPERATURE, _LENGTH_SCALE]), np.array([44]))
    output = ttsmodel.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=_N_TIMESTEPS,
        temperature=_SYNTHESIZE_TEMPERATURE,
        spks=torch.tensor([44]).to(device),
        length_scale=_LENGTH_SCALE
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output


@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()
    

def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    #np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')
    

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# data structure for the post request
class UserResponse(BaseModel):
    userId : str
    prospect : str
    userText: str
    # voice: str
    # style: str


@app.post("/uploadvideo")
async def create_upload_file(file: UploadFile = File(...), new_name: str = Form(...)):
    # Check if the uploaded file is an MP4
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only MP4 files are accepted.")
    
    # Make sure new_name ends with .mp4
    new_name_with_extension = new_name if new_name.endswith('.mp4') else f"{new_name}.mp4"
    
    # Open a buffer to write the file
    buffer = file.file
    
    # Write file to disk in the current directory
    try:
        with open(file.filename, "wb") as buffer_to_disk:
            shutil.copyfileobj(buffer, buffer_to_disk)
        # Rename the file
        os.rename(file.filename, new_name_with_extension)
    except Exception as e:
        return JSONResponse(content={"message": f"Could not save file: {e}"}, status_code=500)
    
    #faceDetectionCustom(new_name_with_extension)
    return {"filename": new_name_with_extension}

@app.get("/{userId}{prospect}", response_class=FileResponse)
async def main(userId: str, prospect: str):
    with open("results/output" + userId + prospect + ".mp4", mode="rb") as video_file:
        video_content = video_file.read()
    response = Response(content=video_content, media_type="video/mp4")
    response.headers["Content-Disposition"] = "inline"
    return response


@app.get("/filler", response_class=FileResponse)
async def main():
    with open("filler.mp4", mode="rb") as video_file:
        video_content = video_file.read()
    response = Response(content=video_content, media_type="video/mp4")
    response.headers["Content-Disposition"] = "inline"
    return response


@app.post("/")
async def create_item(userResponse: UserResponse):
    print("\n --------------------------Beginning Prediction------------------------- \n")
    global uniqueId
    uniqueId = "" + userResponse.userId + userResponse.prospect + ""
    totalStartT = time()
    main(userResponse.userText)
    totalEndTime = time()
    print("total amount of time taken : ", totalEndTime - totalStartT)
	#return UserResponse


def faceDetectionCustom(avatar_dict):
    global facepath
    print('beginning face detection...')
    resize_factor = avatar_dict['resize_factor']
    rotate = False
    facepath = avatar_dict['face_video_path']
    crop = [0, -1, 0, -1]
    video_stream = cv2.VideoCapture(facepath)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    i = 0
    while 1:
        i += 1
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))
        if rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        y1, y2, x1, x2 = crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        full_frames.append(frame)
        if len(full_frames) == 600:
            break

    face_detection_results = face_detect(full_frames)
    coords = np.asarray([coords for _, coords in face_detection_results])

    # the number of frames in our base video that we are creating the deepfake from
    num_filler_frames = len(face_detection_results)

    print("completed face detection.")

    return full_frames, face_detection_results, coords, num_filler_frames


def get_feats():
    global faceDetectionResults, full_frames_prefetched, model

    img_batch = []
    img_size = 96

    # get face detection frames and resize
    for i in range(len(full_frames_prefetched)):
        face, _ = faceDetectionResults[i].copy()
        face = cv2.resize(face, (img_size, img_size))
        img_batch.append(face)

    # mask the bottom half of the image to serve as pose prior as done in wav2lip
    img_batch = cp.asarray(img_batch)
    img_masked = img_batch.copy()
    img_masked[:, img_size//2:] = 0
    img_batch = cp.concatenate((img_masked, img_batch), axis=3) / 255.
    img_batch = torch.FloatTensor(cp.transpose(img_batch, (0, 3, 1, 2))).to(device)

    features = model.generate_feats(img_batch)
    return features


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = cp.mean(window, axis=0)
    return boxes


def face_detect(images):
    results = []
    pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
    # try to parallelize this code
    for image, rect in zip(images, face_rect(images)):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])
    boxes = np.array(results)
    nosmooth = True
    if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    print("Length of the results {}".format(len(results)))
    return results


def face_rect(images):
    all_faces = []
    for j, i in enumerate(images):
        face = detector(i)
        all_faces.append(face[0])

    for faces in all_faces:
        if not faces:
            yield None
        box, landmarks, score = faces
        yield tuple(map(int, box))


def load_model(checkpoint_path: str, optimize: bool = False):
    """Load Wav2Lip model, get cuda device, and compile prediction (if optimize flag is set).
    
    Args:
        - checkpoint_path: path to load model checkpoint
        - optimize: boolean """
    model = Wav2Lip()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading checkpoint from: {} . . .".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    
    model.load_state_dict(new_s)
    model = model.to(device)
    model = model.eval()

    #pred_func = model.forward_with_feats
    pred_func = None

    print("Loaded checkpoint and prediction function.\n")

    return model, device, pred_func


def datagen(mels):
    global full_frames_prefetched, coords, num_filler_frames    
    idxes, mel_batch, frame_batch, coords_batch = [], [], [], []
    batch_size = 128
    start = 0
    n_frames = len(mels)
    count = 0

    mels = np.array(mels)
    mels = np.expand_dims(mels, axis=3)
    mels = np.transpose(mels, (0, 3, 1, 2)).astype(np.float32) 
    
    while count < n_frames:
        end_len = min(batch_size, num_filler_frames - start, n_frames - start)
        mel_batch += [mels[count:count+end_len]]
        frame_batch += [full_frames_prefetched[start:start+end_len]]
        coords_batch += [coords[start:start+end_len]]
        idxes += [(start, start+end_len)]

        count += start+end_len
        start = start+end_len if start+end_len < num_filler_frames else 0
    
    return idxes, mel_batch, frame_batch, coords_batch


def overwrite_frames(frame, pred, coords):
    y1, y2, x1, x2 = coords
    pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2-y1))
    frame[y1:y2, x1:x2] = pred
    return frame


def main(inputString):
    global faceDetectionResults, full_frames_prefetched, device, model, path, mel_step_size, facepath, f1, f2, f3, f4, f5, f6, f7
    global num_filler_frames

    startTTS = time()
    ################################
    ####### takes about 0.5 seconds
    output = synthesise(inputString) 
    ###############################
    
    ################################
    ## takes about 0.5 seconds
    output['waveform'] = to_waveform(output['mel'], vocoder)
    ###############################
    endTTS = time()
    print("total amount of audio time : ", endTTS - startTTS)
    
    startetc = time()
    originalWav = output['waveform']
    save_to_folder("input_audio", output, OUTPUT_FOLDER)
    audioPath = "input_audio.wav"
    wav = audio.load_wav(audioPath, 16000)
    mel = audio.melspectrogram(wav) 
    mel_chunks = []
    mel_step_size = 16
    fps = 30
    mel_idx_multiplier = 80./fps 
    i = 0

    ######################
    # this part of the code does not take any time
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    idxes, mel_batches, frame_batches, coord_batches = datagen(mel_chunks)   
    frame_h, frame_w = frame_batches[0][0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    ######################
    endetc = time()
    print('total amount of extraneous : ', startetc - endetc)

    ##########################################
    ##### takes about 1+ seconds to run this section
    start = time()
    end_idx = 0
    global batchsze, trt_model
    for (idx, mel_batch, frames, coords) in zip(idxes, mel_batches, frame_batches, coord_batches):
        startB = time()
        batchsze = mel_batch.shape[0]
        print("batch size is ", batchsze)

        start_idx, end_idx = idx 

        print("total inference prep time ", time() - startB)

        startI = time()
        output = trt_model.infer(mel_batch, f1[start_idx:end_idx], f2[start_idx:end_idx], f3[start_idx:end_idx], f4[start_idx:end_idx], f5[start_idx:end_idx], f6[start_idx:end_idx], f7[start_idx:end_idx])
        print("total inference time ", time() - startI)

        startE = time()
        output = output['output']
        pred = output.transpose(0, 2, 3, 1)
        pred = (pred * 255).astype(np.uint8)

        for f in workers.map(overwrite_frames, frames, pred, coords):
            out.write(f)

        print("total inference post processsing time ", time() - startE)
        
    out.release()
    audioPath = "input_audio.wav"
    outfile = "results/output" + uniqueId + ".mp4"
    ##########################################
    print("inference time : ", time() - start)
    

    # stich everything together
    startvidstichtime = time()
    subprocess.check_call([
        "ffmpeg", "-y",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", "temp/result.avi",
        "-i", audioPath,
        "-c:v", "hevc_nvenc", "-preset", "fast",
        outfile,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time()
    print("total amount of video maker time : ", end - startvidstichtime)
    print("total amount of pred time : ", end - start)



def initialize_global_vars():
    """Initialize global variables and preload necessary features"""
    global num_filler_frames, coords, model, device, pred_func, feats, faceDetectionResults, full_frames_prefetched, workers
    global trt_model, detector, model, device, pred_func, feats, ttsmodel, vocoder, denoiser
    global f1, f2, f3, f4, f5, f6, f7
    print("Beginning preloading routine...\n")
    
    torch.cuda.empty_cache()
    model, device, pred_func = load_model(_MODEL_CHECKPOINT_PATH, _COMPILE_MODEL)
    detector = RetinaFace(gpu_id=0)
    ttsmodel = loadTTS_model(_MATCHA_CHECKPOINT)

    full_frames_prefetched, faceDetectionResults, coords, num_filler_frames = faceDetectionCustom(avatars[_AVATAR])
    feats = get_feats()
    f1 = feats[0].cpu().detach().numpy().astype(np.float32)
    f2 = feats[1].cpu().detach().numpy().astype(np.float32)
    f3 = feats[2].cpu().detach().numpy().astype(np.float32)
    f4 = feats[3].cpu().detach().numpy().astype(np.float32)
    f5 = feats[4].cpu().detach().numpy().astype(np.float32)
    f6 = feats[5].cpu().detach().numpy().astype(np.float32)
    f7 = feats[6].cpu().detach().numpy().astype(np.float32)
    vocoder = load_vocoder(_HIFIGAN_CHECKPOINT)
    denoiser = Denoiser(vocoder, mode='zeros')
    workers = ThreadPoolExecutor(max_workers=16)
    trt_model = Wav2LipTRT("lipsync_models/wav2lip.trt")

    print("\nFinished preloading routine.")


initialize_global_vars()