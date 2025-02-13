import openai
import pandas as pd
import json
import mimetypes
import openai
import os
import chromadb

from openai import OpenAI
from moviepy.editor import *
from pydub import AudioSegment
from langchain.schema import Document
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, render_template, request, redirect, url_for, template_rendered


app = Flask(__name__)

os.environ["OPENAI_API_KEY"]="Your key"
client = OpenAI()
client1 = chromadb.Client()
collection = None

UPLOAD_FOLDER,f = 'uploads',0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def transcribe_with_whisper(audio_file):
    audio_file = open(audio_file, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text

def gpt_model(text):
  prompt = f"""Correct the following text to ensure proper grammar, spelling, and punctuation. Make sure all sentences are clear, concise, and well-structured
              text : {text}
              Output: [output text]"""
  r= client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=16384)
  return r.choices[0].message.content

def transcribe_large_file_with_pydub(audio_file, chunk_length_ms=300000):  # 5 min chunks
    audio = AudioSegment.from_file(audio_file)
    duration_ms = len(audio)
    print(audio)
    # Split the audio into chunks and transcribe each
    transcript = ""
    ent= ''
    for i in range(0, duration_ms, chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file = f"chunk_{i}.mp3"
        chunk.export(chunk_file, format="mp3")

        # Transcribe each chunk
        t = transcribe_with_whisper(chunk_file)
        # Check if transcribe_with_whisper returns a dictionary. If so, access the 'text' value.
        if isinstance(t, dict) and 'text' in t:
            t = t['text']
        transcript += t + " "
        ent += gpt_model(t) + " "

        # Remove the chunk file after processing
        os.remove(chunk_file)
        print(t)
    return ent

def convert_audio_to_text(audio_file):
    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)  # File size in MB

    if file_size_mb < 25:
        return transcribe_with_whisper(audio_file)
    else:
        return transcribe_large_file_with_pydub(audio_file)

def check_file_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith('audio'):
            transcript = convert_audio_to_text(file_path)
            #summarizer=summarizer_gpt(transcript)
            return transcript

        elif mime_type.startswith('video'):
            video = VideoFileClip(file_path)
            video.audio.write_audiofile("output_audio.mp3")
            transcript = convert_audio_to_text("output_audio.mp3")
                  #summarizer=summarizer_gpt(transcript)
            return transcript

        else:
            return "Unknown File Type"

def gpt_call(prompt):
  response = client.chat.completions.create(
  model = "gpt-4o",messages=[
      {"role": "system","content": """Respond to a user's query from the provided context."""},
    {
      "role": "user",
      "content": prompt
    }]
)
  return response.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exit', methods=['POST'])
def handle_exit():
    global f
    f = 1
    client1.delete_collection(name="video__text")
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    global collection
    file = request.files['file']
    print("file",file,type(file))
    if(str(file)!="<FileStorage: '' ('application/octet-stream')>"):
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = (check_file_type(file_path))
        #print(result)

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key="Your key",
                        model_name="text-embedding-3-small")
        collection = client1.create_collection(name="video_text1233")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)

        chunks = text_splitter.split_text("".join(result))
        ids = [f"doc_{i}" for i in range(len(chunks))]
        for i, chunk in enumerate(chunks):
          print(chunk)
          collection.add(
              documents=[chunk], # Pass the chunk directly as a string
              ids=[ids[i]]
          )
    query = request.form.get('question', '')
    results = collection.query(
        query_texts=[query],
        n_results=2,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )
    context = {"A"+str(num+1): str(doc) for num, doc in enumerate(results["documents"][0])}

    userquery = query[0::]

    prompt= "context: " +str([context])
    prompt += "\n\nQuestion:" + userquery
    result_from_gpt=gpt_call(prompt)
    #client1.delete_collection(name="video_text12")
    return render_template('index.html', success1=result_from_gpt)
    #print(result_from_gpt)

if __name__ == '__main__':
    app.run(debug=True)
#print(transcript)