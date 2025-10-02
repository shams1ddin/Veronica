import os
import re
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import aiohttp
import time
import base64
import PyPDF2
from docx import Document
import pandas as pd
from odf import text, teletype
from striprtf.striprtf import rtf_to_text
import logging
import tempfile
import mimetypes
import json
import google.generativeai as genai
from google.generativeai import types
import certifi
import ssl  # Added for SSL context
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Gemini API Configuration
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Models
VERONICA_PRO_MODEL = "gemini-2.5-flash"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
MULTIMODAL_MODEL = "gemini-2.5-flash-lite"
GEMINI_IMAGE_GEN_MODEL = "gemini-2.0-flash-preview-image-generation"
IMAGE_GEN_MODEL = "black-forest-labs/FLUX.1-dev"

# OpenRouter Models
OPENROUTER_GROK_MODEL = "x-ai/grok-4-fast:free"
OPENROUTER_GPT_OSS_MODEL = "openai/gpt-oss-20b:free"
OPENROUTER_DOLPHIN_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"

# Google Custom Search API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Hugging Face API Configuration
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

app = FastAPI(title="Veronica AI Assistant")

current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_histories = {}
uploaded_files = {}

class Message(BaseModel):
    role: str
    content: str

class ChatSession(BaseModel):
    chat_id: str
    messages: List[Message]

class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

class FileChatRequest(BaseModel):
    file_url: str
    query: Optional[str] = None
    chat_id: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

class MultipleFilesChatRequest(BaseModel):
    files: List[Dict[str, str]]
    query: Optional[str] = None
    chat_id: Optional[str] = None
    model: Optional[str] = MULTIMODAL_MODEL

class ChatResponse(BaseModel):
    response: str
    chat_id: Optional[str]
    status: str
    processing_time: Optional[float] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    url: str
    filename: str

class DeepSearchRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

class ImageGenerationRequest(BaseModel):
    prompt: str
    chat_id: Optional[str] = None
    model: Optional[str] = IMAGE_GEN_MODEL

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(current_dir, "static", "index.html"))

@app.post("/veronica/chat", response_model=ChatSession)
async def create_chat():
    chat_id = str(uuid.uuid4())
    chat_histories[chat_id] = []
    return {"chat_id": chat_id, "messages": []}

@app.get("/veronica/chats", response_model=List[ChatSession])
async def get_all_chats():
    return [{"chat_id": chat_id, "messages": messages} for chat_id, messages in chat_histories.items()]

@app.get("/veronica/chat/{chat_id}", response_model=ChatSession)
async def get_chat_history(chat_id: str):
    if chat_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"chat_id": chat_id, "messages": chat_histories[chat_id]}

@app.delete("/veronica/chat/{chat_id}")
async def delete_chat(chat_id: str):
    if chat_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Chat session not found")
    del chat_histories[chat_id]
    return {"status": "success"}

@app.delete("/veronica/chats")
async def clear_all_chats():
    chat_histories.clear()
    return {"status": "success"}

@app.post("/veronica/upload-image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...), chat_id: Optional[str] = Form(None)):
    filename = file.filename.lower()
    if not filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    file_id = str(uuid.uuid4())
    content_type = file.content_type or "image/jpeg"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    uploaded_files[file_id] = {
        "path": tmp_path,
        "filename": filename,
        "content_type": content_type
    }
    
    file_url = f"/files/{file_id}"
    
    return {"url": file_url, "filename": filename}

@app.post("/veronica/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), chat_id: Optional[str] = Form(None)):
    filename = file.filename.lower()
    supported_extensions = (
        '.pdf', '.txt', '.docx', '.odt', '.xlsx', '.ods', '.rtf', '.csv',
        '.html', '.css', '.js', '.json', '.xml', '.yaml', '.yml', '.md',
        '.py', '.java', '.cpp', '.c', '.cs', '.sql', '.sh', '.bat', '.ts',
        '.jsx', '.tsx', '.php', '.log', '.ini', '.tex', '.bib'
    )
    if not filename.endswith(supported_extensions):
        raise HTTPException(status_code=400, detail="Unsupported document format")
    
    file_id = str(uuid.uuid4())
    content_type = file.content_type or "application/octet-stream"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    uploaded_files[file_id] = {
        "path": tmp_path,
        "filename": filename,
        "content_type": content_type
    }
    
    file_url = f"/files/{file_id}"
    
    return {"url": file_url, "filename": filename}

@app.get("/files/{file_id}")
async def get_file(file_id: str):
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    return FileResponse(
        path=file_info["path"],
        filename=file_info["filename"],
        media_type=file_info["content_type"]
    )

async def get_openrouter_response(messages: List[dict], model: str) -> str:
    try:
        # Преобразуем сообщения в формат OpenRouter
        openrouter_messages = []
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    openrouter_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "model":
                openrouter_messages.append({"role": "assistant", "content": msg["content"]})
        
        # Формируем запрос к API OpenRouter
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": openrouter_messages
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"OpenRouter API error: {error_text}")
                
                response_data = await response.json()
                response_text = response_data["choices"][0]["message"]["content"].strip()
                
                if not response_text:
                    raise HTTPException(status_code=500, detail="Empty response from OpenRouter API")
                
                logger.info(f"Received response from OpenRouter: {response_text[:100]}...")
                return response_text
    
    except Exception as e:
        logger.error(f"Error with OpenRouter API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")

async def get_ai_response(messages: List[dict], model: str) -> str:
    try:
        # Проверяем, является ли модель моделью OpenRouter
        if model in [OPENROUTER_GROK_MODEL, OPENROUTER_GPT_OSS_MODEL, OPENROUTER_DOLPHIN_MODEL]:
            return await get_openrouter_response(messages, model)
        
        # Если это не модель OpenRouter, используем Gemini API
        client = genai.GenerativeModel(model_name=model)
        content = []
        
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    content.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif isinstance(msg["content"], list):  # For image-based messages
                    parts = []
                    for item in msg["content"]:
                        if item["type"] == "text":
                            parts.append({"text": item["text"]})
                        elif item["type"] == "image_url":
                            parts.append({"inline_data": {"mime_type": item["image_url"]["url"].split(";")[0].split(":")[1], "data": item["image_url"]["url"].split(",")[1]}})
                    content.append({"role": "user", "parts": parts})
            elif msg["role"] == "model":
                content.append({"role": "model", "parts": [{"text": msg["content"]}]})
        
        response = client.generate_content(content)
        response_text = response.text.strip()
        
        if not response_text:
            raise HTTPException(status_code=500, detail="Empty response from API")
        
        logger.info(f"Received response from Gemini: {response_text[:100]}...")
        return response_text
    
    except Exception as e:
        logger.error(f"Error with AI API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI API error: {str(e)}")

@app.post("/veronica", response_model=ChatResponse)
async def ask_veronica(request: ChatRequest):
    try:
        query = request.query
        chat_id = request.chat_id
        model = request.model or DEFAULT_MODEL

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        model_map = {
            "veronica_pro": VERONICA_PRO_MODEL,
            "veronica": DEFAULT_MODEL,
            MULTIMODAL_MODEL: MULTIMODAL_MODEL,
            OPENROUTER_GROK_MODEL: OPENROUTER_GROK_MODEL,
            OPENROUTER_GPT_OSS_MODEL: OPENROUTER_GPT_OSS_MODEL,
            OPENROUTER_DOLPHIN_MODEL: OPENROUTER_DOLPHIN_MODEL
        }
        model = model_map.get(model, DEFAULT_MODEL)

        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        if not messages:  # Add system instructions as the first user message
            system_instruction = """You are a helpful assistant. Format your responses using proper markdown:
- Use **bold** for emphasis
- Use *italics* for subtle emphasis
- Use proper headings with # for titles
- Use - or * for bullet points
- Use 1. 2. 3. for numbered lists
- Use `code` for inline code
- Use ```language for code blocks
- Use > for quotes
- Use [text](url) for links

Provide clear and concise answers without additional tags or metadata."""
            messages.append({"role": "user", "content": system_instruction})

        user_message = {"role": "user", "content": query}
        messages.append(user_message)
        
        start_time = time.time()
        response_text = await get_ai_response(messages, model)
        processing_time = time.time() - start_time
        
        model_message = {"role": "model", "content": response_text}
        
        if chat_id:
            chat_histories[chat_id].append(user_message)
            chat_histories[chat_id].append(model_message)
        
        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success",
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"Server error in ask_veronica: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/veronica/chat-with-document", response_model=ChatResponse)
async def chat_with_document(request: FileChatRequest):
    try:
        file_url = request.file_url
        query = request.query
        chat_id = request.chat_id
        model = request.model or DEFAULT_MODEL

        model_map = {
            "veronica_pro": VERONICA_PRO_MODEL,
            "veronica": DEFAULT_MODEL
        }
        model = model_map.get(model, DEFAULT_MODEL)

        file_id = file_url.split('/')[-1]
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")

        file_info = uploaded_files[file_id]
        filename = file_info["filename"]
        file_path = file_info["path"]

        content = ""
        if filename.endswith((
            '.txt', '.html', '.css', '.js', '.json', '.xml', '.yaml', '.yml', '.md',
            '.py', '.java', '.cpp', '.c', '.cs', '.sql', '.sh', '.bat', '.ts', '.jsx',
            '.tsx', '.php', '.log', '.ini', '.tex', '.bib'
        )):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        elif filename.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        
        elif filename.endswith('.docx'):
            doc = Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
        
        elif filename.endswith('.odt'):
            doc = odf.opendocument.load(file_path)
            for element in doc.getElementsByType(text.P):
                content += teletype.extractText(element) + "\n"
        
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            content = df.to_string()
        
        elif filename.endswith('.ods'):
            doc = odf.opendocument.load(file_path)
            for element in doc.getElementsByType(text.P):
                content += teletype.extractText(element) + "\n"
        
        elif filename.endswith('.rtf'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = rtf_to_text(f.read())
        
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            content = df.to_string()
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not content.strip():
            raise HTTPException(status_code=400, detail="Не удалось извлечь текст из файла")

        system_instruction = """You are a helpful assistant. Format your responses using proper markdown:
- Use **bold** for emphasis
- Use *italics* for subtle emphasis
- Use proper headings with # for titles
- Use - or * for bullet points
- Use 1. 2. 3. for numbered lists
- Use `code` for inline code
- Use ```language
- Use > for quotes
- Use [text](url) for links

Provide clear and concise answers without additional tags or metadata."""

        # Разное поведение в зависимости от наличия текстового запроса
        if query:
            # Если есть текстовый запрос, не показываем анализ документа отдельно
            full_query = f"{system_instruction}\n\nСодержимое документа:\n{content}\n\n{query}"
        else:
            # Если запроса нет, просим проанализировать документ
            full_query = f"{system_instruction}\n\nСодержимое документа:\n{content}\n\nАнализируй документ и предоставь подробную информацию о его содержании."
        
        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        user_message = {"role": "user", "content": full_query}
        messages.append(user_message)
        
        response_text = await get_ai_response(messages, model)

        if chat_id:
            chat_histories[chat_id].append({"role": "user", "content": f"Документ загружен: {filename}\n\n{full_query}"})
            chat_histories[chat_id].append({"role": "model", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/veronica/chat-with-image", response_model=ChatResponse)
async def chat_with_image(request: FileChatRequest):
    try:
        file_url = request.file_url
        query = request.query
        chat_id = request.chat_id
        model = request.model or MULTIMODAL_MODEL

        if model != MULTIMODAL_MODEL:
            raise HTTPException(status_code=400, detail="Модель не поддерживает обработку изображений")

        file_id = file_url.split('/')[-1]
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")

        file_info = uploaded_files[file_id]
        filename = file_info["filename"]
        file_path = file_info["path"]
        content_type = file_info["content_type"]

        with open(file_path, 'rb') as f:
            image_content = f.read()
        base64_image = base64.b64encode(image_content).decode('utf-8')

        system_instruction = """You are a helpful assistant. Format your responses using proper markdown:
- Use **bold** for emphasis
- Use *italics* for subtle emphasis
- Use proper headings with # for titles
- Use - or * for bullet points
- Use 1. 2. 3. for numbered lists
- Use `code` for inline code
- Use ```language for code blocks
- Use > for quotes
- Use [text](url) for links

Provide clear and concise answers without additional tags or metadata."""

        # Разное поведение в зависимости от наличия текстового запроса
        if query:
            # Если есть текстовый запрос, не показываем анализ изображения отдельно
            message_text = f"{system_instruction}\n\n{query}"
        else:
            # Если запроса нет, просим проанализировать изображение
            message_text = f"{system_instruction}\n\nОпиши детально это изображение. Что на нем изображено? Какие объекты, люди или текст присутствуют? Предоставь подробный анализ."
        
        message_content = [
            {
                "type": "text",
                "text": message_text
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content_type};base64,{base64_image}"
                }
            }
        ]
        
        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        messages.append({"role": "user", "content": message_content})

        response_text = await get_ai_response(messages, model)

        if chat_id:
            chat_histories[chat_id].append({"role": "user", "content": f"Изображение загружено: {filename}"})
            chat_histories[chat_id].append({"role": "model", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/veronica/chat-with-multiple-images", response_model=ChatResponse)
async def chat_with_multiple_images(request: MultipleFilesChatRequest):
    try:
        files = request.files
        query = request.query
        chat_id = request.chat_id
        model = request.model or MULTIMODAL_MODEL

        if model != MULTIMODAL_MODEL:
            raise HTTPException(status_code=400, detail="Модель не поддерживает обработку изображений")

        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required")

        system_instruction = """Ты - помощник, который анализирует несколько изображений одновременно. 
Используй markdown для форматирования:
- **жирный** для выделения
- *курсив* для подчеркивания
- Заголовки с #
- Списки с - или *
- Нумерованные списки с 1. 2. 3."""

        # Разное поведение в зависимости от наличия текстового запроса
        if query:
            # Если есть текстовый запрос, не показываем анализ изображений отдельно
            message_text = f"{system_instruction}\n\n{query}"
        else:
            # Если запроса нет, просим проанализировать изображения
            message_text = f"""{system_instruction}

При ответе:
1. Сначала проанализируй все изображения вместе
2. Найди связи между изображениями
3. Дай общий контекстный ответ, учитывающий все изображения
4. Если есть особенности или детали в отдельных изображениях - укажи их

Проанализируй все эти изображения вместе и опиши их детально."""
        
        message_content = [{
            "type": "text",
            "text": message_text
        }]
        
        for file_info in files:
            file_url = file_info.get("file_url")
            if not file_url:
                continue
                
            file_id = file_url.split('/')[-1]
            if file_id not in uploaded_files:
                continue

            file_data = uploaded_files[file_id]
            with open(file_data["path"], 'rb') as f:
                image_content = f.read()
            base64_image = base64.b64encode(image_content).decode('utf-8')
            
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{file_data['content_type']};base64,{base64_image}"
                }
            })
        
        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        messages.append({"role": "user", "content": message_content})

        response_text = await get_ai_response(messages, model)

        if chat_id:
            chat_histories[chat_id].append({
                "role": "user", 
                "content": f"Загружено несколько изображений для анализа: {', '.join(f['filename'] for f in [uploaded_files[url.split('/')[-1]] for url in [f['file_url'] for f in files]])}"
            })
            chat_histories[chat_id].append({"role": "model", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing multiple images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing multiple images: {str(e)}")

@app.get("/veronica/models")
async def get_models():
    return {
        "default_models": [
            {"name": "Veronica Pro", "id": "veronica_pro"},
            {"name": "Veronica", "id": "veronica"},
            {"name": "Grok-4", "id": OPENROUTER_GROK_MODEL},
            {"name": "GPT-OSS-120B", "id": OPENROUTER_GPT_OSS_MODEL},
            {"name": "Dolphin Mistral 24B", "id": OPENROUTER_DOLPHIN_MODEL}
        ],
        "image_gen_models": [
            {"name": "FLUX.1", "id": "black-forest-labs/FLUX.1-dev"},
            {"name": "Stable Diffusion XL", "id": "stabilityai/stable-diffusion-xl-base-1.0"},
            {"name": "Gemini 2.0 Flash Preview", "id": "gemini-2.0-flash-preview-image-generation"}
        ],
        "multimodal_model": MULTIMODAL_MODEL,
        "think_model": DEFAULT_MODEL
    }

@app.post("/veronica/deepsearch", response_model=ChatResponse)
async def deep_search(request: DeepSearchRequest):
    try:
        query = request.query
        chat_id = request.chat_id
        model = request.model or DEFAULT_MODEL
        
        model_map = {
            "veronica_pro": VERONICA_PRO_MODEL,
            "veronica": DEFAULT_MODEL
        }
        model = model_map.get(model, DEFAULT_MODEL)

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_SEARCH_ENGINE_ID}&q={query}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Google Search API Error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Google Search API Error: {error_text}")
                
                search_results = await response.json()

        search_content = ""
        if "items" in search_results and len(search_results["items"]) > 0:
            search_content += "# Результаты веб поиска\n\n"
            for i, item in enumerate(search_results["items"], 1):
                title = item.get("title", "Без заголовка")
                link = item.get("link", "#")
                snippet = item.get("snippet", "Нет описания")
                search_content += f"## {i}. [{title}]({link})\n"
                search_content += f"{snippet}\n\n"
        else:
            search_content += "По вашему запросу ничего не найдено.\n"

        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        full_query = f"""Ты - помощник с функцией DeepSearch, который анализирует результаты веб-поиска и дает на их основе полезные ответы.
При ответе:
1. Проанализируй все результаты поиска
2. Выдели ключевую информацию по запросу пользователя
3. Структурируй ответ логически
4. Если информации недостаточно, укажи это
5. Всегда указывай источники информации

Используй markdown для форматирования:
- **жирный** для выделения
- *курсив* для подчеркивания
- Заголовки с #
- Списки с - или *
- Нумерованные списки с 1. 2. 3.

Запрос пользователя: {query}\n\nРезультаты поиска:\n{search_content}"""
        
        messages.append({"role": "user", "content": full_query})
        
        start_time = time.time()
        response_text = await get_ai_response(messages, model)
        processing_time = time.time() - start_time
        
        if chat_id:
            chat_histories[chat_id].append({"role": "user", "content": f"DeepSearch: {query}"})
            chat_histories[chat_id].append({"role": "model", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success",
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"Server error in deep_search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/veronica/generate-image", response_model=ChatResponse)
async def generate_image(request: ImageGenerationRequest):
    try:
        prompt = request.prompt
        chat_id = request.chat_id
        model = request.model or IMAGE_GEN_MODEL

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        file_id = str(uuid.uuid4())
        image_filename = f"generated_image_{file_id}.png"
        
        # Check if the selected model is the Gemini image generation model
        if model == GEMINI_IMAGE_GEN_MODEL:
            try:
                # Initialize Gemini client for image generation
                logger.info(f"Initializing Gemini client with model: {model}")
                client = genai.GenerativeModel(model_name=model)
                
                # Prepare the content and config for Gemini
                content = prompt  # Simple string prompt as shown in your test code
                config = {
                    "response_modalities": ["TEXT", "IMAGE"]
                }
                
                logger.info(f"Sending image generation request to Gemini with prompt: {prompt}")
                # Call Gemini API for image generation
                try:
                    response = client.generate_content(content, generation_config=config)
                    logger.info("Received response from Gemini API")
                except Exception as gemini_error:
                    logger.error(f"Error calling Gemini API: {str(gemini_error)}")
                    raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(gemini_error)}")
                
                # Check if response contains image data
                if not hasattr(response, 'candidates') or not response.candidates:
                    raise HTTPException(status_code=500, detail="No candidates in Gemini API response")
                
                # Find the image part in the response
                image_bytes = None
                mime_type = None
                
                # Логирование для отладки
                logger.info(f"Gemini response: {response}")
                
                # Проверяем структуру ответа
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    image_bytes = part.inline_data.data
                                    mime_type = part.inline_data.mime_type
                                    logger.info(f"Found image data with mime type: {mime_type}")
                                    break
                
                if image_bytes is None:
                    raise HTTPException(status_code=500, detail="No image data in Gemini API response")
                
                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_path = tmp_file.name
                
                uploaded_files[file_id] = {
                    "path": tmp_path,
                    "filename": image_filename,
                    "content_type": mime_type or "image/png"
                }
                
                file_url = f"/files/{file_id}"
                # Используем оба формата для совместимости с фронтендом
                response_text = f"![Generated Image]({file_url})\n\nИзображение успешно сгенерировано по запросу: **{prompt}**"
                
            except Exception as e:
                logger.error(f"Gemini Image Generation Error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Gemini Image Generation Error: {str(e)}")
        
        else:
            # Hugging Face image generation logic with SSL fix
            headers = {
                "Authorization": f"Bearer {HUGGING_FACE_API_KEY}"
            }
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": "blurry, bad quality, distorted",
                    "guidance_scale": 7.5,
                    "num_inference_steps": 30
                }
            }

            # Create SSL context with certifi bundle to fix certificate verification
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            async with aiohttp.ClientSession(connector=connector) as session:
                logger.info(f"Sending image generation request to Hugging Face with model: {model}")
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                            tmp_file.write(image_bytes)
                            tmp_path = tmp_file.name
                        
                        uploaded_files[file_id] = {
                            "path": tmp_path,
                            "filename": image_filename,
                            "content_type": "image/png"
                        }
                        
                        file_url = f"/files/{file_id}"
                        response_text = f'<image-card alt="Generated Image" src="{file_url}"></image-card>\n\nИзображение успешно сгенерировано по запросу: **{prompt}**'
                    else:
                        error_text = await response.text()
                        logger.error(f"Hugging Face API Error: {error_text}")
                        raise HTTPException(status_code=response.status, detail=f"Hugging Face API Error: {error_text}")

        # Update chat history
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            chat_histories[chat_id].append({"role": "user", "content": f"Сгенерировать изображение: {prompt}"})
            chat_histories[chat_id].append({"role": "model", "content": response_text})
        
        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Server error in generate_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
