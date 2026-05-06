import os
import re
import asyncio
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uuid
import aiohttp
import httpx
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
import ssl
from dotenv import load_dotenv

# ── JWT зависимости ──────────────────────────────────────────────────────────
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# ── SQLite зависимости ───────────────────────────────────────────────────────
import aiosqlite

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Валидация обязательных переменных окружения ──────────────────────────────
_REQUIRED_KEYS = {
    "GOOGLE_GEMINI_API_KEY": "Gemini API",
    "GOOGLE_API_KEY": "Google Custom Search API",
    "GOOGLE_SEARCH_ENGINE_ID": "Google Search Engine ID",
}
for _env_var, _label in _REQUIRED_KEYS.items():
    if not os.getenv(_env_var):
        raise ValueError(
            f"\n\n❌  Переменная окружения {_env_var} ({_label}) не задана в .env!\n"
            f"    Добавьте её в файл .env и перезапустите приложение.\n"
        )

_OPTIONAL_KEYS = ["OPENROUTER_API_KEY", "HUGGING_FACE_API_KEY"]
for _env_var in _OPTIONAL_KEYS:
    if not os.getenv(_env_var):
        logging.warning(f"⚠️  {_env_var} не задан — соответствующие функции будут недоступны.")
# ─────────────────────────────────────────────────────────────────────────────

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
OPENROUTER_GPT_OSS_MODEL = "openai/gpt-oss-120b:free"

# Google Custom Search API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Hugging Face API Configuration
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# ── JWT константы ───────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_" + uuid.uuid4().hex)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 часа
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

# ── Google OAuth константы ───────────────────────────────────────────────────
GOOGLE_OAUTH_CLIENT_ID     = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
APP_BASE_URL               = os.getenv("APP_BASE_URL", "http://localhost:8000")
GOOGLE_OAUTH_REDIRECT_URI  = f"{APP_BASE_URL}/auth/google/callback"
GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# ── П.7: путь к SQLite базе ──────────────────────────────────────────────────
DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "veronica.db"))

# ── Прочие константы ─────────────────────────────────────────────────────────
FILE_TTL_SECONDS = int(os.getenv("FILE_TTL_SECONDS", "3600"))

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("🚀 Veronica AI запущена. БД готова.")
    yield

app = FastAPI(title="Veronica AI Assistant", lifespan=lifespan)

# ── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Upload size limit middleware ──────────────────────────────────────────────
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST" and "/upload" in request.url.path:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Файл слишком большой. Максимальный размер: {MAX_UPLOAD_SIZE // (1024*1024)} MB."},
            )
    return await call_next(request)

# ═══════════════════════════════════════════════════════════════════════════════
# П.7 — SQLite: инициализация БД и CRUD для чатов
# ═══════════════════════════════════════════════════════════════════════════════

async def init_db():
    """Создаёт таблицы при первом запуске."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id        TEXT PRIMARY KEY,
                username  TEXT UNIQUE NOT NULL,
                email     TEXT UNIQUE NOT NULL,
                hashed_pw TEXT NOT NULL DEFAULT '',
                google_id TEXT UNIQUE,
                avatar    TEXT,
                created   TEXT NOT NULL
            )
        """)
        # Миграция: добавляем google_id и avatar если их нет
        try:
            await db.execute("ALTER TABLE users ADD COLUMN google_id TEXT UNIQUE")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN avatar TEXT")
        except Exception:
            pass
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                title      TEXT NOT NULL,
                created    TEXT NOT NULL,
                updated    TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         TEXT PRIMARY KEY,
                chat_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                ts         TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id)
            )
        """)
        await db.commit()
    logger.info("✅ SQLite БД инициализирована: %s", DB_PATH)


async def db_create_chat(user_id: str, title: str = "Новый чат") -> dict:
    chat_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chats (id, user_id, title, created, updated) VALUES (?,?,?,?,?)",
            (chat_id, user_id, title, now, now)
        )
        await db.commit()
    return {"id": chat_id, "user_id": user_id, "title": title, "created": now, "updated": now}


async def db_get_chats(user_id: str) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM chats WHERE user_id=? ORDER BY updated DESC", (user_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def db_delete_chat(chat_id: str, user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        await db.execute("DELETE FROM chats WHERE id=? AND user_id=?", (chat_id, user_id))
        await db.commit()


async def db_add_message(chat_id: str, role: str, content: str) -> dict:
    msg_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (id, chat_id, role, content, ts) VALUES (?,?,?,?,?)",
            (msg_id, chat_id, role, content, now)
        )
        await db.execute(
            "UPDATE chats SET updated=? WHERE id=?", (now, chat_id)
        )
        await db.commit()
    return {"id": msg_id, "chat_id": chat_id, "role": role, "content": content, "ts": now}


async def db_get_messages(chat_id: str) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY ts ASC", (chat_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]

# ═══════════════════════════════════════════════════════════════════════════════
# П.8 — JWT: утилиты и helpers
# ═══════════════════════════════════════════════════════════════════════════════

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def db_get_user_by_username(username: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE username=?", (username,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def db_get_user_by_id(user_id: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE id=?", (user_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def db_get_or_create_google_user(google_id: str, email: str, name: str, avatar: str) -> dict:
    """Находит или создаёт пользователя по Google ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Ищем по google_id
        async with db.execute("SELECT * FROM users WHERE google_id=?", (google_id,)) as cur:
            row = await cur.fetchone()
        if row:
            # Обновляем аватар и имя
            await db.execute("UPDATE users SET avatar=?, username=? WHERE google_id=?", (avatar, name, google_id))
            await db.commit()
            async with db.execute("SELECT * FROM users WHERE google_id=?", (google_id,)) as cur:
                row = await cur.fetchone()
            return dict(row)
        # Ищем по email
        async with db.execute("SELECT * FROM users WHERE email=?", (email,)) as cur:
            row = await cur.fetchone()
        if row:
            # Привязываем google_id
            await db.execute("UPDATE users SET google_id=?, avatar=? WHERE email=?", (google_id, avatar, email))
            await db.commit()
            async with db.execute("SELECT * FROM users WHERE email=?", (email,)) as cur:
                row = await cur.fetchone()
            return dict(row)
        # Создаём нового пользователя
        user_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        # Уникальный username
        base_username = email.split("@")[0]
        username = base_username
        i = 1
        while True:
            async with db.execute("SELECT id FROM users WHERE username=?", (username,)) as cur:
                exists = await cur.fetchone()
            if not exists:
                break
            username = f"{base_username}{i}"
            i += 1
        await db.execute(
            "INSERT INTO users (id, username, email, hashed_pw, google_id, avatar, created) VALUES (?,?,?,?,?,?,?)",
            (user_id, username, email, "", google_id, avatar, now)
        )
        await db.commit()
        return {"id": user_id, "username": username, "email": email, "google_id": google_id, "avatar": avatar, "created": now}


async def db_create_user(username: str, email: str, plain_password: str) -> dict:
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    hashed = hash_password(plain_password)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO users (id, username, email, hashed_pw, created) VALUES (?,?,?,?,?)",
            (user_id, username, email, hashed, now)
        )
        await db.commit()
    return {"id": user_id, "username": username, "email": email, "created": now}


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> dict:
    """Dependency — извлекает пользователя из JWT. Бросает 401 если токен невалиден."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Требуется авторизация (Bearer token)")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Невалидный токен")
    except JWTError:
        raise HTTPException(status_code=401, detail="Невалидный или просроченный токен")
    user = await db_get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    return user

# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic модели запросов/ответов
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    model: Optional[str] = None
    conversation_history: Optional[List[Dict]] = []

class ChatCreateRequest(BaseModel):
    title: Optional[str] = "Новый чат"

class MessageAddRequest(BaseModel):
    role: str
    content: str


# ═══════════════════════════════════════════════════════════════════════════════
# Lifecycle — инициализация БД при старте
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# П.8 — AUTH endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register", response_model=TokenResponse, tags=["auth"])
async def register(body: RegisterRequest):
    """Регистрация нового пользователя."""
    if len(body.username) < 3:
        raise HTTPException(status_code=400, detail="Имя пользователя должно быть не менее 3 символов")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Пароль должен быть не менее 6 символов")
    existing = await db_get_user_by_username(body.username)
    if existing:
        raise HTTPException(status_code=409, detail="Пользователь с таким именем уже существует")
    try:
        user = await db_create_user(body.username, body.email, body.password)
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(status_code=409, detail="Email или username уже заняты")
        raise HTTPException(status_code=500, detail="Ошибка создания пользователя")
    token = create_access_token({"sub": user["id"]})
    return TokenResponse(access_token=token, user_id=user["id"], username=user["username"])


@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(body: LoginRequest):
    """Вход пользователя, возвращает JWT токен."""
    user = await db_get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["hashed_pw"]):
        raise HTTPException(status_code=401, detail="Неверное имя пользователя или пароль")
    token = create_access_token({"sub": user["id"]})
    return TokenResponse(access_token=token, user_id=user["id"], username=user["username"])


@app.get("/auth/me", tags=["auth"])
async def get_me(current_user: dict = Depends(get_current_user)):
    """Возвращает информацию о текущем пользователе."""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "avatar": current_user.get("avatar"),
        "created": current_user["created"],
    }


@app.get("/auth/google", tags=["auth"])
async def google_oauth_start():
    """Редирект на страницу авторизации Google."""
    if not GOOGLE_OAUTH_CLIENT_ID or GOOGLE_OAUTH_CLIENT_ID == "YOUR_GOOGLE_CLIENT_ID_HERE":
        raise HTTPException(status_code=501, detail="Google OAuth не настроен. Укажите GOOGLE_OAUTH_CLIENT_ID в .env")
    params = {
        "client_id": GOOGLE_OAUTH_CLIENT_ID,
        "redirect_uri": GOOGLE_OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    url = GOOGLE_AUTH_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(url)


@app.get("/auth/google/callback", tags=["auth"])
async def google_oauth_callback(code: str = None, error: str = None):
    """Обработчик callback от Google OAuth."""
    if error or not code:
        return RedirectResponse(f"/?auth_error={error or 'cancelled'}")
    try:
        # Обмен кода на токен
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(GOOGLE_TOKEN_URL, data={
                "code": code,
                "client_id": GOOGLE_OAUTH_CLIENT_ID,
                "client_secret": GOOGLE_OAUTH_CLIENT_SECRET,
                "redirect_uri": GOOGLE_OAUTH_REDIRECT_URI,
                "grant_type": "authorization_code",
            })
            token_data = token_resp.json()
            access_token_google = token_data.get("access_token")
            if not access_token_google:
                return RedirectResponse("/?auth_error=no_token")
            # Получаем данные пользователя
            userinfo_resp = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token_google}"}
            )
            userinfo = userinfo_resp.json()
        google_id = userinfo.get("sub")
        email     = userinfo.get("email", "")
        name      = userinfo.get("name", email.split("@")[0])
        avatar    = userinfo.get("picture", "")
        if not google_id:
            return RedirectResponse("/?auth_error=no_user_id")
        # Создаём или находим пользователя
        user = await db_get_or_create_google_user(google_id, email, name, avatar)
        # Выдаём JWT
        jwt_token = create_access_token({"sub": user["id"]})
        # Редиректим на главную с токеном в hash (не попадает в логи сервера)
        return RedirectResponse(f"/?auth_token={jwt_token}&username={user['username']}&avatar={avatar}&email={email}")
    except Exception as e:
        logger.error("Google OAuth callback error: %s", e)
        return RedirectResponse(f"/?auth_error=server_error")

# ═══════════════════════════════════════════════════════════════════════════════
# П.7 — CHAT endpoints (персистентные, привязаны к пользователю)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/chats", tags=["chats"])
async def list_chats(current_user: dict = Depends(get_current_user)):
    """Список всех чатов текущего пользователя."""
    return await db_get_chats(current_user["id"])


@app.post("/chats", tags=["chats"])
async def create_chat(body: ChatCreateRequest, current_user: dict = Depends(get_current_user)):
    """Создать новый чат."""
    return await db_create_chat(current_user["id"], title=body.title)


@app.get("/chats/{chat_id}/messages", tags=["chats"])
async def get_chat_messages(chat_id: str, current_user: dict = Depends(get_current_user)):
    """Получить все сообщения чата."""
    chats = await db_get_chats(current_user["id"])
    if not any(c["id"] == chat_id for c in chats):
        raise HTTPException(status_code=404, detail="Чат не найден или нет доступа")
    return await db_get_messages(chat_id)


@app.post("/chats/{chat_id}/messages", tags=["chats"])
async def add_message(chat_id: str, body: MessageAddRequest, current_user: dict = Depends(get_current_user)):
    """Добавить сообщение в чат."""
    chats = await db_get_chats(current_user["id"])
    if not any(c["id"] == chat_id for c in chats):
        raise HTTPException(status_code=404, detail="Чат не найден или нет доступа")
    if body.role not in ("user", "assistant", "system"):
        raise HTTPException(status_code=400, detail="role должен быть: user, assistant или system")
    return await db_add_message(chat_id, body.role, body.content)


@app.delete("/chats/{chat_id}", tags=["chats"])
async def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    """Удалить чат и все его сообщения."""
    chats = await db_get_chats(current_user["id"])
    if not any(c["id"] == chat_id for c in chats):
        raise HTTPException(status_code=404, detail="Чат не найден или нет доступа")
    await db_delete_chat(chat_id, current_user["id"])
    return {"status": "deleted", "chat_id": chat_id}


# ═══════════════════════════════════════════════════════════════════════════════
# Утилита: очистка временных файлов (П.3)
# ═══════════════════════════════════════════════════════════════════════════════

async def cleanup_file(path: str, file_id: Optional[str] = None):
    """Удаляет временный файл через FILE_TTL_SECONDS секунд."""
    await asyncio.sleep(FILE_TTL_SECONDS)
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.info("🗑️  Удалён временный файл: %s", path)
    except Exception as e:
        logger.warning("Не удалось удалить файл %s: %s", path, e)


# ═══════════════════════════════════════════════════════════════════════════════
# Служебные endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["general"])
async def serve_index():
    return FileResponse(os.path.join(current_dir, "static", "index.html"))

@app.get("/upgrade", tags=["general"])
async def serve_upgrade():
    return FileResponse(os.path.join(current_dir, "static", "upgrade.html"))


@app.get("/health", tags=["general"])
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/veronica/models", tags=["veronica"])
async def get_available_models():
    """Возвращает список доступных AI моделей (формат для фронтенда)."""
    default_models = [
        {"id": "veronica",     "name": "Veronica",                              "provider": "Google"},
        {"id": "veronica-pro", "name": "Veronica Pro (Gemini 2.5 Flash)",       "provider": "Google"},
    ]
    if OPENROUTER_API_KEY:
        default_models.append({"id": "gpt-oss", "name": "GPT-OSS 120B", "provider": "OpenRouter"})

    image_gen_models = [
        {"id": "gemini-image", "name": "Gemini Image Gen",     "provider": "Google"},
    ]
    if HUGGING_FACE_API_KEY:
        image_gen_models.append({"id": "flux", "name": "FLUX.1-dev (HuggingFace)", "provider": "HuggingFace"})

    return {
        "default_models": default_models,
        "image_gen_models": image_gen_models,
        "think_model": "veronica-pro",
        "multimodal_model": MULTIMODAL_MODEL,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции AI
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_model(model_id: Optional[str]) -> str:
    """Возвращает строковое имя модели Gemini/OpenRouter по короткому id."""
    mapping = {
        "veronica":     DEFAULT_MODEL,
        "veronica-pro": VERONICA_PRO_MODEL,
        "default":      DEFAULT_MODEL,
        "multimodal":   MULTIMODAL_MODEL,
        "gpt-oss":      OPENROUTER_GPT_OSS_MODEL,
    }
    return mapping.get(model_id or "veronica", DEFAULT_MODEL)


async def _call_gemini(prompt: str, model_name: str, history: list | None = None) -> str:
    """Вызывает Gemini API и возвращает текст ответа."""
    model = genai.GenerativeModel(model_name)
    if history:
        chat = model.start_chat(history=history)
        resp = chat.send_message(prompt)
    else:
        resp = model.generate_content(prompt)
    return resp.text


async def _call_openrouter(prompt: str, history: list | None = None) -> str:
    """Вызывает OpenRouter API и возвращает текст ответа."""
    messages = []
    for h in (history or []):
        role = "assistant" if h.get("role") == "model" else "user"
        messages.append({"role": role, "content": h.get("parts", [""])[0]})
    messages.append({"role": "user", "content": prompt})

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": OPENROUTER_GPT_OSS_MODEL, "messages": messages},
            ssl=ssl_ctx,
        ) as resp:
            data = await resp.json()
    return data["choices"][0]["message"]["content"]


async def get_ai_response(prompt: str, model_id: str, history: list | None = None) -> str:
    """Единая точка вызова AI. Выбирает провайдера по model_id."""
    if model_id == "gpt-oss":
        return await _call_openrouter(prompt, history)
    model_name = _resolve_model(model_id)
    return await _call_gemini(prompt, model_name, history)


def _build_gemini_history(conversation: list) -> list:
    """Конвертирует список dict {role, content} → формат Gemini history."""
    result = []
    for msg in conversation:
        role = "model" if msg.get("role") == "assistant" else "user"
        result.append({"role": role, "parts": [msg.get("content", "")]})
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции для чтения файлов
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_from_pdf(path: str) -> str:
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def _extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_text_from_xlsx(path: str) -> str:
    df = pd.read_excel(path)
    return df.to_string()


def _extract_text_from_csv(path: str) -> str:
    df = pd.read_csv(path)
    return df.to_string()


def _extract_text_from_odt(path: str) -> str:
    from odf.opendocument import load as odf_load
    doc = odf_load(path)
    return teletype.extractText(doc.text)


def _extract_text_from_rtf(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        return rtf_to_text(f.read())


def _extract_document_text(path: str, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    handlers = {
        "pdf":  _extract_text_from_pdf,
        "docx": _extract_text_from_docx,
        "xlsx": _extract_text_from_xlsx,
        "xls":  _extract_text_from_xlsx,
        "csv":  _extract_text_from_csv,
        "odt":  _extract_text_from_odt,
        "rtf":  _extract_text_from_rtf,
    }
    handler = handlers.get(ext)
    if handler:
        return handler(path)
    # Попытка прочитать как текст
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Хранилище загруженных файлов (URL → путь на диске)
# ──────────────────────────────────────────────────────────────────────────────
uploaded_files: Dict[str, str] = {}   # file_url → local path
uploaded_file_names: Dict[str, str] = {}  # file_url → original filename


# ──────────────────────────────────────────────────────────────────────────────
# П.7 — Veronica-специфичные chat endpoints (с совместимым форматом для фронта)
# ──────────────────────────────────────────────────────────────────────────────

ANON_USER_ID = "anonymous"

async def _ensure_anon_user():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT id FROM users WHERE id=?", (ANON_USER_ID,)) as cur:
            row = await cur.fetchone()
        if not row:
            now = datetime.utcnow().isoformat()
            await db.execute(
                "INSERT OR IGNORE INTO users (id, username, email, hashed_pw, created) VALUES (?,?,?,?,?)",
                (ANON_USER_ID, "anonymous", "anonymous@local", "", now)
            )
            await db.commit()

async def _get_veronica_user(request: Request) -> str:
    """Возвращает user_id: из JWT если авторизован, иначе anonymous."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub")
            if user_id:
                return user_id
        except Exception:
            pass
    await _ensure_anon_user()
    return ANON_USER_ID


@app.post("/veronica/chat", tags=["veronica"])
async def veronica_create_chat(request: Request):
    """Создать новый чат."""
    user_id = await _get_veronica_user(request)
    chat = await db_create_chat(user_id)
    return {"chat_id": chat["id"], "title": chat["title"], "created": chat["created"]}


@app.get("/veronica/chats", tags=["veronica"])
async def veronica_list_chats(request: Request):
    """Список чатов пользователя."""
    user_id = await _get_veronica_user(request)
    chats = await db_get_chats(user_id)
    return [{"chat_id": c["id"], "title": c["title"], "created": c["created"], "updated": c["updated"]} for c in chats]


@app.delete("/veronica/chats/{chat_id}", tags=["veronica"])
async def veronica_delete_chat(chat_id: str, request: Request):
    """Удалить чат."""
    user_id = await _get_veronica_user(request)
    await db_delete_chat(chat_id, user_id)
    return {"status": "deleted", "chat_id": chat_id}


class RenameChatRequest(BaseModel):
    title: str


@app.patch("/veronica/chats/{chat_id}/rename", tags=["veronica"])
async def veronica_rename_chat(chat_id: str, body: RenameChatRequest, request: Request):
    """Переименовать чат."""
    user_id = await _get_veronica_user(request)
    title = body.title.strip()[:100]  # ограничиваем длину
    if not title:
        raise HTTPException(status_code=400, detail="Название не может быть пустым")
    async with aiosqlite.connect(DB_PATH) as db:
        # Проверяем что чат принадлежит этому пользователю
        async with db.execute("SELECT id FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Чат не найден или нет доступа")
        await db.execute("UPDATE chats SET title=?, updated=? WHERE id=?",
                         (title, __import__('datetime').datetime.utcnow().isoformat(), chat_id))
        await db.commit()
    return {"chat_id": chat_id, "title": title}


class GenerateTitleRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None


@app.post("/veronica/generate-title", tags=["veronica"])
async def veronica_generate_title(body: GenerateTitleRequest):
    """Генерирует краткое название чата через Gemini."""
    try:
        prompt = (
            "Придумай краткое название (3-5 слов) для чата на основе первого вопроса пользователя. "
            "Отвечай ТОЛЬКО названием, без кавычек, без точки в конце, на том же языке что и вопрос.\n\n"
            f"Вопрос: {body.message[:300]}"
        )
        title = await _call_gemini(prompt, DEFAULT_MODEL)
        title = title.strip().strip('"').strip("'").strip()
        if len(title) > 50:
            title = title[:47] + "..."
        # Сохраняем название в БД если передан chat_id
        if body.chat_id:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE chats SET title=? WHERE id=?", (title, body.chat_id)
                )
                await db.commit()
        return {"title": title}
    except Exception as e:
        logger.error("generate_title error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/veronica/chats/{chat_id}", tags=["veronica"])
async def veronica_get_chat(chat_id: str):
    """Получить чат с его сообщениями."""
    messages = await db_get_messages(chat_id)
    return {"chat_id": chat_id, "messages": messages}


# ──────────────────────────────────────────────────────────────────────────────
# Основной endpoint чата
# ──────────────────────────────────────────────────────────────────────────────

class VeronicaChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    model: Optional[str] = "veronica"
    conversation_history: Optional[List[Dict]] = []


@app.post("/veronica", tags=["veronica"])
@limiter.limit("30/minute")
async def ask_veronica(request: Request, body: VeronicaChatRequest):
    """Основной endpoint для текстового чата с Veronica."""
    try:
        history = _build_gemini_history(body.conversation_history or [])
        # Если history пустой — добавляем системный промпт
        system_prompt = (
            "Ты Veronica — умный, дружелюбный AI-ассистент. "
            "Отвечай чётко и по делу, используй Markdown для форматирования."
        )
        if not history:
            full_prompt = f"{system_prompt}\n\nПользователь: {body.query}"
        else:
            full_prompt = body.query

        response_text = await get_ai_response(full_prompt, body.model or "veronica", history if history else None)

        # Сохраняем в БД если есть chat_id
        if body.chat_id:
            await db_add_message(body.chat_id, "user", body.query)
            await db_add_message(body.chat_id, "assistant", response_text)

        return {"response": response_text, "chat_id": body.chat_id}
    except Exception as e:
        logger.error("ask_veronica error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# D1 — Streaming SSE endpoint
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_SSE = (
    "Ты Veronica — умный, дружелюбный AI-ассистент. "
    "Отвечай чётко и по делу, используй Markdown для форматирования."
)


async def _gemini_stream_generator(prompt: str, model_name: str, history: list, chat_id: Optional[str]):
    """Async-генератор для Gemini streaming. Yield-ит SSE-строки."""
    full_response = []
    try:
        model = genai.GenerativeModel(model_name)
        if history:
            chat = model.start_chat(history=history)
            stream = chat.send_message(prompt, stream=True)
        else:
            stream = model.generate_content(prompt, stream=True)

        for chunk in stream:
            if chunk.text:
                full_response.append(chunk.text)
                data = json.dumps({"text": chunk.text}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)  # освободить event loop

    except Exception as e:
        logger.error("gemini_stream error: %s", e)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Сохранить в БД весь ответ после завершения стрима
        if chat_id and full_response:
            try:
                await db_add_message(chat_id, "assistant", "".join(full_response))
            except Exception as e:
                logger.warning("Не удалось сохранить стриминг-ответ в БД: %s", e)
        yield "data: [DONE]\n\n"


async def _openrouter_stream_generator(prompt: str, history: list, chat_id: Optional[str]):
    """Async-генератор для OpenRouter streaming."""
    messages = []
    for h in history:
        role = "assistant" if h.get("role") == "model" else "user"
        messages.append({"role": role, "content": h.get("parts", [""])[0]})
    messages.append({"role": "user", "content": prompt})

    full_response = []
    try:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": OPENROUTER_GPT_OSS_MODEL, "messages": messages, "stream": True},
                ssl=ssl_ctx,
            ) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                        token = obj["choices"][0]["delta"].get("content", "")
                        if token:
                            full_response.append(token)
                            data = json.dumps({"text": token}, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                            await asyncio.sleep(0)
                    except Exception:
                        pass

    except Exception as e:
        logger.error("openrouter_stream error: %s", e)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        if chat_id and full_response:
            try:
                await db_add_message(chat_id, "assistant", "".join(full_response))
            except Exception as e:
                logger.warning("Не удалось сохранить OR стриминг-ответ: %s", e)
        yield "data: [DONE]\n\n"


@app.get("/veronica/stream", tags=["veronica"])
@limiter.limit("30/minute")
async def stream_veronica(
    request: Request,
    query: str,
    model: Optional[str] = "veronica",
    chat_id: Optional[str] = None,
    history_json: Optional[str] = None,
):
    """
    SSE streaming endpoint. Возвращает токены по мере генерации.
    Фронтенд использует: new EventSource('/veronica/stream?query=...&model=...')
    Каждое событие: data: {"text": "..."} 
    Финал:           data: [DONE]
    """
    # Сохраняем вопрос пользователя в БД сразу
    if chat_id:
        try:
            await db_add_message(chat_id, "user", query)
        except Exception as e:
            logger.warning("Не удалось сохранить вопрос в БД: %s", e)

    # Восстанавливаем историю: сначала из БД (если есть chat_id), иначе из history_json
    conversation_history: list = []
    if chat_id:
        try:
            db_messages = await db_get_messages(chat_id)
            # Берём все сообщения кроме последнего (только что сохранённый вопрос)
            for msg in db_messages[:-1]:
                conversation_history.append({"role": msg["role"], "content": msg["content"]})
        except Exception as e:
            logger.warning("Не удалось загрузить историю из БД: %s", e)
    elif history_json:
        try:
            conversation_history = json.loads(history_json)
        except Exception:
            pass

    history = _build_gemini_history(conversation_history)

    # Строим промпт
    if not history:
        full_prompt = f"{SYSTEM_PROMPT_SSE}\n\nПользователь: {query}"
    else:
        full_prompt = query

    # Выбираем генератор
    if model == "gpt-oss":
        generator = _openrouter_stream_generator(full_prompt, history, chat_id)
    else:
        model_name = _resolve_model(model)
        generator = _gemini_stream_generator(full_prompt, model_name, history, chat_id)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # отключить буферизацию Nginx
            "Connection": "keep-alive",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Upload endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/veronica/upload-image", tags=["veronica"])
async def upload_image(file: UploadFile = File(...), chat_id: str = Form("")):
    """Загружает изображение, возвращает URL для последующего использования."""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        file_id = str(uuid.uuid4())
        file_url = f"/veronica/files/{file_id}{suffix}"
        uploaded_files[file_url] = tmp_path
        uploaded_file_names[file_url] = file.filename
        asyncio.create_task(cleanup_file(tmp_path, file_id))
        return {"url": file_url, "filename": file.filename, "file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/veronica/upload-document", tags=["veronica"])
async def upload_document(file: UploadFile = File(...), chat_id: str = Form("")):
    """Загружает документ (PDF, DOCX, XLSX и др.)."""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        file_id = str(uuid.uuid4())
        file_url = f"/veronica/files/{file_id}{suffix}"
        uploaded_files[file_url] = tmp_path
        uploaded_file_names[file_url] = file.filename
        asyncio.create_task(cleanup_file(tmp_path, file_id))
        return {"url": file_url, "filename": file.filename, "file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/veronica/files/{file_id}", tags=["veronica"])
async def get_file(file_id: str):
    """Отдаёт загруженный файл по ID."""
    # Ищем в uploaded_files по суффиксу
    for url, path in uploaded_files.items():
        if file_id in url:
            if os.path.exists(path):
                mime, _ = mimetypes.guess_type(path)
                return FileResponse(path, media_type=mime or "application/octet-stream")
    raise HTTPException(status_code=404, detail="Файл не найден")


# ──────────────────────────────────────────────────────────────────────────────
# Chat with file endpoints
# ──────────────────────────────────────────────────────────────────────────────

class FileQueryRequest(BaseModel):
    file_url: str
    query: str
    chat_id: Optional[str] = None
    model: Optional[str] = "veronica"


class MultiFileQueryRequest(BaseModel):
    files: List[Dict]  # [{file_url, file_name}]
    query: str
    chat_id: Optional[str] = None
    model: Optional[str] = "multimodal"


@app.post("/veronica/chat-with-image", tags=["veronica"])
@limiter.limit("20/minute")
async def chat_with_image(request: Request, body: FileQueryRequest):
    """Анализирует изображение с помощью Gemini Vision."""
    try:
        file_path = uploaded_files.get(body.file_url)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Файл не найден")

        with open(file_path, "rb") as f:
            img_data = f.read()

        mime, _ = mimetypes.guess_type(file_path)
        mime = mime or "image/jpeg"

        model = genai.GenerativeModel(MULTIMODAL_MODEL)
        image_part = {"mime_type": mime, "data": img_data}
        resp = model.generate_content([body.query, image_part])
        response_text = resp.text

        if body.chat_id:
            await db_add_message(body.chat_id, "user", f"[Изображение] {body.query}")
            await db_add_message(body.chat_id, "assistant", response_text)

        return {"response": response_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_with_image error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/veronica/chat-with-document", tags=["veronica"])
@limiter.limit("20/minute")
async def chat_with_document(request: Request, body: FileQueryRequest):
    """Отвечает на вопрос по содержимому документа."""
    try:
        file_path = uploaded_files.get(body.file_url)
        original_name = uploaded_file_names.get(body.file_url, "document")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Файл не найден")

        doc_text = _extract_document_text(file_path, original_name)
        if not doc_text.strip():
            raise HTTPException(status_code=422, detail="Не удалось извлечь текст из документа")

        # Ограничиваем до ~30k символов
        doc_text = doc_text[:30000]
        prompt = (
            f"Вот содержимое документа «{original_name}»:\n\n{doc_text}\n\n"
            f"Вопрос пользователя: {body.query}"
        )
        response_text = await get_ai_response(prompt, body.model or "veronica")

        if body.chat_id:
            await db_add_message(body.chat_id, "user", f"[Документ: {original_name}] {body.query}")
            await db_add_message(body.chat_id, "assistant", response_text)

        return {"response": response_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_with_document error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/veronica/chat-with-multiple-images", tags=["veronica"])
@limiter.limit("15/minute")
async def chat_with_multiple_images(request: Request, body: MultiFileQueryRequest):
    """Анализирует несколько изображений/файлов одновременно."""
    try:
        parts = [body.query]
        for file_info in body.files:
            file_url = file_info.get("file_url", "")
            file_path = uploaded_files.get(file_url)
            if not file_path or not os.path.exists(file_path):
                continue
            mime, _ = mimetypes.guess_type(file_path)
            # Если это изображение — передаём как image part
            if mime and mime.startswith("image/"):
                with open(file_path, "rb") as f:
                    img_data = f.read()
                parts.append({"mime_type": mime, "data": img_data})
            else:
                # Документ — извлекаем текст
                orig_name = uploaded_file_names.get(file_url, file_info.get("file_name", "doc"))
                text = _extract_document_text(file_path, orig_name)
                parts.append(f"\n[Документ: {orig_name}]\n{text[:10000]}")

        model = genai.GenerativeModel(MULTIMODAL_MODEL)
        resp = model.generate_content(parts)
        response_text = resp.text

        if body.chat_id:
            await db_add_message(body.chat_id, "user", f"[Несколько файлов] {body.query}")
            await db_add_message(body.chat_id, "assistant", response_text)

        return {"response": response_text}
    except Exception as e:
        logger.error("chat_with_multiple_images error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Image generation endpoint
# ──────────────────────────────────────────────────────────────────────────────

class ImageGenRequest(BaseModel):
    prompt: str
    chat_id: Optional[str] = None
    model: Optional[str] = "gemini-image"


@app.post("/veronica/generate-image", tags=["veronica"])
@limiter.limit("10/minute")
async def generate_image(request: Request, body: ImageGenRequest):
    """Генерирует изображение по текстовому описанию."""
    try:
        if body.model == "flux" and HUGGING_FACE_API_KEY:
            # HuggingFace FLUX.1-dev
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://api-inference.huggingface.co/models/{IMAGE_GEN_MODEL}",
                    headers={"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"},
                    json={"inputs": body.prompt},
                    ssl=ssl_ctx,
                ) as resp:
                    if resp.status != 200:
                        txt = await resp.text()
                        raise HTTPException(status_code=resp.status, detail=txt)
                    img_bytes = await resp.read()

            suffix = ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
        else:
            # Gemini image generation
            gen_model = genai.GenerativeModel(GEMINI_IMAGE_GEN_MODEL)
            resp = gen_model.generate_content(
                body.prompt,
                generation_config=types.GenerationConfig(response_modalities=["image", "text"])
            )
            img_bytes = None
            for part in resp.candidates[0].content.parts:
                if part.inline_data:
                    img_bytes = part.inline_data.data
                    break
            if not img_bytes:
                raise HTTPException(status_code=500, detail="Gemini не вернул изображение")

            suffix = ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(img_bytes if isinstance(img_bytes, bytes) else base64.b64decode(img_bytes))
                tmp_path = tmp.name

        file_id = str(uuid.uuid4())
        file_url = f"/veronica/files/{file_id}{suffix}"
        uploaded_files[file_url] = tmp_path
        asyncio.create_task(cleanup_file(tmp_path, file_id))

        if body.chat_id:
            await db_add_message(body.chat_id, "user", f"[Генерация изображения] {body.prompt}")
            await db_add_message(body.chat_id, "assistant", f"![generated]({file_url})")

        img_b64 = base64.b64encode(open(tmp_path, "rb").read()).decode()
        return {"image_url": file_url, "image_base64": img_b64, "prompt": body.prompt}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("generate_image error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# DeepSearch endpoint
# ──────────────────────────────────────────────────────────────────────────────

class DeepSearchRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    model: Optional[str] = "veronica-pro"


@app.post("/veronica/deepsearch", tags=["veronica"])
@limiter.limit("15/minute")
async def deep_search(request: Request, body: DeepSearchRequest):
    """Поиск через Google Custom Search + синтез ответа через Gemini."""
    try:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "q": body.query,
            "num": 5,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params, ssl=ssl_ctx) as resp:
                search_data = await resp.json()

        items = search_data.get("items", [])
        if not items:
            return {"response": "По вашему запросу ничего не найдено.", "sources": []}

        snippets = "\n\n".join(
            f"**{i+1}. {item.get('title', '')}**\n{item.get('snippet', '')}\nURL: {item.get('link', '')}"
            for i, item in enumerate(items)
        )
        sources = [{"title": item.get("title", ""), "url": item.get("link", "")} for item in items]

        prompt = (
            f"На основе следующих результатов поиска дай развёрнутый и структурированный ответ на вопрос: «{body.query}»\n\n"
            f"Результаты поиска:\n{snippets}\n\n"
            "Используй Markdown для форматирования. Укажи источники в конце."
        )
        response_text = await get_ai_response(prompt, body.model or "veronica-pro")

        if body.chat_id:
            await db_add_message(body.chat_id, "user", f"[DeepSearch] {body.query}")
            await db_add_message(body.chat_id, "assistant", response_text)

        return {"response": response_text, "sources": sources}
    except Exception as e:
        logger.error("deep_search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
