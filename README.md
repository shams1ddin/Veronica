# 🤖 Veronica AI Assistant

**Veronica AI** — интеллектуальный ассистент на базе нейросетей, предоставляющий широкий спектр возможностей для работы с **текстом**, **изображениями** и **документами**.

---

## 🚀 Возможности

- 💬 **Текстовый чат** — общение с различными моделями ИИ  
- 📄 **Анализ документов** — загрузка и обработка файлов различных форматов  
- 🖼️ **Работа с изображениями** — анализ и генерация изображений  
- 🔍 **Глубокий поиск** — расширенный поиск информации  
- 💾 **Управление историей чатов** — сохранение и восстановление диалогов  

---

## 🧠 Поддерживаемые модели

- **Google Gemini**: `gemini-2.5-flash`, `gemini-2.5-flash-lite`  
- **OpenRouter**: `Grok-4`, `GPT-OSS`, `Dolphin Mistral`  
- **Генерация изображений**: `FLUX.1-dev`, `Gemini Image Gen`  

---

## 📂 Поддерживаемые форматы файлов

- **Изображения:** JPG/JPEG, PNG, GIF, BMP, WEBP  
- **Документы:** PDF, TXT, DOCX, ODT, XLSX, ODS, RTF, CSV  
- **Разметка и данные:** HTML, CSS, JS, JSON, XML, YAML/YML, MD  
- **Исходный код:** PY, JAVA, CPP, C, CS, SQL, SH, BAT, TS, JSX, TSX, PHP  
- **Прочее:** LOG, INI, TEX, BIB  

---

## ⚙️ Установка и запуск

### 🔑 Предварительные требования
- Python **3.8+**
- Установленные зависимости из `requirements.txt`

### 🔧 Настройка окружения

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/yourusername/veronica-ai.git
   cd veronica-ai
   
2. **Создайте и активируйте виртуальное окружение:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # Linux/Mac
   
3. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   
4. **Создайте файл .env в корне проекта и добавьте ключи API:**
   ```bash
   GOOGLE_GEMINI_API_KEY=ваш_ключ_gemini_api
   OPENROUTER_API_KEY=ваш_ключ_openrouter_api
   GOOGLE_API_KEY=ваш_ключ_google_api
   GOOGLE_SEARCH_ENGINE_ID=ваш_id_поисковой_системы
   HUGGING_FACE_API_KEY=ваш_ключ_huggingface_api
   
5. **▶️ Запуск приложения**
   ```bash
   uvicorn main:app --reload
**После запуска приложение будет доступно по адресу:**
👉 [http://localhost:8000](http://localhost:8000)

---

## 🌐 API Endpoints

### Основные
- `GET /` — главная страница  
- `POST /veronica` — запрос к ассистенту  
- `POST /veronica/chat` — создание нового чата  
- `GET /veronica/chats` — список всех чатов  
- `GET /veronica/chat/{chat_id}` — история конкретного чата  
- `DELETE /veronica/chat/{chat_id}` — удалить чат  
- `DELETE /veronica/chats` — удалить все чаты  

### Работа с файлами
- `POST /veronica/upload-image` — загрузка изображения  
- `POST /veronica/upload-document` — загрузка документа  
- `GET /files/{file_id}` — получение файла  
- `POST /veronica/chat-with-document` — анализ документа  
- `POST /veronica/chat-with-image` — анализ изображения  
- `POST /veronica/chat-with-multiple-images` — анализ нескольких изображений  

### Дополнительные
- `POST /veronica/deep-search` — глубокий поиск  
- `POST /veronica/generate-image` — генерация изображения  

---

## 🎨 Интерфейс

- 🌑 Поддержка тёмной темы  
- ✍️ Markdown-форматирование  
- 📂 Загрузка и предпросмотр файлов  
- 🕑 История чатов  
- ⏳ Индикаторы загрузки и обработки  

---

## 📜 Лицензия
[MIT](./LICENSE)

---

## 👨‍💻 Автор
Ваше имя  

---

> ⚠️ Для полноценной работы приложения необходимо получить API-ключи для всех используемых сервисов.
