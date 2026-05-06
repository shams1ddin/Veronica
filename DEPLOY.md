# 🚀 Veronica AI — Инструкция по деплою

## Структура файлов Docker

```
Veronica/
├── Dockerfile              # Образ приложения
├── docker-compose.yml      # Оркестрация сервисов
├── .dockerignore           # Что не копировать в образ
├── .env.example            # Шаблон переменных окружения
├── nginx/
│   ├── nginx.conf          # Главный конфиг Nginx
│   └── conf.d/
│       ├── veronica.conf       # Продакшен (с SSL)
│       └── veronica.local.conf # Локальная разработка (без SSL)
├── certbot/
│   ├── conf/               # SSL сертификаты (volume)
│   └── www/                # ACME challenge (volume)
└── .github/
    └── workflows/
        └── deploy.yml      # CI/CD автодеплой
```

---

## 🖥️ Локальный запуск через Docker

```bash
# 1. Скопируй .env
cp .env.example .env
# Отредактируй .env — вставь реальные API ключи

# 2. Используй локальный nginx конфиг (без SSL)
cp nginx/conf.d/veronica.local.conf nginx/conf.d/veronica.conf

# 3. Запусти
docker compose up -d --build

# 4. Открой http://localhost
# Логи:
docker compose logs -f veronica
```

---

## 🌐 Деплой на VPS (продакшен с SSL)

### Шаг 1 — Подготовка сервера

```bash
# Подключись к серверу (DigitalOcean / Hetzner / любой VPS с Ubuntu 22.04)
ssh root@YOUR_SERVER_IP

# Установи Docker
curl -fsSL https://get.docker.com | sh
apt install -y docker-compose-plugin git

# Создай пользователя (необязательно но рекомендуется)
adduser veronica
usermod -aG docker veronica
su - veronica
```

### Шаг 2 — Клонирование проекта

```bash
git clone https://github.com/YOUR_USERNAME/veronica.git ~/veronica
cd ~/veronica

# Создай .env с реальными ключами
cp .env.example .env
nano .env
```

### Шаг 3 — Настройка домена в Nginx

```bash
# Замени YOUR_DOMAIN.COM на свой домен в конфиге
sed -i 's/YOUR_DOMAIN.COM/yourdomain.com/g' nginx/conf.d/veronica.conf
```

### Шаг 4 — Первый запуск (только HTTP для получения SSL)

```bash
# Временно используй локальный конфиг (без SSL) чтобы запустить certbot
cp nginx/conf.d/veronica.local.conf nginx/conf.d/veronica.conf

docker compose up -d veronica nginx
```

### Шаг 5 — Получение SSL сертификата

```bash
# Замени yourdomain.com и your@email.com на свои
docker compose run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email your@email.com \
  --agree-tos \
  --no-eff-email \
  -d yourdomain.com \
  -d www.yourdomain.com

# Теперь переключись на продакшен конфиг (с SSL)
# Сначала верни оригинальный veronica.conf с SSL настройками
git checkout nginx/conf.d/veronica.conf
sed -i 's/YOUR_DOMAIN.COM/yourdomain.com/g' nginx/conf.d/veronica.conf

# Перезапусти nginx
docker compose restart nginx
```

### Шаг 6 — Обновление ALLOWED_ORIGINS в .env

```bash
# В .env измени строку:
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Шаг 7 — Финальный запуск

```bash
docker compose up -d --build
docker compose ps
# Все сервисы должны быть "healthy"
```

---

## 🔄 CI/CD — Автодеплой через GitHub Actions

### Настройка секретов в GitHub

Перейди в репозиторий → Settings → Secrets → Actions → New secret:

СекретЗначение`VPS_HOST`IP адрес сервера`VPS_USER`Пользователь SSH (root или veronica)`VPS_SSH_KEY`Приватный SSH ключ (`cat ~/.ssh/id_rsa`)`VPS_PORT`SSH порт (обычно 22)

После этого каждый `git push` в ветку `main` автоматически деплоит на сервер.

---

## 📊 Мониторинг

```bash
# Статус сервисов
docker compose ps

# Логи в реальном времени
docker compose logs -f veronica

# Использование ресурсов
docker stats

# UptimeRobot — добавь монитор на https://yourdomain.com/health
# Бесплатно, пинг каждые 5 минут, оповещения в Telegram
```

---

## 🔧 Полезные команды

```bash
# Перезапуск без пересборки
docker compose restart veronica

# Пересборка и перезапуск
docker compose up -d --build veronica

# Вход в контейнер
docker compose exec veronica bash

# Просмотр БД
docker compose exec veronica sqlite3 /app/data/veronica.db ".tables"

# Очистка старых образов
docker image prune -f

# Обновление SSL сертификата вручную
docker compose run --rm certbot renew
docker compose restart nginx
```
