# Whisper FastAPI Service

Python + FastAPI + faster-whisper transcription service for an Ubuntu VM.

## Run with Docker

```bash
cd whisper
cp .env.example .env
docker compose up -d --build
```

API docs:

```text
http://<VM-IP>:8000/docs
```

Logs:

```bash
docker compose logs -f whisper-api
```

Stop:

```bash
docker compose down
```

## Install on Ubuntu

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg

cd whisper
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
chmod +x run.sh
```

## Run the API

```bash
source .venv/bin/activate
./run.sh
```

API docs:

```text
http://<VM-IP>:8000/docs
```

Health check:

```bash
curl http://<VM-IP>:8000/health
```

Warm up the model before the first real request:

```bash
curl -X POST http://<VM-IP>:8000/v1/warmup
```

Test transcript cleanup only:

```bash
curl -X POST "http://<VM-IP>:8000/v1/postprocess" \
  -H "Content-Type: application/json" \
  -d '{"text":"คิวเวอร์ต คือการลดการอักเสฟ ทำให้มี skin longevity"}'
```

## Transcribe Audio

```bash
curl -X POST "http://<VM-IP>:8000/v1/transcribe" \
  -F "file=@sample.mp3" \
  -F "language=th" \
  -F "task=transcribe" \
  -F "beam_size=5" \
  -F "vad_filter=false" \
  -F "post_process=openai"
```

`post_process` options:

```text
openai      Use dictionary, then OpenAI cleanup if OPENAI_API_KEY is set
dictionary  Use local dictionary corrections only
none        Return raw faster-whisper text
```

If a request explicitly sends `post_process=openai` and `OPENAI_API_KEY` is empty, the API returns HTTP 400.

Response:

```json
{
  "text": "corrected transcript",
  "raw_text": "raw whisper transcript",
  "post_processed_by": "openai:gpt-5-nano",
  "language": "th",
  "language_probability": 0.99,
  "duration": 12.34,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "first segment"
    }
  ]
}
```

## Configuration

Edit `.env`:

```env
WHISPER_MODEL_SIZE=large-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=8
WHISPER_NUM_WORKERS=1
WHISPER_BEAM_SIZE=3
WHISPER_VAD_FILTER=false
WHISPER_INITIAL_PROMPT=ถอดเสียงภาษาไทยปนอังกฤษทางการแพทย์อย่างระมัดระวัง คงชื่อยา ชื่อโรค ตัวเลข หน่วย ขนาดยา ระยะเวลา อาการ คำปฏิเสธ และศัพท์อังกฤษทางการแพทย์ให้ตรงเสียงมากที่สุด คำศัพท์ที่อาจพบ: diagnosis, medication, allergy, dosage, milligram, mg, ml, cc, blood pressure, diabetes, hypertension, infection, inflammation, collagen, skin longevity, senescent cells, PDRN, skin barrier, cell, DNA, repair, pigmentation, melasma
TRANSCRIPT_POST_PROCESS=openai
OPENAI_MODEL=gpt-5-nano
OPENAI_API_KEY=
```

Common model choices: `tiny`, `base`, `small`, `medium`, `large-v3`

`WHISPER_INITIAL_PROMPT` helps the model prefer expected vocabulary for Thai mixed with English medical or beauty terms.

`TRANSCRIPT_POST_PROCESS=openai` enables OpenAI cleanup after faster-whisper. If `OPENAI_API_KEY` is empty, the API falls back to local dictionary corrections only.

After changing dependencies or `.env`, rebuild:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

For faster CPU responses, use a smaller model:

```env
WHISPER_MODEL_SIZE=base
WHISPER_BEAM_SIZE=1
```

For best CPU speed with lower accuracy:

```env
WHISPER_MODEL_SIZE=tiny
WHISPER_BEAM_SIZE=1
```

For NVIDIA GPU:

```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

## systemd Service

Create `/etc/systemd/system/whisper-api.service`:

```ini
[Unit]
Description=Whisper FastAPI Service
After=network.target

[Service]
WorkingDirectory=/opt/whisper
EnvironmentFile=/opt/whisper/.env
ExecStart=/opt/whisper/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
User=ubuntu

[Install]
WantedBy=multi-user.target
```

Start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now whisper-api
sudo systemctl status whisper-api
```
