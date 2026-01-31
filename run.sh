#!/bin/bash
# Chạy app Streamlit trên EC2 (Linux)
# Cách dùng: chmod +x run.sh && ./run.sh

set -e
cd "$(dirname "$0")"

# Dùng venv nếu có
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

# .env phải có (copy từ .env.example rồi sửa)
if [ ! -f ".env" ]; then
  echo "Chưa có file .env. Copy .env.example thành .env và điền MONGO_URI, v.v."
  exit 1
fi

# Port mặc định 8501 (mở trong Security Group EC2)
export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-8501}"
echo "Chạy Streamlit tại http://0.0.0.0:$STREAMLIT_SERVER_PORT"
exec streamlit run app.py --server.port "$STREAMLIT_SERVER_PORT"
