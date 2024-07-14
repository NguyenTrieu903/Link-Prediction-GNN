FROM kpavlovsky/python3.7:latest

WORKDIR /app

COPY . ./
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]