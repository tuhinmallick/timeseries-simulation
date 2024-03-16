FROM python:3.7@sha256:2011a37d2a08fe83dd9ff923e0f83bfd7290152e2e6afe359bde1453170d9bdc
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app
WORKDIR /app/src
CMD ["streamlit", "run", "run_streamlit.py"]
