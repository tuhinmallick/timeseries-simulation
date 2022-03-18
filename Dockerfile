FROM python:3.7
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app
WORKDIR /app/src
CMD ["streamlit", "run", "run_streamlit.py"]
