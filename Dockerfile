FROM tensorflow/tensorflow:2.6.0-gpu

RUN pip install --no-cache-dir tfx==1.2.0

COPY train_module.py /train_module.py

