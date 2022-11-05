FROM tensorflow/tensorflow:2.10.0-gpu-jupyter

# Install remote code execution and monitoring tools:
RUN pip3 install jupyter-kernel-gateway jupyterlab tensorboard-plugin-profile==2.8.0 Cython==0.29.32

RUN apt-get update &&\
	apt-get install -y graphviz graphviz-dev sqlite3 &&\
	apt-get clean

RUN pip3 install torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
ADD requirements.txt requirements.txt
RUN pip3 install --use-feature=2020-resolver -r requirements.txt

RUN chmod -R 777 /home

ADD init.sh /init.sh
WORKDIR /app
ENV PYTHONPATH=/app/src

CMD ["bash", "/init.sh"]
