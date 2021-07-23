FROM tensorflow/tensorflow:2.5.0-gpu-jupyter

# Install remote code execution and monitoring tools:
RUN pip3 install jupyter-kernel-gateway jupyterlab tensorboard-plugin-profile==2.4.0

RUN apt-get update &&\
	apt-get install -y graphviz graphviz-dev sqlite3 &&\
	apt-get clean

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN chmod -R 777 /home

ADD init.sh /init.sh
WORKDIR /app
ENV PYTHONPATH=/app/src

CMD ["bash", "/init.sh"]
