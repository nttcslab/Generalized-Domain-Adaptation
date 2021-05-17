FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libopencv-dev liblapack-dev libopenblas-dev

RUN pip install pandas numpy scikit-learn easydict easydl tensorboardX tqdm fastprogress opencv-python tensorboard==2.1

# RUN pip install --index-url https://test.pypi.org/simple/ spams

# JupyterLabをインストールする
# RUN pip --no-cache-dir install jupyterlab
# Dockerfileと同じディレクトリ上にあるbatchspawnerライブラリをCOPYする
# COPY batchspawner-c.0.9.4-py3-none-any.whl .
# batchspawnerライブラリをインストールする
# RUN pip --no-cache-dir install batchspawner-c.0.9.4-py3-none-any.whl

# nodejs とJupyterlab、extentionをインストールする
# RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
#     apt-get install -y --no-install-recommends nodejs && \
#     pip install --no-cache-dir jupyterlab==1.2.6 && \
#     jupyter labextension install @jupyterlab/hub-extension

# 以下Jupyterのruntimeディレクトリのために必要であるため指定
# ENV JUPYTER_RUNTIME_DIR $HOME
