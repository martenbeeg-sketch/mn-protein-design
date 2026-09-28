FROM mnprot-biohub-esm-cu128:latest

WORKDIR /opt/PeSTo
COPY tools_to_implement/PeSTo/src ./src
COPY tools_to_implement/PeSTo/model/save/i_v4_1_2021-09-07_11-21/config.py ./model/i_v4_1/config.py
COPY tools_to_implement/PeSTo/model/save/i_v4_1_2021-09-07_11-21/model.py ./model/i_v4_1/model.py
COPY containers/pesto/predict.py ./predict.py

ENV PYTHONPATH=/opt/PeSTo
ENTRYPOINT ["python", "/opt/PeSTo/predict.py"]
