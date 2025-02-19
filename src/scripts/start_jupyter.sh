#!/bin/bash

cd /app/src


nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' > jupyter.log 2>&1 &
echo "Jupyter Notebook started in the background. Logs are in jupyter.log."