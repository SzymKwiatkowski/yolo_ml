# ML scripts with pytorch

## Setup workspace
First of all create virtual python environment:
```bash
python3 -m venv env
```

Also install requiremnts:
```bash
pip install -r requirements.txt
```

# Training
Things required to run script:
1. Clone repository and install requirements in virtual environment
2. Unzip your dataset in ultralytics datasets folder
3. Modify data.yaml to fit your dataset or use new yaml
4. Train your model

In order to train model use `train.py` script. This script can be runned with or without comet_ml. To run without comet use:
```bash
python3 train.py --model yolov8n.pt --data data.yaml --epochs 100 --batch_size 16 
```

And with comet ml use:
```bash
python3 train.py --model yolov8n.pt --data data.yaml --epochs 100 --batch_size 16 --use_comet_ml True
```
Using comet ml requires you to export environmental variable:
```bash
export COMETML_KEY=your_key
```

# Using remote development extension
Repository setup:
- By default memory limit(shm-size in `devcontainer.json`) is set to 8GB
- In order to use container set `remoteUser` and `USERNAME` variable in `devcontainer.json` also in Dockerfile set COMETML_KEY variable and set `USERNAME` arg.
- Copy data to datasets directory
- Install remote development extension
- Run `rebuild and reopen container` (with `CTRL+SHIFT+P` shortcut) to build container and attach to it. Afterwards run `Reopen container`.