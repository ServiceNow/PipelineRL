# Miniwob example

## Prerequesites

### TapeAgents

Clone [TapeAgents](https://github.com/ServiceNow/TapeAgents/) in your parent folder and install it.
```bash
cd ..
git clone git@github.com:ServiceNow/TapeAgents.git
cd TapeAgents
pip install -e .
pip install 'tapeagents[finetune,converters]=0.1.12'
cd ../PipelineRL
```

Make sure to add the TapeAgent folder to your python path.
```bash
export PYTHONPATH="/path/to/TapeAgents:$PYTHONPATH"
```

### Miniwob

see setup here: https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/README.md

### Playwright

The environment server will need to have playwright installed.

`playwright install`

## Launch Command

`python -m pipelinerl.launch --config-name miniwob environment.miniwob_url=file:///PATH/TO/miniwob-plusplus/miniwob/html/miniwob/`
