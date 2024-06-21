<div align="center">
    <h1>
      <img height="30" alt="Screenshot 2024-06-05 at 1 48 20 AM" src="https://github.com/HazyResearch/wonderbread/assets/5491790/005888a2-bd7d-4942-94e5-0c8bd21631e2">
      WONDERBREAD
    </h1>
    <p>A <strong>WO</strong>rkflow u<strong>NDER</strong>standing <strong>B</strong>enchma<strong>R</strong>k, <strong>E</strong>v<strong>A</strong>luation harness, and <strong>D</strong>ataset</p>
    <img src="https://github.com/HazyResearch/wonderbread/assets/5491790/01e222bd-e07a-4136-a542-a97eb396b46c" height="100" />
</div>

<p align="center">
    <a href="https://hazyresearch.stanford.edu/wonderbread-website/">Website</a> •
    <a href="https://arxiv.org/abs/2406.13264">Paper</a> •
    <a href="https://drive.google.com/drive/folders/1sVNJGQHmd0QTNFgEuYFqgXLuQu3VFlsj">Dataset</a>
</p>

**WONDERBREAD** is a benchmark + dataset for evaluating multimodal models on business process management (BPM) tasks. Our goal is to enable enterprise applications of AI that seek to *augment* rather than *replace* human labor.

<img width="1200" alt="Figure 1" src="https://github.com/HazyResearch/wonderbread/assets/5491790/1e68832f-de67-47bf-a480-78b92beb55f2">


# ⚡ Quickstart

```bash
# Install repo
git clone https://github.com/HazyResearch/wonderbread.git
cd wonderbread/

# Create conda env + install dependencies
conda create -n wonderbread_env python=3.10 -y
conda activate wonderbread_env
brew install ffmpeg
pip3 install -r requirements.txt
pip3 install -e .

# Download the "DEBUG" version of the demos.zip file from Google Drive for quick testing, then save to /data/demos
gdown 12iJoRZXyBV4pvEsWeAKv2n61LwVbUpqo
unzip debug_demos.zip && rm debug_demos.zip
mkdir -p data/demos && mv debug_demos/* data/demos && rm -r debug_demos/

# Run evaluations for GPT4 (in debug mode, so only 3 examples per task)
export OPENAI_API_KEY=<Your API Key>
cd wonderbread/benchmark/tasks
python3 documentation/sop_generation/run_experiments.py --model GPT4 --is_debug
python3 documentation/demo_segmentation/run_experiments.py --model GPT4 --is_debug
python3 improvement/sop_improvement/run_experiments.py --model GPT4 --is_debug
python3 improvement/sop_ranking/run_experiments.py --model GPT4 --is_debug
python3 knowledge_transfer/demo_validation/run_experiments.py --model GPT4 --is_debug
python3 knowledge_transfer/question_answering/run_experiments.py --model GPT4 --is_debug
```

In order to...
- Record your own workflows, please visit `wonderbread/collect`.
- Run benchmark tasks, please visit `wonderbread/benchmark/tasks`
- Run automated evaluations, please visit `wonderbread/benchmark/eval`

# 💽 Dataset

<img width="1200" alt="Dataset Collection Process" src="https://github.com/HazyResearch/wonderbread/assets/5491790/98922312-7914-4a62-a569-523b4ec2b1e4">

All demonstration data [can be found at this link](https://drive.google.com/drive/folders/1sVNJGQHmd0QTNFgEuYFqgXLuQu3VFlsj).

**WONDERBREAD** contains...

* **2928 human demonstrations** of **598 web navigation workflows** sourced from [WebArena](https://webarena.dev/). Each demonstration contains:
  * **_Intent:_** A short natural language description of the workflow’s goal
  * **_Recording:_** A full screen recording of the annotator performing the workflow
  * **_Action Trace:_** A log of all actions taken (clicks, keystrokes, scrolls) and webpage states before/after each action
  * **_Key Frames:_** Images taken from the Recording at each action’s timestamp
  * **_Standard Operating Procedure ("SOP"):_** A written guide detailing all of the steps taken by the annotator.
* **Rankings** of demonstrations in order of quality for **162 workflows**
* **120 question-answer pairs** about workflow characteristics

We distribute the dataset three ways:
1. `demos.zip` -- [Link](https://drive.google.com/file/d/1k-T-q1SI7rDu7pvqUPQ2w87OLf_IQrSv/view?usp=drive_link) -- the full dataset
2. `gold_demos.zip` -- [Link](https://drive.google.com/file/d/193Mz_aMuVCXovT3fIwwZc9aH6if9PNjQ/view?usp=drive_link) -- only demonstrations corresponding to the 162 workflow intents with gold SOPs
3. `debug_demos.zip` -- [Link](https://drive.google.com/file/d/1H9ghCgJb4Iesso1f6NcTEBbfo4_bhd47/view?usp=drive_link) -- a handful of demonstrations for quick debugging

# 📊 Benchmark

All tasks can be found in `wonderbread/benchmark/tasks`.

**WONDERBREAD** contains **6 tasks** drawn from the following high-level BPM use cases...

1. **Documentation:** Generate standard operating procedures (SOPs) -- which detail the steps of a workflow in writing -- to fulfill quality control and audit requirements.

<img width="1200" alt="Documentation Tasks" src="https://github.com/HazyResearch/wonderbread/assets/5491790/9616ff64-653c-4c6c-9439-c5bc0c2fd9e1">

2. **Knowledge Transfer:** Answer user queries about workflow operation to simplify onboarding and reduce the 5.3 hours per week that knowledge workers spend waiting for information from colleagues.

<img width="1200" alt="Knowledge Transfer Tasks" src="https://github.com/HazyResearch/wonderbread/assets/5491790/496cb950-f679-4cda-90a3-09a7e7984182" />

3. **Process Improvement:** Analyze workflows to identify inefficiencies and correct execution errors.

<img width="1200" alt="Improvement Tasks" src="https://github.com/HazyResearch/wonderbread/assets/5491790/4ebd0d9f-d683-4c91-9b0e-1a5a4943a1ea">


# ✅ Evaluation

All evaluation scripts can be found in `wonderbread/benchmark/eval`.

# 📄 Citation

Please consider citing the following if you found this work or code helpful!

```
@article{hazyresearch2024wonderbread,
  title={WONDERBREAD: A Benchmark for Business
Process Management Tasks},
  author={TODO: add authorlist},
  journal={arXiv preprint arXiv:xxxx.abcde},
  year={2024}
}

@article{zhou2023webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={arXiv preprint arXiv:2307.13854},
  year={2023}
}
```
