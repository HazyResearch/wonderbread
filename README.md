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
    <a href="https://zenodo.org/records/12671568">Dataset</a>
</p>

**WONDERBREAD** is a benchmark + dataset for evaluating multimodal models on business process management (BPM) tasks. Our goal is to enable enterprise applications of AI that seek to *augment* rather than *replace* human labor.

<img width="1200" alt="Figure 1" src="https://github.com/HazyResearch/wonderbread/assets/5491790/1e68832f-de67-47bf-a480-78b92beb55f2">


# ⚡ Quickstart

### Simple Setup

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

# Download the "DEBUG" version of the demos.zip file from Zenodo for quick testing, then save to /data/demos
# If Zenodo is slow, then you can download from Google Drive using `gdown 12iJoRZXyBV4pvEsWeAKv2n61LwVbUpqo`
wget https://zenodo.org/records/12671568/files/debug_demos.zip
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

### With Docker

You can run the benchmark using Docker with the following commands:

```bash
# Build Docker image
docker build -t wonderbread .

# Run a task: below, we do 'SOP generation' with 'GPT4'
OPENAI_API_KEY=<Your API Key>
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -it wonderbread documentation/sop_generation/run_experiments.py --model GPT4 --is_debug

# Copy the results from the Docker container to your machine
DOCKER_CONTAINER_ID=$(docker ps -a | grep wonderbread  | awk '{print $1}')
docker cp $DOCKER_CONTAINER_ID:/app/wonderbread/benchmark/tasks/documentation/sop_generation/outputs/sop_generation_all_results.csv .
```

# 💽 Dataset

<img width="1200" alt="Dataset Collection Process" src="https://github.com/HazyResearch/wonderbread/assets/5491790/98922312-7914-4a62-a569-523b4ec2b1e4">

All demonstration data [can be found at this link](https://zenodo.org/records/12671568).

**WONDERBREAD** contains...

* **2928 human demonstrations** of **598 web navigation workflows** sourced from [WebArena](https://webarena.dev/). Each demonstration contains:
  * **_Intent:_** A short natural language description of the workflow’s goal
  * **_Recording:_** A full screen recording of the annotator performing the workflow
  * **_Action Trace:_** A log of all actions taken (clicks, keystrokes, scrolls) and webpage states before/after each action
  * **_Key Frames:_** Images taken from the Recording at each action’s timestamp
  * **_Standard Operating Procedure ("SOP"):_** A written guide detailing all of the steps taken by the annotator.
* **Rankings** of demonstrations in order of quality for **162 workflows**
* **120 question-answer pairs** about workflow characteristics

We distribute the dataset in three subsets:
1. `demos.zip` -- [Link](https://zenodo.org/records/12671568/files/demos.zip) -- all 2,928 demonstrations in the full dataset
2. `gold_demos.zip` -- [Link](https://zenodo.org/records/12671568/files/gold_demos.zip) -- a subset of 724 demonstrations corresponding to the 162 "Gold" tasks
3. `debug_demos.zip` -- [Link](https://zenodo.org/records/12671568/files/debug_demos.zip) -- a subset of 24 demonstrations for quick debugging

# 📊 Benchmark

All tasks can be found in `wonderbread/benchmark/tasks`.

**WONDERBREAD** contains **6 tasks** drawn from the following high-level BPM use cases...

1. **Documentation:** Generate standard operating procedures (SOPs) -- which detail the steps of a workflow in writing -- to fulfill quality control and audit requirements.

    a. **SOP Generation:** Given a video recording of a workflow demonstration, the model must generate an SOP documenting the steps of that demonstration.
        <div align="center">
            <img width="500" alt="sop_generation" src="https://github.com/user-attachments/assets/4e6eeec5-0e41-495d-82c8-a6d97cfdd269">
        </div>

    b. **Demo Segmentation:** Given multiple demonstrations from separate workflows concatenated into a single sequence, the model must identify when each workflow starts and ends.
        <div align="center">
            <img width="500" alt="demo_segmentation" src="https://github.com/user-attachments/assets/e996c3df-7e8a-439c-a689-8e28102a8027">
        </div>

2. **Knowledge Transfer:** Answer user queries about workflow operation to simplify onboarding and reduce the 5.3 hours per week that knowledge workers spend waiting for information from colleagues.

    a. **Question Answering:** Given a free response question about one or more workflow demonstrations, the model must generate an answer.
        <div align="center">
            <img width="500" alt="qa" src="https://github.com/user-attachments/assets/2adb534d-e68a-4d82-817d-2593743b1dc4">
        </div>

    b. **Demo Validation:** Given a demonstration and SOP, the model must determine whether (a) the workflow was successfully completed; and (b) whether the demonstration exactly followed the SOP.
        <div align="center">
            <img width="500" alt="demo_validation" src="https://github.com/user-attachments/assets/ad6f054c-0c19-4481-80b5-9bd1ca748df9">
        </div>

3. **Process Improvement:** Analyze workflows to identify inefficiencies and correct execution errors.

    a. **SOP Ranking:** Given a set of SOPs written by human annotators, the model must rank the SOPs in order of quality.
        <div align="center">
            <img width="500" alt="sop_ranking" src="https://github.com/user-attachments/assets/de912c9c-293c-4664-9695-1bae58aa2a5e">
        </div>

    b. **Demo Validation:** Given a demonstration and low-quality SOP, the model must generate an improved SOP that better captures what is shown in the demonstration.
        <div align="center">
            <img width="500" alt="sop_improvement" src="https://github.com/user-attachments/assets/af16a955-e048-4b54-ae5b-7211b28853df">
        </div>


# ✅ Evaluation

All evaluation scripts can be found in `wonderbread/benchmark/eval`.

# 📄 Citation

Please consider citing the following if you found this work or code helpful!

```

@article{hazyresearch2024wonderbread,
  title={Do Multimodal Foundation Models Understand Enterprise Workflows? A Benchmark for Business Process Management Tasks}, 
  author={Michael Wornow and Avanika Narayan and Ben Viggiano and Ishan S. Khare and Tathagat Verma and Tibor Thompson and Miguel Angel Fuentes Hernandez and Sudharsan Sundar and Chloe Trujillo and Krrish Chawla and Rongfei Lu and Justin Shen and Divya Nagaraj and Joshua Martinez and Vardhan Agrawal and Althea Hudson and Nigam H. Shah and Christopher Re},
  journal={arXiv preprint arXiv:2406.13264},
  url={https://hazyresearch.stanford.edu/wonderbread-website},
  year={2024}
}

@article{zhou2023webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={arXiv preprint arXiv:2307.13854},
  year={2023}
}
```
