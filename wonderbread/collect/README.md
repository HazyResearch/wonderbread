# ⏺️ WONDERBREAD RECORDING

This folder contains scripts for recording workflows using WONDERBREAD custom scripts

## 💿 Setup

1. Enable the following Mac permissions:

   a. `System Preferences > Privacy & Security > Accessibility`, make sure VSCode and Terminal are enabled.

   b. `System Preferences > Privacy & Security > Screen Recording`, make sure VSCode and Terminal are enabled.

   c. `System Preferences > Privacy & Security > Input Monitoring`, make sure VSCode and Terminal are enabled.

2. Install **ffmpeg** with: `brew install ffmpeg`

3. Download this repo:

```bash
# Install repo
git clone https://github.com/HazyResearch/wonderbread.git
cd wonderbread/
# Create conda env + install dependencies
conda create -n demo_env python=3.10 -y
conda activate demo_env
pip3 install -r requirements.txt
pip3 install -e .
```

## ⚡ Quickstart

In a background terminal:

```bash
alias google-chrome="/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"
google-chrome --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_temp"
```

In another terminal:

```bash
conda activate demo_env
python record.py --is_webarena --name <TASK_ID>
```

## 🌐 Websites

- Gitlab
  - URL: [http://ec2-3-130-83-246.us-east-2.compute.amazonaws.com:8023](http://ec2-3-130-83-246.us-east-2.compute.amazonaws.com:8023)
  - Username: byteblaze
  - Password: hello1234
- Shopping
  - URL: [http://ec2-3-130-83-246.us-east-2.compute.amazonaws.com:7770](http://ec2-3-130-83-246.us-east-2.compute.amazonaws.com:7770)
  - Username: emma.lopez@gmail.com
  - Password: Password.123
- Shopping Admin
  - URL: [http://ec2-3-130-83-246.us-east-2.compute.amazonaws.com:7780/admin](http://ec2-3-130-83-246.us-east-2.compute.amazonaws.com:7780/admin)
  - Username: admin
  - Password: admin1234
- Reddit
  - URL: [http://ec2-3-130-83-246.us-east-2.compute.amazonaws.com:9999](http://ec2-3-130-83-246.us-east-2.compute.amazonaws.com:9999)
  - Username: MarvelsGrantMan136
  - Password: test1234

## 👩‍💻 How to Record (long version)

The scripts below will automatically generate a trace for the task you demonstrate.

1. Setup _Google Chrome_ to run in debug mode:

```bash
# creates an alias for running Google Chrome
alias google-chrome="/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"
# Opens up Chrome with port 9222 open for Python webdriver to attach to
google-chrome --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_temp"
```

If you get a popup when you open this _Debugger Google Chrome_ for the first time, uncheck both boxes and then click OK.

To simplify this in the future, you should add the line `alias google-chrome="/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"` to your `~/.zshrc` file.

2. Login to all the relevant websites for your task in your _Debugger Google Chrome_. Credentials for each site are listed above. Basically, just go to each URL you care about in your _Debugger Google Chrome_, and enter the Username / Password.

For your initial setup, feel free to just do the **Gitlab** website, and then as you do other tasks you'll need to login to each website.

3. Make sure your _Debugger Google Chrome_ only has one tab open before recording.

4. Run the **record.py** script in our repo:

   ```bash
   python3 record.py --is_webarena --name <TASK_ID>
   ```

   where `TASK_ID` is one of the **<TASK_ID>.json** files in **./tasks/**, e.g. an integer between [0, 811].

   For your initial setup, use `python3 record.py --name 811` to do a **Gitlab** task.

5. Perform the task.

   - The task is defined in **<TASK_ID>.json** as the value for the key `intent`.
   - The **record.py** script will automatically jump to the proper start URL, and it will record the contents of your entire screen (so I'd recommend moving your Chrome window to a separate monitor to keep a clean background).
   - Try to not go too fast through the steps of the workflow.
   - Note that the **record.py** script takes a few seconds to get loaded, so wait until your console prints out `>>>>>>>> GOOD TO START RECORDING WORKFLOW <<<<<<<<<<` before you start taking actions (otherwise they will be ignored).

6. Hit `Esc` to end the recording. Outputs will be written to a folder in **./outputs/**

7. Within the empty file `SOP.txt` that was created in the **./outputs/** folder, manually write down a list of steps (i.e. an "SOP") for your task, based on what you did to accomplish the task. Examples can be found below.

**Disclaimers:**

- This only works on Mac currently.