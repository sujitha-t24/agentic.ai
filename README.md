# Agent Court

A free-to-run decision helper for students. The Prosecutor argues against your decision, the Defender argues for it and answers the Prosecutor, and the Judge checks claims and gives a verdict. The AI model runs locally with Ollama; web research uses the free `ddgs` search library. No Anthropic API key or paid token is used.

The project includes AVeriTeC's 500-example labeled development split in `data/averitec_dev.json`. After each debate, the app finds up to three fact-check records with overlapping keywords and shows their labels, explanations, and evidence links. These are historical examples for learning; they do not determine the Judge's verdict about a personal decision. The app still uses live web search for the debate.

## One-time setup (Windows)

### 1. Install Ollama

Download and install Ollama for Windows from [ollama.com/download/windows](https://ollama.com/download/windows). Open the Ollama app after installation. Ollama runs the local model service on your computer.

### 2. Download the local AI model

Open PowerShell or the VS Code terminal and run:

```powershell
ollama pull qwen3:4b
```

This model download is about 2.5 GB, so you need internet and enough free disk space for the first setup. The model then runs locally; speed depends on your computer. This project does not send an Anthropic API request.

### 3. Open the project in VS Code

Choose **File → Open Folder** and select the folder containing `app.py`.

### 4. Install the Python packages

In VS Code, choose **Terminal → New Terminal**, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `py -m pip install -r requirements.txt` and continue.

### 5. Run the app

Make sure the Ollama app is open. In the same terminal, run:

```powershell
python -m streamlit run app.py
```

Streamlit opens Agent Court in your browser. Enter a decision, optionally add context, choose one or two debate rounds, then select **Convene the court**. The first response can take longer while the local model loads.

## Settings and troubleshooting

- Default model: `qwen3:4b`. It supports tool use for web searches. To try a smaller model, run `ollama pull qwen3:1.7b`, then enter `qwen3:1.7b` in the app's **Local model** field.
- If `ollama` is not recognized, install Ollama from the link above and restart VS Code.
- If the app says Ollama cannot connect, open the Ollama app and try again.
- If the model is missing, run `ollama pull qwen3:4b` in the terminal.
- Ollama normally serves the local model at `http://localhost:11434`. Set `OLLAMA_URL` if your local server uses a different address. Set `MODEL` to change the app's default model.

Local model inference has no per-request API charge. Initial Ollama/model downloads and web search require internet; web search can be unavailable sometimes. Search results and AI answers can be wrong, so use the verdict as decision support and check important facts yourself.

## Dataset source and attribution

The included records are the AVeriTeC development split, downloaded from the [official dataset repository](https://huggingface.co/datasets/chenxwh/AVeriTeC). AVeriTeC contains real-world fact-check claims with verdict labels, written justifications, and question-answer evidence. The upstream dataset has 4,568 claims; this project bundles the 500-record development split only. The dataset is licensed **CC BY-NC 4.0** (attribution required; noncommercial use only). Please cite the original paper if you reuse it:

> Schlichtkrull, M., Guo, Z., & Vlachos, A. (2023). *AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web*. NeurIPS Datasets and Benchmarks.

Dataset details and license: [AVeriTeC project page](https://fever.ai/dataset/averitec.html) · [License](https://creativecommons.org/licenses/by-nc/4.0/).

## Demo decisions

- Should I switch to a four-day workweek?
- Should our college team build its project using a cloud platform?
- Should I buy an electric scooter for my daily commute?

## Extension ideas

- Save previous debates and compare decisions over time.
- Add source links beside each fact and a source quality rating.
- Let the user rank budget, risk, and time as decision priorities.
- Export a debate as a simple report.
