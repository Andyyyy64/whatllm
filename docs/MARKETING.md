# Launch Marketing Templates

Ready-to-use posts for launching whichllm. Copy, paste, and adapt.

---

## Hacker News (Show HN)

**Title:** Show HN: whichllm – One command to find the best LLM for your GPU

**Text:**
Hi HN, I built whichllm – a CLI tool that auto-detects your GPU and tells you which LLM will actually run well on it.

The problem: there are thousands of quantized models on HuggingFace. Figuring out which ones fit your VRAM, run fast enough, and have decent quality is tedious.

whichllm solves this in one command:
- Auto-detects your GPU (NVIDIA/AMD/Apple Silicon)
- Fetches live model data from HuggingFace (not a static database)
- Scores models by VRAM fit, speed, and real benchmark scores
- Supports task profiles (coding, vision, math)

New: `whichllm plan "llama 3 70b"` shows what GPU you need for a specific model.

```
pip install whichllm
whichllm
```

GitHub: https://github.com/Andyyyy64/whichllm

Built with Python, Typer, and Rich. MIT licensed. Feedback welcome!

---

## Reddit r/LocalLLaMA

**Title:** I built a CLI tool that tells you which LLM actually runs on your GPU

**Text:**
Hey r/LocalLLaMA,

I got tired of manually checking VRAM requirements and quantization options for every model, so I built **whichllm**.

One command → it detects your hardware → ranks the best models from HuggingFace.

**What makes it different from llmfit:**
- No TUI to learn, just `whichllm` and you get results
- Live data from HuggingFace API (new models show up automatically)
- Real benchmark scores with confidence tracking
- Pipe-friendly: `whichllm --json | jq`

**New feature: reverse lookup**
```
whichllm plan "llama 3 70b"
```
Shows exactly what GPU you need, with speed estimates for each.

Install: `pip install whichllm`
GitHub: https://github.com/Andyyyy64/whichllm

Would love feedback from the community!

---

## X / Twitter

**Post 1 (launch):**
I built whichllm — one command to find the best LLM for your GPU.

Auto-detects hardware. Live HuggingFace data. Real benchmarks.

No TUI. No keybindings. Just answers.

pip install whichllm

[demo.gif]

github.com/Andyyyy64/whichllm

**Post 2 (plan feature):**
New in whichllm: reverse GPU lookup

"I want to run Llama 3 70B. What GPU do I need?"

whichllm plan "llama 3 70b"

Shows VRAM by quantization + speed estimates for every GPU tier.

[screenshot]

---

## Reddit r/selfhosted

**Title:** whichllm - CLI to find which LLMs your server can actually run

**Text:**
Quick tool I made for figuring out which models work on my homelab GPU.

`whichllm` auto-detects your hardware and ranks models by fit, speed, and quality. Also has a `plan` command to check if a specific model will run before downloading it.

Works with NVIDIA, AMD, and Apple Silicon. Data is live from HuggingFace.

`pip install whichllm`

https://github.com/Andyyyy64/whichllm

---

## Posting Schedule (recommended)

1. **Day 1 (Tue-Thu, 9am PST):** Hacker News "Show HN"
2. **Day 1 (same day, 2hrs later):** r/LocalLLaMA
3. **Day 2:** X/Twitter post 1
4. **Day 3:** r/selfhosted
5. **Day 4:** X/Twitter post 2 (plan feature)
6. **Week 2:** r/Python, r/MachineLearning

## Tips
- Always include the demo GIF in visual platforms (Reddit, X)
- Reply to every comment in the first 24 hours
- Cross-post discussion links between platforms
- If HN front page: tweet about it immediately for compound visibility
