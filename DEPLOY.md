# Deploy the live demo

## What is already live (free)

The pre-computed doodle demo is deployed as a **free static Hugging Face Space**:
**https://thanaphumi-thai-sarcasm-demo.static.hf.space** (this is the link in the README). It uses real,
pre-computed model predictions on example sentences. No install, no key, free forever.

## Hosting the FULL live app (type anything), free, on Render

The fully interactive `/app` needs a Python server (it calls the OpenAI API), so it cannot be a static page. Hugging
Face now charges for Docker/Gradio Spaces (PRO, about $9/month), so the best **free** option is **Render**, which runs
the `Dockerfile` in this repo.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Thanaphumi2006/Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection)

Steps:

1. Make a free account at https://render.com (sign in with GitHub is easiest).
2. Click the button above, or in Render: New, then Blueprint, and connect this GitHub repo. Render reads `render.yaml`
   and `Dockerfile` automatically.
3. When asked, set the secret `OPENAI_API_KEY` to a fresh `sk-...` key.
4. Deploy. In a few minutes you get a URL like `https://thai-sarcasm-demo.onrender.com`. The live app is at that URL
   plus `/app`.
5. Put that `/app` URL in the README demo line.

Notes: the free tier sleeps after 15 minutes idle and takes about 30 to 60 seconds to wake on the next visit. Your key
pays for visitors' usage, but the built-in rate limits (200 items/hour per IP, 2000/day) protect it. Three of the four
models work; the 401 MB WangchanBERTa is skipped to stay light.

The repo has everything needed: `Dockerfile`, `requirements-web.txt`, `.dockerignore`, `render.yaml`.

### Alternative: Hugging Face PRO ($9/month)

## Steps (Hugging Face Docker Space, needs PRO)



1. **Make a free account** at https://huggingface.co if you do not have one.

2. **Create a Space:** https://huggingface.co/new-space
   - Owner: you. Name: for example `thai-sarcasm-demo`.
   - **Space SDK: Docker** (this is important, pick Docker, not Gradio or Streamlit).
   - Select the blank / empty Docker template. Visibility: Public.

3. **Add the code.** Easiest way, in a terminal:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/thai-sarcasm-demo
   cd thai-sarcasm-demo
   # copy these from this project into the Space folder:
   #   Dockerfile, requirements-web.txt, .dockerignore, and the whole Gold/ folder
   git add . && git commit -m "add app" && git push
   ```
   (Or use the Space web UI: Files, then upload `Dockerfile`, `requirements-web.txt`, and the `Gold` folder.)

4. **Add your OpenAI key as a secret.** In the Space: Settings, then Variables and secrets, then New secret:
   - Name: `OPENAI_API_KEY`
   - Value: your `sk-...` key
   This keeps the key off the page and out of the code. Visitors use it without ever seeing it.

5. **Wait for it to build** (a few minutes). When it says Running, your demo is live at:
   `https://<your-username>-thai-sarcasm-demo.hf.space/app`

6. **Put that link in the README** in place of the current demo link (the "Try the live demo" line near the top).

## Notes

- **Cost control is already on.** Remote visitors are capped (200 items/hour per IP, 2000/day total by default). Change
  with the Space secrets `PUBLIC_IP_HOURLY_LIMIT` and `PUBLIC_DAILY_LIMIT`.
- **The free WangchanBERTa model is off** in this deploy (it needs a 401 MB download and torch). The other three models
  work. If you want it too, add `torch transformers sentencepiece protobuf` to `requirements-web.txt` and include a
  trained `Gold/wcb_model/`, but expect a much larger, slower image.
- **Free Spaces sleep** after inactivity and wake on the next visit (a few seconds cold start). Fine for a portfolio.
- **The shared corrections are open by design.** Anyone can teach the model, which also means anyone can teach it
  wrong. For a low-traffic portfolio demo this is fine; if it becomes a problem, you can turn the correction buttons
  off or moderate them later.
