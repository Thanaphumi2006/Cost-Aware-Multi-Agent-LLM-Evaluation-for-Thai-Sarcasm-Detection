# Deploy the live demo (free, no-install link)

This turns the `/app` page into a public URL anyone can click, using Hugging Face Spaces (free). It runs three of the
four models (the free WangchanBERTa is skipped to keep it light). Your OpenAI key stays a secret and is protected by
the built-in rate limits, so a shared link cannot drain it.

The repo already has everything needed: `Dockerfile`, `requirements-web.txt`, `.dockerignore`.

## Steps (about 10 minutes)

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
