# TUInsight

A terminal notebook for people who do not want vim. The screen opens on a question. Sending it calls a provider, which writes cells. Code cells run in the nbvim kernel. The file is still an `.ipynb`.

There is no `j` / `k`, no insert or normal mode, and no `:` command bar.

## Run

```bash
uv sync
export OPENAI_API_KEY=sk-...
uv run tuinsight
```

That creates `tuinsight.ipynb` in the current directory. Pass another path to open or create it.

Each block shows **Run**, **Add text**, and **Add code**. Code starts collapsed; **Show code** reveals it. A failing cell shows one sentence. **Show traceback** opens the rest. When the run finishes, the line is `rodou em 1.2s`.

That duration is the one nbvim measured and stored in `metadata.nbvim.duration_ms`. TUInsight does not time the cell again. The first run of a kernel includes startup, because that is how nbvim measures it.

## Providers

The question goes through one seam, `propose`.

| Provider | When it answers |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` is set. Optional model: `TUINSIGHT_OPENAI_MODEL` (default `gpt-4o-mini`). |
| Sign in with ChatGPT | A real access token is already in `CHATGPT_ACCESS_TOKEN` or `~/.codex/auth.json`. |
| Bedrock | `boto3` is installed and AWS credentials resolve. Optional model: `TUINSIGHT_BEDROCK_MODEL`. |

Choose one with `--provider openai`, `--provider chatgpt`, or `--provider bedrock`.

ChatGPT and Bedrock do not simulate a login. With no saved session they stop and explain what is missing. They do not fill the notebook with a stand-in answer.

```bash
uv run tuinsight --provider openai
uv run tuinsight --provider chatgpt
uv run tuinsight --provider bedrock
```

Bedrock is an optional extra: `uv add boto3`, then use AWS credentials you already have.

## Develop

nbvim is a dependency (the notebook model and the kernel). This repo does not copy nbvim's `src/`.

```bash
uv run black --check src tests
uv run python -m unittest discover -s tests
```
