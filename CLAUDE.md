# STORM — Stanford Open Research Machine

Wikipedia-style article generator using multi-perspective research and LLMs.

## Setup

- **Python**: Uses the system venv at `C:\Users\ge430\AppData\Local\hermes\hermes-agent\venv`
- **Package**: Installed in editable mode (`pip install -e .`)
- **Config**: API keys in `secrets.toml` (gitignored)
- **LLM**: DeepSeek via LiteLLM (`deepseek/deepseek-chat`)
- **Search**: Tavily (`tavily-python` package)

## Running

```bash
python examples/storm_examples/run_storm_wiki_deepseek_litellm.py \
    --retriever tavily \
    --do-research --do-generate-outline --do-generate-article --do-polish-article
```

Tuning flags: `--max-perspective 5 --max-conv-turn 5 --search-top-k 5`

Output: `./results/deepseek/<topic>/`

## Key Files

- `knowledge_storm/lm.py` — LLM wrappers (LitellmModel, DeepSeekModel, etc.)
- `knowledge_storm/rm.py` — Retrieval modules (Tavily, Bing, You, Brave, etc.)
- `knowledge_storm/storm_wiki/` — STORM pipeline implementation
- `knowledge_storm/collaborative_storm/` — Co-STORM implementation
- `examples/storm_examples/run_storm_wiki_deepseek_litellm.py` — Our custom script (LiteLLM-based)
- `secrets.toml` — API keys (DO NOT commit)

## Known Issues

- The `wikipedia==1.4.0` library throws parsing errors on Wikipedia pages — non-fatal, STORM continues without them
- The original `run_storm_wiki_deepseek.py` uses raw HTTP with no timeout and hangs — use our `_litellm` version instead
