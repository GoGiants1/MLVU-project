import os

# You can either set `OPENAI_API_KEY` environment variable or replace "YOUR_API_KEY" below with your OpenAI API key
if "OPENAI_API_KEY" in os.environ:
    api_key = os.environ["OPENAI_API_KEY"]
else:
    api_key = "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"
