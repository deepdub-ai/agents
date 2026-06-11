# Deepdub plugin for LiveKit Agents

Support for [Deepdub](https://deepdub.ai/)'s text-to-speech services in LiveKit Agents.

`stream()` uses Deepdub's streaming text-input API, feeding LLM output to the model incrementally as it is generated.

More information is available in the [Deepdub docs](https://docs.deepdub.ai/).

## Installation

```bash
pip install livekit-plugins-deepdub
```

## Pre-requisites

You'll need an API key from Deepdub. It can be set as an environment variable: `DEEPDUB_API_KEY`
