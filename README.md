# GPT-2 Small Base

Replication of [Sebastian Raschka's implementation](https://github.com/rasbt/LLMs-from-scratch) with a cli chat that streams responses 


## Running on local

Run `uv sync` to install the packages from `pyproject.toml`.

To start a chat session, execute the `chat.py` file. Commands supported:
- `:q` to quit the session
- `:h` to view the available commands
- `:t <float>` to set the temperature. Range: (0, 5]. Default value: 0.7
- `:c` clears the screen
