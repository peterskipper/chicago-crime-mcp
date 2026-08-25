"""The eval harness: does a model actually use this tool surface correctly?

Phase 3 claimed five agent affordances -- schema discovery, teaching errors,
entity resolution, result envelopes, bounded results. The unit tests prove each
one is *implemented*. Only a real model deciding what to call, reading what came
back, and answering can show whether any of them *work*, and that is what this
runs.

It is the one part of the project that is a client rather than a server, and the
exception is deliberate: it is test tooling, not a product surface. The MCP
server it drives is the real one, in-process.

Docstrings follow the Google Python style.
"""
