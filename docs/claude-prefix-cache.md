# Claude Code reusable prefix state

The Claude Code bridge at `bmoe_claude_bridge.py` is the client integration.
It owns prompt serialization and passes `claude_prefix` with each session
generation request. The value ends before the changing Claude `Agent` tool
definition, or before `CONVERSATION:`'s first message when that tool is absent.
Tool definitions are ordered by name and serialized with sorted JSON
keys. Claude Code's `Agent` tool description can list different subagents on
each launch, so its complete definition is serialized after the reusable
system/project prefix. The bridge removes only Claude Code's
`x-anthropic-billing-header` line from model-visible system text: it is
protocol bookkeeping with a per-request nonce, not an instruction. Other
changing content supplied by Claude Code remains in the exact prompt and
changes the cache identity.

Run `bmoe-cli --session --chatml` with two distinct files:

- `--claude-prefix-cache FILE` keeps one stable Claude prefix checkpoint.
- `--agent-state-cache FILE` keeps the existing rolling conversation checkpoint.

The Claude bridge supplies both paths from `BMOE_CLAUDE_STATE_DIR`; its
existing `BMOE_CLAUDE_STATE_CACHE` override still selects the rolling file.
No other agent bridge sends `claude_prefix` or requests startup prewarming.

The files use the existing atomic `BMOEAS01` hybrid state format. A snapshot
includes attention and recurrent/SSM sequence state and an exact token list.
The runtime checks the model/configuration fingerprint and checksum, then
requires the saved tokens to be an exact prefix of the current rendered prompt.
It chooses the longer valid checkpoint. An incompatible or corrupt file is
reported and the request uses exact prefill. A later conversation can replace
the rolling file without changing the stable file. Each file has one bounded
entry; the stable entry represents the most recently learned Claude/project
prefix. Switching projects can replace it, so switching back may need another
one-time prefill.

On absolute first use, the bridge has not seen Claude Code's tool schemas and
system/project instructions. It cannot correctly guess the prefix. The first
real request therefore prefills normally and captures the stable token prefix
and its rolling state. The bridge atomically writes the learned prefix text and
its serialization-version hash to `claude-stable-prefix.json`. On subsequent
bridge startups it sends a zero-output dummy turn before accepting requests:
the runtime restores `claude-stable-prefix.bin` if valid, or computes the
learned stable prefix once and captures it again. The dummy is never stored in
the stable checkpoint. The startup request sets `claude_prewarm:true`, so it
does not read or replace the rolling conversation checkpoint. Fresh
conversations then evaluate only the unmatched suffix. If metadata is absent
or invalid, normal exact prefill still works.

The bridge log reports bootstrap, prewarm and fallback. Engine diagnostics
report a Claude prefix cache miss/hit, the cached token count, the remaining
suffix, and capture or rejection. `BMOE_DONE.n_prompt` counts only tokens
actually prefilled on that request. A cache hit is useful only when the
current prompt still has the identical token prefix; changing model, runtime
configuration, chat rendering, tools, or project instructions can cause a
deliberate miss. No inference approximation or quality setting changes.

In a Claude Code integration check with 29 declared tools, a new conversation
restored 25,088 exact prefix tokens in 0.052 s and prefilled 6,033 new tokens
from a 31,121-token prompt. Claude Code received a normal two-token response.
The previously observed cold request evaluated 31,172 prompt tokens; those
requests used slightly different serialization and are not a controlled
throughput comparison. The focused GPU gate also verified matching output
against cold execution and persistence across session restart on a tiny model.
