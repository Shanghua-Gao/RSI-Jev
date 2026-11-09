# agentguard

```
$ agentguard eval apps/agentguard/seed.jsonl
```

A Claude Code hook that asks a decision model whether a tool call is clearly safe, and skips the
permission prompt when the model is very sure. It can only remove a prompt. It never denies, and
when anything goes wrong you get the normal prompt.

**Measured** with the v7.0-VL 3B MLX 4-bit build served on an M2 Max (threshold 0.85, chosen
on `seed.jsonl` only):

| set | calls | dangerous | false allows | 95% upper bound | safe calls allowed |
|---|---|---|---|---|---|
| seed.jsonl | 60 | 30 | 0 | 0.095 | 19 of 30 |
| holdout.jsonl | 40 | 25 | 0 | 0.113 | 8 of 15 |

The model alone, without the blocklist, also allowed no dangerous call in either set at 0.85
(at 0.5 it allowed 2 of 30 and 3 of 25). These are small sets: an upper bound of 0.11 is all
they can show. Median latency per call 156 to 182 ms.

## Run it

On a Mac, [install RSI-Jev](../../docs/mac.md) and ask `decide` one question so the model is on disk.
Then add this to `.claude/settings.json`. Elsewhere, start a server and set `AGENTGUARD_URL`.

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash|Read|Grep|Glob|Edit|Write",
  "hooks": [{"type": "command", "command": "agentguard hook", "timeout": 5}]}]}}
```

Set `AGENTGUARD_URL` (default `http://127.0.0.1:8000`), `AGENTGUARD_MIN_P` (default 0.85) and
`AGENTGUARD_LOG` (default `~/.agentguard/log.jsonl`) in the hook command if you need to. It uses
only the standard library.

## What it does

1. Only Bash, Read, Grep, Glob, Edit and Write, in the `default` and `acceptEdits` modes, are
   considered. Everything else goes to the normal prompt.
2. A blocklist runs first, and nothing overrides it: deletes, `git` beyond reading, force flags,
   anything piped into a shell, `sudo`, network tools, package installs, inline code, obfuscation,
   secrets and credentials, writes outside the project, and writes to `.claude`, `.git`, CI and
   build files. It is deliberately eager: a false hit costs one prompt.
3. The model is asked for P(safe). At or above the threshold the hook allows the call.

Server down, a timeout of 3 seconds, a bad answer, a crash or a call too long to judge whole all
print nothing, which is the normal prompt. The hook never exits 2, because that would be a denial.
`permissions.deny` and `ask` rules in your settings still apply: the hooks reference says they are
evaluated whatever a hook returns.

The log has one JSON line per call: tool, decision, reason, P(safe), and a truncated command with
secrets redacted, or just the path. File contents are never logged.

## Measuring it

`agentguard eval FILE` asks the model about every labelled call, blocked or not, and prints the
false-allow rate with a one-sided 95% upper bound, the share auto-allowed, a threshold sweep, and
the same sweep for the model with no blocklist.

`seed.jsonl` is 60 calls, 30 dangerous, labelled by hand by the author of the blocklist, so it
flatters the blocklist. `holdout.jsonl` is 40 calls, 25 dangerous, written adversarially by a
separate agent that never saw the blocklist or the seed set.

What has been measured, with no model involved:

| set | dangerous stopped by the blocklist | safe sent to a human by the blocklist |
|---|---|---|
| seed | 28 of 30 | 2 of 30 |
| holdout, first pass | 24 of 25 | 3 of 15 |

The dangerous calls that get past the blocklist are up to the model. Those are 2 on the seed set
and 1 on the holdout, and the holdout has not been looked at, so that it stays unseen for the
first model run. With 30 dangerous calls, even zero false allows only bounds the rate at about
10%.

A blocklist on command text can be beaten, and a model reads text an attacker may have written.
Treat this as a way to see fewer prompts, not as a sandbox.

Tests: `pytest tests/test_agentguard.py` (a fake server, no model).
