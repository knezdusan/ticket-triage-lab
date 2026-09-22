# Day 2 — Python idiom (Mon 21 Sep)

Keyboard only. You write every function body in `day2.py`. The tests in `test_day2.py` tell you when you're right.

## Before you start (5 min)

1. **Turn off AI autocomplete in the editor for today** — Devin tab-completion, Copilot, anything. Autocomplete writing the body defeats the whole day.
2. Allowed: `docs.python.org`, the REPL (`uv run python`), and asking Claude to *explain* something. Not to write it.
3. Run one block at a time:

```bash
uv run pytest exercises -k b1 -q      # block 1 only
uv run pytest exercises -q            # everything
```

Every test fails at the start. That's correct. Your job is to turn them green, one block at a time.

## Plan for the day (~6h with breaks)

| Block | Time | Topic | TypeScript you already know |
|---|---|---|---|
| b1 | 30 min | Types and basic shapes | `Record<string, number>`, `T \| undefined` |
| b2 | 60 min | Comprehensions | `.filter().map()`, `Object.fromEntries` |
| b3 | 45 min | Standard library | `reduce`, `(acc[k] ??= []).push()`, RegExp, `Date` |
| — | break | | |
| b4 | 45 min | Enums, dataclasses, dunder methods | string enums, `readonly` classes, `toString()` |
| b5 | 45 min | Exceptions and context managers | `try/finally`, `using` |
| — | break | | |
| b6 | 45 min | Decorators | higher-order function wrappers |
| b7 | 45 min | async / await | `Promise.all`, `p-limit` |
| b8 | 30 min | Generators | `function*` / `yield` |
| drill | 20 min | From blank | see the end |

---

## b1 — Types and basic shapes

- `list[dict]` ≈ `Record<string, unknown>[]`. `dict[str, int]` ≈ `Record<string, number>`.
- `str | None` ≈ `string | undefined`. There is one "nothing": `None`.
- Type hints are **not enforced at runtime.** They are for you, the editor, and tools. Pydantic (tomorrow) is where types actually get checked.
- Empty containers are falsy: `if not tickets:` means "if the list is empty".

**Write:** `count_by_priority`. Do it with a plain loop first. You'll redo it better in b3.

## b2 — Comprehensions

The single most Python thing there is. Read them left to right as "give me X, for each item, if condition".

```python
[t["id"] for t in tickets if t["priority"] == "P1"]      # list
{t["id"]: t for t in tickets}                              # dict
{t["module"] for t in tickets if t["module"]}              # set
any(t["priority"] == "P1" for t in tickets)                # generator into any()
```

Also learn today: `sorted(items, key=...)`, `enumerate`, `zip`, `any`, `all`.

**Write:** `short_descriptions`, `index_by_id`, `unique_modules`, `any_p1`. One line each is possible. Aim for that.

## b3 — Standard library you'll actually use

- `collections.Counter` — counting, done. Redo `count_by_priority` in one line with it.
- `collections.defaultdict(list)` — the `??= []` pattern, built in.
- `re` — `re.findall`, raw strings `r"..."`, `\b` word boundaries.
- `datetime`, `timedelta` — date arithmetic without a library.
- `pathlib.Path` and `json` — you'll need both in b5.

**Write:** `group_by_assignment`, `extract_tcodes` (SAP transaction codes like `VA01`, `ME21N`, `ST22`; uppercase only; no duplicates; keep first-seen order), `sla_due`.

## b4 — Enums, dataclasses, dunder methods

- `class Priority(StrEnum)` — an enum whose values are real strings, so `Priority.P1 == "P1"` is true.
- `@dataclass(frozen=True)` — writes `__init__`, `__eq__`, `__repr__` for you, and makes the object read-only.
- `@property` — a method that reads like a field. Like a TS getter.
- "Dunder" = double underscore. `__str__` ≈ `toString()`. `__eq__`, `__lt__`, `__len__` are how Python operators and built-ins talk to your class.
- `self` is explicit in every method. You'll forget it at least twice today. Everyone does.

**Write:** the `Priority` enum, `derive_priority` (the impact × urgency matrix from the ITSM notes), the `Ticket` dataclass with `is_urgent` and `__str__`, and `sort_by_urgency`.

The matrix to implement:

| impact \\ urgency | high | medium | low |
|---|---|---|---|
| **high** | P1 | P2 | P3 |
| **medium** | P2 | P3 | P4 |
| **low** | P3 | P4 | P4 |

## b5 — Exceptions and context managers

- `try / except / else / finally`. Catch specific exceptions, never a bare `except:`.
- `raise NewError(...) from original` keeps the cause chain. This matters when debugging a failed LLM call three layers down.
- `with open(path) as f:` closes the file even if something throws. That's a context manager.
- `contextlib.contextmanager` lets you write your own with one `yield`.

**Write:** `load_tickets` (reads JSON Lines — one JSON object per line; skip blank lines; a broken line raises `TicketLoadError` with the line number in the message; a missing file raises `TicketLoadError` *from* the original error). Then `timer` — a context manager that yields a dict and puts the elapsed milliseconds into it under `"ms"` when the block ends.

## b6 — Decorators

A decorator is a function that takes a function and returns a wrapped one. `@retry(times=3)` above a function is the same as `fn = retry(times=3)(fn)`. You've written this exact wrapper in TypeScript.

- Use `functools.wraps` so the wrapped function keeps its name.
- Know it exists: `functools.lru_cache` — memoisation in one line.

**Write:** `retry`. Retry only on the listed exception types. Anything else goes straight through. After the last attempt, re-raise the original exception. No sleeping — keep it simple.

This is the one you'll use on Friday, around real model calls.

## b7 — async / await

Same idea as JavaScript, one important difference: Python doesn't have an event loop running by default. `asyncio.run(main())` starts one.

- `asyncio.gather(*tasks)` ≈ `Promise.all` — results come back in input order.
- `asyncio.Semaphore(n)` ≈ `p-limit(n)` — caps how many run at once.
- `async with semaphore:` acquires and releases around a block.

**Write:** `classify_all` — run an async function over every text, never more than `max_concurrency` at once, results in input order.

This is your VectorMatch rate-limiter lesson, in Python. The test checks the peak concurrency is exactly the cap.

## b8 — Generators

`yield` makes a function lazy — it produces values one at a time, like `function*` in JS. You'll use this to send texts to the embedding API in batches.

- Python 3.12 has `itertools.batched`, which does exactly this. Write it yourself first, then look at the built-in.
- New 3.12 generic syntax: `def chunks[T](items: Sequence[T], size: int)` — very close to TypeScript's `<T>`.

**Write:** `chunks`.

---

## Watch out for (the classic traps for JS people)

- **Mutable default arguments.** `def f(x=[])` shares one list across every call. Use `x=None`, then create the list inside.
- `is` checks identity, `==` checks value. Use `is` only for `None`: `if x is None`.
- `/` always gives a float. `//` is integer division.
- No `++`. No `&&` / `||` — use `and` / `or`. No `!` — use `not`.
- Dicts keep insertion order. `sorted` is stable. You can rely on both.

## End-of-day drill (20 min)

Close `day2.py`. Open a new empty file. Write `retry`, `chunks` and `classify_all` again from memory. Then run the tests against your new versions.

Whatever you can't write from blank is tomorrow morning's first 15 minutes.
