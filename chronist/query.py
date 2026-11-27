"""Run query language -> parameterized SQL.

    config.env = "seaquest" and config.model.arch = "lstm" and steps >= 20000
    (config.optim.lr < 1e-3 or tags contains "baseline") and not config.model.dropout exists
    metric.loss < 0.1 and status in [running, finished]

Fields (every data source but run fields has a prefix, so names never clash)
    config.<key>   a flattened config key (``config.model.arch``); a sub-config
                   named ``config`` is ``config.config.<key>``; odd key names
                   are quoted after the prefix: ``config.`learning rate```
    metric.<key>   a metric's latest value (``metric.loss``, ``metric.train/loss``)
    run fields     id / run_id, name, status, tags, notes, created, updated,
                   heartbeat, steps (no prefix)
Operators
    = != < <= > >= , in [..], not in [..], contains, exists, and, or, not, ( )
Values
    "strings", 'strings', numbers, true, false, null, or bare words (= strings).

Semantics: ``a != x`` is ``not (a = x)``, so it also matches runs that do not
have ``a`` at all. ``contains`` is a case-insensitive substring test for
strings and an element test for lists (``config.layers contains 64``).
Nothing is ever evaluated as Python. The same names are used for table
columns, sorting and grouping.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union


class QueryError(ValueError):
    def __init__(self, msg: str, pos: int = -1):
        super().__init__(msg if pos < 0 else f"{msg} (at position {pos})")
        self.pos = pos


# ---------------------------------------------------------------- tokens
TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<bq>`[^`]*`)
  | (?P<date>\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?![\w.:-]))
  | (?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w./-]))
  | (?P<op><=|>=|!=|==|=|<|>)
  | (?P<punct>[()\[\],])
  | (?P<ident>[A-Za-z_][\w./:-]*)
""", re.VERBOSE)

KEYWORDS = {"and", "or", "not", "in", "contains", "exists", "true", "false", "null"}


@dataclass
class Tok:
    kind: str  # str, bq, num, op, punct, ident, kw, end
    text: str
    pos: int
    value: Any = None


def tokenize(src: str, partial: bool = False) -> List[Tok]:
    """Tokenize. With partial=True an unterminated string/backtick at the end
    is returned as a token (kind 'str_open'/'bq_open') instead of raising."""
    toks: List[Tok] = []
    i = 0
    while i < len(src):
        m = TOKEN_RE.match(src, i)
        if not m:
            ch = src[i]
            if partial and ch in "\"'`":
                kind = "bq_open" if ch == "`" else "str_open"
                toks.append(Tok(kind, src[i:], i, src[i + 1:]))
                return toks
            raise QueryError(f"unexpected character {ch!r}", i)
        kind = m.lastgroup
        text = m.group()
        if kind != "ws":
            t = Tok(kind, text, i)
            if kind == "str":
                t.value = re.sub(r"\\(.)", r"\1", text[1:-1])
            elif kind == "date":
                t.kind, t.value = "str", text
            elif kind == "bq":
                t.value = text[1:-1]
            elif kind == "num":
                t.value = float(text) if any(c in text for c in ".eE") else int(text)
            elif kind == "ident" and text.lower() in KEYWORDS:
                t.kind, t.value = "kw", text.lower()
            elif kind == "op" and text == "==":
                t.text = "="
            toks.append(t)
        i = m.end()
    toks.append(Tok("end", "", len(src)))
    return toks


# ---------------------------------------------------------------- AST
@dataclass
class Field:
    kind: str  # 'run', 'config', 'last'
    name: str


@dataclass
class Cmp:
    field: Field
    op: str  # = != < <= > >= in contains exists
    value: Any = None


@dataclass
class BoolOp:
    op: str  # and / or
    args: List[Any]


@dataclass
class Not:
    arg: Any


Node = Union[Cmp, BoolOp, Not]

RUN_FIELDS = {
    "id": ("r.id", "num"),
    "run_id": ("r.id", "num"),
    "name": ("r.name", "text"),
    "status": ("r.status", "text"),
    "notes": ("r.notes", "text"),
    "created": ("r.created", "time"),
    "updated": ("r.updated", "time"),
    "heartbeat": ("r.heartbeat", "time"),
    "steps": ("r.steps", "num"),
    "tags": (None, "tags"),
}


# Every data source except run fields has its own prefix, so names are never
# ambiguous: config keys are config.<key>, metric values metric.<key>.
PREFIXES = {"config.": "config", "metric.": "last"}


def field_from_name(text: str) -> Optional[Field]:
    for p, kind in PREFIXES.items():
        if text.startswith(p) and len(text) > len(p):
            return Field(kind, text[len(p):])
    if text in RUN_FIELDS:
        return Field("run", text)
    return None


def unknown_field_error(text: str, pos: int) -> QueryError:
    return QueryError(f"unknown field {text!r}: config keys are written config.{text}, metric values "
                      f"metric.{text}; run fields are {', '.join(sorted(k for k in RUN_FIELDS if k != 'id'))}", pos)


class Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = tokenize(src)
        self.i = 0

    def peek(self) -> Tok:
        return self.toks[self.i]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def accept_kw(self, kw: str) -> bool:
        t = self.peek()
        if t.kind == "kw" and t.value == kw:
            self.i += 1
            return True
        return False

    def expect_punct(self, p: str) -> None:
        t = self.next()
        if t.kind != "punct" or t.text != p:
            raise QueryError(f"expected {p!r}, got {t.text or 'end of query'!r}", t.pos)

    def parse(self) -> Optional[Node]:
        if self.peek().kind == "end":
            return None
        node = self.parse_or()
        t = self.peek()
        if t.kind != "end":
            raise QueryError(f"unexpected {t.text!r}", t.pos)
        return node

    def parse_or(self) -> Node:
        args = [self.parse_and()]
        while self.accept_kw("or"):
            args.append(self.parse_and())
        return args[0] if len(args) == 1 else BoolOp("or", args)

    def parse_and(self) -> Node:
        args = [self.parse_not()]
        while self.accept_kw("and"):
            args.append(self.parse_not())
        return args[0] if len(args) == 1 else BoolOp("and", args)

    def parse_not(self) -> Node:
        if self.accept_kw("not"):
            return Not(self.parse_not())
        t = self.peek()
        if t.kind == "punct" and t.text == "(":
            self.next()
            node = self.parse_or()
            self.expect_punct(")")
            return node
        return self.parse_cmp()

    def parse_field(self) -> Field:
        t = self.next()
        if t.kind == "ident":
            nxt = self.peek()
            if t.text in PREFIXES and nxt.kind == "bq" and nxt.pos == t.pos + len(t.text):
                self.next()  # config.`odd key name`
                return Field(PREFIXES[t.text], nxt.value)
            f = field_from_name(t.text)
            if f is not None:
                return f
            if t.text == "last" and nxt.kind == "punct" and nxt.text == "(":
                raise QueryError("last(<metric>) is now written metric.<metric>", t.pos)
            raise unknown_field_error(t.text, t.pos)
        if t.kind == "bq":
            raise QueryError(f"write config.`{t.value}` for a config key (or metric.`{t.value}`)", t.pos)
        raise QueryError(f"expected a field name, got {t.text or 'end of query'!r}", t.pos)

    def parse_value(self) -> Any:
        t = self.next()
        if t.kind in ("str", "num"):
            return t.value
        if t.kind == "kw" and t.value in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[t.value]
        if t.kind == "ident":
            return t.text  # bare word = string
        raise QueryError(f"expected a value, got {t.text or 'end of query'!r}", t.pos)

    def parse_list(self) -> List[Any]:
        self.expect_punct("[")
        vals: List[Any] = []
        if self.peek().kind == "punct" and self.peek().text == "]":
            self.next()
            return vals
        while True:
            vals.append(self.parse_value())
            t = self.next()
            if t.kind == "punct" and t.text == "]":
                return vals
            if not (t.kind == "punct" and t.text == ","):
                raise QueryError("expected ',' or ']'", t.pos)

    def parse_cmp(self) -> Node:
        f = self.parse_field()
        t = self.next()
        if t.kind == "op":
            return Cmp(f, t.text, self.parse_value())
        if t.kind == "kw":
            if t.value == "in":
                return Cmp(f, "in", self.parse_list())
            if t.value == "not" and self.accept_kw("in"):
                return Not(Cmp(f, "in", self.parse_list()))
            if t.value == "contains":
                return Cmp(f, "contains", self.parse_value())
            if t.value == "exists":
                return Cmp(f, "exists")
        raise QueryError(f"expected an operator after {f.name!r}, got {t.text or 'end of query'!r}", t.pos)


def parse(src: str) -> Optional[Node]:
    return Parser(src).parse()


# ---------------------------------------------------------------- SQL
SQL_OPS = {"=": "=", "!=": "=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


def _time_value(v: Any, pos_hint: str) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        try:
            d = _dt.datetime.fromisoformat(v)
        except ValueError:
            raise QueryError(f"cannot parse time {v!r} for {pos_hint}; use ISO format like 2026-11-01")
        if d.tzinfo is None:
            d = d.astimezone()  # local time
        return d.timestamp()
    raise QueryError(f"invalid time value {v!r}")


class Compiler:
    def __init__(self):
        self.params: List[Any] = []

    def p(self, v: Any) -> str:
        self.params.append(v)
        return "?"

    def compile(self, node: Node) -> str:
        if isinstance(node, BoolOp):
            return "(" + f" {node.op.upper()} ".join(self.compile(a) for a in node.args) + ")"
        if isinstance(node, Not):
            return f"(NOT {self.compile(node.arg)})"
        if node.op == "!=":
            return f"(NOT {self.compile(Cmp(node.field, '=', node.value))})"
        if node.op == "in":
            if not node.value:
                return "0"
            return "(" + " OR ".join(self.compile(Cmp(node.field, "=", v)) for v in node.value) + ")"
        f = node.field
        if f.kind == "run":
            return self.run_field(f.name, node.op, node.value)
        if f.kind == "last":
            return self.last_field(f.name, node.op, node.value)
        return self.config_field(f.name, node.op, node.value)

    def run_field(self, name: str, op: str, v: Any) -> str:
        col, typ = RUN_FIELDS[name]
        if typ == "tags":
            if op == "exists":
                return "EXISTS (SELECT 1 FROM tags t WHERE t.run_id=r.id)"
            if op in ("=", "contains"):
                return f"EXISTS (SELECT 1 FROM tags t WHERE t.run_id=r.id AND t.tag={self.p(str(v))})"
            raise QueryError(f"operator {op!r} not supported for tags; use = or contains")
        if op == "exists":
            return f"({col} IS NOT NULL)"
        if op == "contains":
            return f"(instr(lower({col}), lower({self.p(str(v))})) > 0)"
        if typ == "time":
            v = _time_value(v, name)
        elif typ == "num" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
            raise QueryError(f"{name} needs a number, got {v!r}")
        elif typ == "text" and v is not None and not isinstance(v, str):
            v = str(v)
        if v is None:
            return f"({col} IS NULL)"
        return f"({col} {SQL_OPS[op]} {self.p(v)})"

    def last_field(self, key: str, op: str, v: Any) -> str:
        base = f"SELECT 1 FROM summary s WHERE s.run_id=r.id AND s.key={self.p(key)}"
        if op == "exists":
            return f"EXISTS ({base})"
        if op == "contains" or not isinstance(v, (int, float)) or isinstance(v, bool):
            raise QueryError(f"last({key}) must be compared with a number")
        return f"EXISTS ({base} AND s.value {SQL_OPS[op]} {self.p(float(v))})"

    def config_field(self, key: str, op: str, v: Any) -> str:
        base = f"SELECT 1 FROM config c WHERE c.run_id=r.id AND c.key={self.p(key)}"
        if op == "exists":
            return f"EXISTS ({base})"
        if op == "contains":
            # strings: case-insensitive substring; lists: an element equal to the value
            sv = str(v).lower() if not isinstance(v, bool) else ("1" if v else "0")
            return (f"EXISTS ({base} AND ((c.vtype='str' AND instr(lower(c.v_text), {self.p(sv)}) > 0) "
                    f"OR (c.vtype='list' AND EXISTS (SELECT 1 FROM json_each(c.v_text) j "
                    f"WHERE lower(CAST(j.value AS TEXT)) = {self.p(sv)}))))")
        sop = SQL_OPS[op]
        if v is None:
            if op != "=":
                raise QueryError("null can only be compared with = or !=")
            return f"EXISTS ({base} AND c.vtype='null')"
        if isinstance(v, bool):
            if op != "=":
                raise QueryError("booleans can only be compared with = or !=")
            return f"EXISTS ({base} AND c.vtype='bool' AND c.v_num={self.p(int(v))})"
        if isinstance(v, (int, float)):
            return f"EXISTS ({base} AND c.vtype IN ('int','float') AND c.v_num {sop} {self.p(v)})"
        return f"EXISTS ({base} AND c.vtype='str' AND c.v_text {sop} {self.p(v)})"


def to_sql(src: str) -> Tuple[str, List[Any]]:
    """Compile a query to a SQL boolean expression over ``runs r``."""
    node = parse(src or "")
    if node is None:
        return "1", []
    c = Compiler()
    return c.compile(node), c.params


# ---------------------------------------------------------------- completion
@dataclass
class CompletionContext:
    kind: str  # 'field', 'value', 'operator', 'none'
    prefix: str  # partial text being completed
    start: int  # where the partial text starts in the query
    field: Optional[Field] = None
    quote: Optional[str] = None  # open quote char for value completion


def completion_context(src: str) -> CompletionContext:
    """What is being typed at the end of ``src`` (the text before the cursor)?"""
    try:
        toks = tokenize(src, partial=True)
    except QueryError:
        return CompletionContext("none", "", len(src))
    if toks and toks[-1].kind == "end":
        toks = toks[:-1]
    trailing_space = src[-1:].isspace() if src else True

    partial = None
    if toks and not trailing_space and toks[-1].kind in ("ident", "str_open", "bq_open", "num", "kw"):
        partial = toks[-1]
        prev = toks[:-1]
    else:
        prev = toks

    def field_before(i: int) -> Optional[Field]:
        """Field ending at prev[i] (inclusive)."""
        t = prev[i] if 0 <= i < len(prev) else None
        if t is None:
            return None
        if t.kind == "bq" and i >= 1 and prev[i - 1].kind == "ident" and prev[i - 1].text in PREFIXES:
            return Field(PREFIXES[prev[i - 1].text], t.value)
        if t.kind == "ident":
            return field_from_name(t.text)
        return None

    last = prev[-1] if prev else None
    # value position: after an operator, after 'in [' or ',' inside a list, after contains
    in_value = False
    fld = None
    if last is not None:
        if last.kind == "op" or (last.kind == "kw" and last.value == "contains"):
            in_value, fld = True, field_before(len(prev) - 2)
        elif last.kind == "punct" and last.text in ("[", ","):
            j = len(prev) - 1
            while j >= 0 and not (prev[j].kind == "kw" and prev[j].value == "in"):
                j -= 1
            if j >= 1:
                in_value, fld = True, field_before(j - 1)
                if fld is None and j >= 2 and prev[j - 1].kind == "kw" and prev[j - 1].value == "not":
                    fld = field_before(j - 2)

    if in_value:
        if partial is None:
            return CompletionContext("value", "", len(src), fld)
        if partial.kind == "str_open":
            return CompletionContext("value", partial.value, partial.pos, fld, quote=partial.text[0])
        return CompletionContext("value", partial.text, partial.pos, fld)

    # field position: start, after and/or/not, after '('
    field_pos = last is None or (last.kind == "kw" and last.value in ("and", "or", "not")) \
        or (last.kind == "punct" and last.text == "(")
    if field_pos:
        if partial is None:
            return CompletionContext("field", "", len(src))
        if partial.kind == "bq_open":
            return CompletionContext("field", partial.value, partial.pos, quote="`")
        if partial.kind in ("ident", "kw"):
            return CompletionContext("field", partial.text, partial.pos)
        return CompletionContext("none", "", len(src))

    # right after a complete field (not a bare-word value): suggest operators
    before = prev[-2] if len(prev) >= 2 else None
    after_op = before is not None and (before.kind == "op" or (before.kind == "kw" and before.value == "contains"))
    if not after_op and field_before(len(prev) - 1) is not None:
        return CompletionContext("operator", partial.text if partial else "",
                                 partial.pos if partial else len(src), field_before(len(prev) - 1))
    # after a complete comparison: suggest and/or
    return CompletionContext("operator", partial.text if partial else "",
                             partial.pos if partial else len(src))
