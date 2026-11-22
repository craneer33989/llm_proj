"""Budgeted tool environment for future teacher collection and RL rollouts.

No reference answers or supporting-fact labels are imported by this module.
"""
import json
from pathlib import Path

TOOLS = [
    {"type": "function", "function": {"name": "local_search", "description": "Search a shared static English Wikipedia paragraph corpus. Search again using new entities when needed.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_document", "description": "Read a source paragraph returned by local_search; this is not a complete live Wikipedia article.",
     "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"], "additionalProperties": False}}},
]


class SearchSession:
    def __init__(self, engine, question, max_calls=6, mode="bm25", log_path=None):
        if not 1 <= max_calls <= 20:
            raise ValueError("max_calls must be 1..20")
        self.engine, self.question, self.mode = engine, question, mode
        self.max_calls, self.trace, self.observed_docs = max_calls, [], set()
        self.log_path = Path(log_path) if log_path else None

    def call(self, name, arguments):
        if len(self.trace) >= self.max_calls:
            return {"error": "tool_budget_exhausted", "remaining_calls": 0}
        try:
            if name == "local_search":
                if set(arguments) - {"query", "top_k"}:
                    raise ValueError("Unexpected search arguments")
                top_k = arguments.get("top_k", 5)
                if not 1 <= top_k <= 10:
                    raise ValueError("top_k must be 1..10")
                result = self.engine.search(arguments["query"], top_k, self.mode)
                self.observed_docs.update(r["doc_id"] for r in result["results"])
            elif name == "read_document":
                if set(arguments) != {"document_id"}:
                    raise ValueError("Expected document_id only")
                if arguments["document_id"] not in self.observed_docs:
                    raise ValueError("Read only a document returned by search")
                result = self.engine.read(arguments["document_id"])
            else:
                raise ValueError("Unknown tool")
        except (ValueError, KeyError, TypeError) as error:
            result = {"error": str(error)}
        entry = {"step": len(self.trace)+1, "tool": name, "arguments": arguments, "result": result}
        self.trace.append(entry)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {**result, "remaining_calls": self.max_calls-len(self.trace)}
