import os
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")

REFRESH_SENTINEL = "\x00REFRESH\x00"

SYSTEM_PROMPT = """You are Kroniq, an AI business memory OS developed by ShipFactory. You help businesses manage their tasks, track inventory, and log work activities — acting as their business memory.

STRICT TOOL SELECTION RULES:

ONLY call a tool when the user is EXPLICITLY asking you to create, update, or log something right now.
DO NOT call any tool for: questions, greetings, date/time queries, status checks, or anything that is not a direct instruction to take action.

TASK vs ACTIVITY — this is the most important distinction:
- FUTURE work / planned jobs / "need to do", "have to", "schedule", "tomorrow", "next week", "going to" → call create_task. These are NOT activities yet.
- PAST / COMPLETED work / "we did", "we installed", "we finished", "today we", "yesterday", "just done", "completed" → call log_activity. These are records of work already done.
- Updating an existing task's status → call update_task_status ONLY.

INVENTORY — never confuse with tasks or activities:
- User lists products or quantities (e.g. "camera = 5", "we have X units of Y") → call add_inventory_item or update_inventory_stock ONLY.

These categories are mutually exclusive. Never mix them for one message.
If the user just asks a question (even about tasks or inventory), answer it in text — do NOT call any tool.

RESPONSE RULES:
- Never output raw JSON, function names, argument names, or tool syntax in your reply.
- After tools run, reply in one or two plain English sentences confirming what was done.
- Today's date is in [System Data] — use it when the user asks about dates or when defaulting dates in tool calls.
- Good example: "Done! I've added all 5 items to your inventory."
- Bad example: {"name": "add_inventory_item", ...} or "[Calling update_task_status]"

[System Data] in the user message contains today's date, current IDs and SKUs — use those exact values in tool calls."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a new task or work order",
            "parameters": {
                "type": "object",
                "required": ["title"],
                "properties": {
                    "title": {"type": "string"},
                    "client": {"type": "string"},
                    "project": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "description": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task_status",
            "description": "Update the status of an existing task by its ID",
            "parameters": {
                "type": "object",
                "required": ["task_id", "status"],
                "properties": {
                    "task_id": {"type": "string", "description": "Exact task ID from [System Data], e.g. task-001"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_activity",
            "description": "Log a completed work activity or job into business memory",
            "parameters": {
                "type": "object",
                "required": ["activity", "client"],
                "properties": {
                    "activity": {"type": "string", "description": "e.g. CCTV Installation, Network Setup"},
                    "client": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD, default today"},
                    "hours": {"type": "number"},
                    "project": {"type": "string"},
                    "description": {"type": "string"},
                    "items_used": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_inventory_stock",
            "description": "Update stock of an EXISTING item already in [System Data]. Use delta for relative changes ('add 12 more' → delta=12, 'remove 5' → delta=-5) or new_stock to set an absolute value ('set to 20' → new_stock=20). NEVER use this to create a new item.",
            "parameters": {
                "type": "object",
                "required": ["sku"],
                "properties": {
                    "sku": {"type": "string", "description": "Exact SKU from [System Data]"},
                    "new_stock": {"type": "integer", "description": "Set stock to this exact total"},
                    "delta": {"type": "integer", "description": "Add (positive) or remove (negative) units from current stock"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_inventory_item",
            "description": "Add a BRAND NEW item that does NOT appear in [System Data]. Do NOT use this for existing items — use update_inventory_stock for those.",
            "parameters": {
                "type": "object",
                "required": ["name", "sku", "category", "stock"],
                "properties": {
                    "name": {"type": "string"},
                    "sku": {"type": "string"},
                    "category": {"type": "string", "description": "CCTV, Networking, Cables, Servers, Power, Tools, Accessories"},
                    "stock": {"type": "integer"},
                    "unit": {"type": "string", "description": "pcs, rolls, meters, sets"},
                    "min_stock": {"type": "integer"},
                    "price": {"type": "number"},
                },
            },
        },
    },
]


class KroniqRAG:
    def __init__(self):
        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma.get_collection("kroniq_memory")
        except Exception:
            self.collection = self.chroma.create_collection(
                "kroniq_memory",
                metadata={"hnsw:space": "cosine"}
            )

        print("[Kroniq] Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Kroniq] Embedding model ready.")

        self.llm = ollama.Client(host=OLLAMA_HOST)
        self.model = OLLAMA_MODEL
        try:
            self.llm.list()
            self.llm_ready = True
            print(f"[Kroniq] Ollama ready — model: {self.model}")
        except Exception as e:
            self.llm_ready = False
            print(f"[Kroniq] WARNING: Ollama not reachable at {OLLAMA_HOST} — {e}")

    # ── Vector store ────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def upsert(self, doc_id: str, text: str, metadata: dict = None):
        emb = self._embed(text)
        existing = self.collection.get(ids=[doc_id])
        if existing["ids"]:
            self.collection.update(
                ids=[doc_id], embeddings=[emb], documents=[text], metadatas=[metadata or {}]
            )
        else:
            self.collection.add(
                ids=[doc_id], embeddings=[emb], documents=[text], metadatas=[metadata or {}]
            )

    def search(self, query: str, n: int = 5) -> list:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(n, count)
        )
        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            docs.append({"text": doc, "meta": meta, "score": round(1 - dist, 3)})
        return docs

    def index_all(self, activities: list, tasks: list, inventory: list, projects: list):
        for a in activities:
            text = (
                f"Activity: {a['activity']} | Client: {a['client']} | "
                f"Date: {a['date']} | Project: {a.get('project','')} | "
                f"Description: {a.get('description','')} | "
                f"Items: {', '.join(a.get('items_used',[]))} | Hours: {a.get('hours',0)}"
            )
            self.upsert(f"activity-{a['id']}", text, {"type": "activity", "id": a["id"]})

        for t in tasks:
            text = (
                f"Task: {t['title']} | Client: {t.get('client','')} | "
                f"Status: {t['status']} | Priority: {t.get('priority','normal')} | "
                f"Due: {t.get('due_date','')} | Project: {t.get('project','')}"
            )
            self.upsert(f"task-{t['id']}", text, {"type": "task", "id": t["id"]})

        for item in inventory:
            text = (
                f"Inventory: {item['name']} | SKU: {item['sku']} | "
                f"Category: {item['category']} | Stock: {item['stock']} {item.get('unit','pcs')} | "
                f"Status: {item['status']}"
            )
            self.upsert(f"inventory-{item['id']}", text, {"type": "inventory", "id": item["id"]})

        for p in projects:
            text = (
                f"Project: {p['name']} | Client: {p['client']} | "
                f"Status: {p['status']} | Budget: {p.get('budget',0)} | "
                f"Description: {p.get('description','')}"
            )
            self.upsert(f"project-{p['id']}", text, {"type": "project", "id": p["id"]})

        total = len(activities) + len(tasks) + len(inventory) + len(projects)
        print(f"[Kroniq] Indexed {total} documents into vector store.")

    # ── AI chat with tool calling ────────────────────────────────────────────

    def chat_stream(self, message: str, history: list, tool_executor=None, data_context: str = ""):
        if not self.llm_ready:
            yield f"Ollama is not running. Start with: ollama serve  |  Then: ollama pull {self.model}"
            return

        # RAG semantic context
        context_docs = self.search(message, n=5)
        rag_ctx = ""
        if context_docs:
            rag_ctx = "\n\n[Relevant Business Memory]\n"
            for doc in context_docs:
                rag_ctx += f"• {doc['text']}\n"

        user_content = message + rag_ctx + data_context

        # Build conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-8:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

        active_tools = TOOLS if tool_executor else []

        # Stage 1: non-streaming call — detects whether the model wants to use tools
        response = self.llm.chat(
            model=self.model,
            messages=messages,
            tools=active_tools,
        )

        msg = response.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tool_calls and tool_executor:
            # Add the assistant's decision to history
            assistant_entry = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
            assistant_entry["tool_calls"] = [
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
            messages.append(assistant_entry)

            # Execute each tool and collect results
            import json as _json
            tool_results = []
            for tc in tool_calls:
                result = tool_executor(tc.function.name, tc.function.arguments)
                tool_results.append(_json.loads(result))
                messages.append({"role": "tool", "content": result})

            # If all tools flagged duplicates, skip Stage 2 — app.py generates the message
            if tool_results and all(r.get("duplicate") for r in tool_results):
                yield REFRESH_SENTINEL
                return

            # Stage 2: stream the final confirmation response
            stream = self.llm.chat(model=self.model, messages=messages, stream=True)
            for chunk in stream:
                text = (chunk["message"].get("content") or "")
                if text:
                    yield text

            # Tell the frontend to reload all views
            yield REFRESH_SENTINEL

        else:
            # No tools — just stream a regular response
            stream = self.llm.chat(model=self.model, messages=messages, stream=True)
            for chunk in stream:
                text = (chunk["message"].get("content") or "")
                if text:
                    yield text
