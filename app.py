import json
import os
import re
import uuid
from datetime import datetime
from flask import Flask, jsonify, render_template, request, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from rag import KroniqRAG, REFRESH_SENTINEL

load_dotenv()

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Bootstrap ────────────────────────────────────────────────────────────────

rag = KroniqRAG()

def _load(filename: str) -> list:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def _save(filename: str, data: list):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, filename), "w") as f:
        json.dump(data, f, indent=2)

# Seed vector store on startup
rag.index_all(
    _load("activities.json"),
    _load("tasks.json"),
    _load("inventory.json"),
    _load("projects.json"),
)

# ── Frontend ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

# ── Tool executor (called by the AI when it uses a tool) ─────────────────────

def _word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _tool_executor(fn_name: str, args: dict) -> str:
    try:
        if fn_name == "create_task":
            tasks = _load("tasks.json")
            new_title = args.get("title", "New Task")
            new_client = args.get("client", "").lower()
            for t in tasks:
                if t.get("status") == "completed":
                    continue
                score = _word_overlap(new_title, t["title"])
                if new_client and new_client in t.get("client", "").lower():
                    score += 0.15
                if score >= 0.5:
                    return json.dumps({
                        "ok": False,
                        "duplicate": True,
                        "message": f"Task '{t['title']}' already exists (ID: {t['id']}, status: {t['status']}). Not adding a duplicate.",
                    })
            new_id = f"task-{str(uuid.uuid4())[:8]}"
            task = {
                "id": new_id,
                "title": args.get("title", "New Task"),
                "client": args.get("client", ""),
                "project": args.get("project", ""),
                "priority": args.get("priority", "medium"),
                "status": "pending",
                "due_date": args.get("due_date", ""),
                "assigned_to": "Denni Alwin",
                "description": args.get("description", ""),
            }
            tasks.insert(0, task)
            _save("tasks.json", tasks)
            rag.upsert(
                f"task-{new_id}",
                f"Task: {task['title']} | Client: {task['client']} | Status: pending | Priority: {task['priority']}",
                {"type": "task", "id": new_id},
            )
            return json.dumps({"ok": True, "id": new_id, "message": f"Task '{task['title']}' created (ID: {new_id})"})

        elif fn_name == "update_task_status":
            task_id = args.get("task_id", "")
            status = args.get("status", "")
            tasks = _load("tasks.json")
            for task in tasks:
                if task["id"] == task_id:
                    task["status"] = status
                    _save("tasks.json", tasks)
                    return json.dumps({"ok": True, "message": f"Task '{task['title']}' updated to {status}"})
            return json.dumps({"ok": False, "message": f"Task ID '{task_id}' not found"})

        elif fn_name == "log_activity":
            activities = _load("activities.json")
            new_activity_name = args.get("activity", "")
            new_client = args.get("client", "")
            new_date = args.get("date", datetime.now().strftime("%Y-%m-%d"))
            for a in activities:
                same_date = a.get("date", "") == new_date
                same_client = a.get("client", "").lower() == new_client.lower()
                if same_date and same_client:
                    score = _word_overlap(new_activity_name, a.get("activity", ""))
                    if score >= 0.4:
                        return json.dumps({
                            "ok": False,
                            "duplicate": True,
                            "message": f"Activity '{a['activity']}' for {a['client']} on {a['date']} already logged (ID: {a['id']}). Not adding a duplicate.",
                        })
            new_id = f"act-{str(uuid.uuid4())[:8]}"
            activity = {
                "id": new_id,
                "date": args.get("date", datetime.now().strftime("%Y-%m-%d")),
                "activity": args.get("activity", "Work Activity"),
                "client": args.get("client", ""),
                "project": args.get("project", ""),
                "technician": "Denni Alwin",
                "items_used": args.get("items_used", []),
                "description": args.get("description", ""),
                "status": "completed",
                "hours": int(args.get("hours", 0)),
            }
            activities.insert(0, activity)
            _save("activities.json", activities)
            rag.upsert(
                f"activity-{new_id}",
                f"Activity: {activity['activity']} | Client: {activity['client']} | Date: {activity['date']} | Description: {activity['description']}",
                {"type": "activity", "id": new_id},
            )
            return json.dumps({"ok": True, "id": new_id, "message": f"Activity '{activity['activity']}' for {activity['client']} logged on {activity['date']}"})

        elif fn_name == "update_inventory_stock":
            sku = args.get("sku", "")
            delta = args.get("delta")
            new_stock_arg = args.get("new_stock")
            inventory = _load("inventory.json")
            for item in inventory:
                if item["sku"] == sku or item["id"] == sku:
                    old = item["stock"]
                    if delta is not None:
                        item["stock"] = max(0, old + int(delta))
                    elif new_stock_arg is not None:
                        item["stock"] = max(0, int(new_stock_arg))
                    else:
                        return json.dumps({"ok": False, "message": "Provide either delta or new_stock"})
                    item["movement"] = item["stock"] - old
                    item["status"] = _stock_status(item["stock"], item.get("min_stock", 5))
                    _save("inventory.json", inventory)
                    return json.dumps({"ok": True, "message": f"Stock for {item['name']} updated from {old} to {item['stock']}"})
            return json.dumps({"ok": False, "message": f"SKU '{sku}' not found. Check [System Data] for valid SKUs."})

        elif fn_name == "add_inventory_item":
            inventory = _load("inventory.json")
            # Safety: if SKU already exists, treat as a stock update instead
            existing_sku = args.get("sku", "")
            for item in inventory:
                if item["sku"] == existing_sku or item["id"] == existing_sku:
                    return _tool_executor("update_inventory_stock", {"sku": existing_sku, "new_stock": args.get("stock", item["stock"])})
            new_id = args.get("sku", f"ITEM-{str(uuid.uuid4())[:6].upper()}")
            stock = int(args.get("stock", 0))
            min_stock = int(args.get("min_stock", 5))
            item = {
                "id": new_id,
                "name": args.get("name", "New Item"),
                "sku": args.get("sku", new_id),
                "category": args.get("category", "General"),
                "stock": stock,
                "unit": args.get("unit", "pcs"),
                "status": _stock_status(stock, min_stock),
                "min_stock": min_stock,
                "price": float(args.get("price", 0)),
                "movement": 0,
                "location": args.get("location", ""),
            }
            inventory.append(item)
            _save("inventory.json", inventory)
            rag.upsert(
                f"inventory-{new_id}",
                f"Inventory: {item['name']} | SKU: {item['sku']} | Category: {item['category']} | Stock: {item['stock']} {item['unit']}",
                {"type": "inventory", "id": new_id},
            )
            return json.dumps({"ok": True, "message": f"Item '{item['name']}' (SKU: {new_id}) added to inventory"})

        return json.dumps({"ok": False, "message": f"Unknown tool: {fn_name}"})

    except Exception as e:
        return json.dumps({"ok": False, "message": f"Tool error: {str(e)}"})


def _build_data_context() -> str:
    """Compact snapshot of today's date, tasks + inventory so the AI knows IDs/SKUs for tool calls."""
    today = datetime.now().strftime("%Y-%m-%d")
    tasks = _load("tasks.json")
    inventory = _load("inventory.json")
    ctx = f"\n\n[System Data — use exact IDs/SKUs when calling update tools]\nToday's date: {today}\nTasks:\n"
    for t in tasks[:15]:
        ctx += f"  {t['id']}: {t['title']} [{t['status']}] - {t.get('client', '')}\n"
    ctx += "Inventory:\n"
    for item in inventory[:15]:
        ctx += f"  SKU={item['sku']}: {item['name']} stock={item['stock']}{item.get('unit','pcs')} [{item['status']}]\n"
    return ctx


# ── Inventory message parser ─────────────────────────────────────────────────

_UNIT_MAP = {
    'coil': 'rolls', 'coils': 'rolls', 'roll': 'rolls',
    'unit': 'pcs', 'units': 'pcs', 'piece': 'pcs', 'pieces': 'pcs', 'pc': 'pcs',
    'meter': 'meters', 'metre': 'meters', 'meters': 'meters', 'metres': 'meters',
    'set': 'sets',
}
_SKIP_WORDS = ('we have', 'i have', 'several', 'products', 'inventory', 'following', 'below', 'these')
_QUESTION_STARTS = ('what', 'how', 'can', 'show', 'list', 'tell', 'find', 'is ', 'are ',
                    'do ', 'does', 'did', 'which', 'where', 'when', 'why', 'give', 'get')

def _parse_inventory_message(message):
    """Return list of {name, qty, unit} ONLY for structured inventory listing messages.
    Requires -> or → separators; rejects questions and regular sentences."""
    msg = message.strip()
    # Must have a list separator to qualify
    if '->' not in msg and '→' not in msg:
        return []
    # Questions and normal sentences should go to AI
    if '?' in msg:
        return []
    low = msg.lower()
    if any(low.startswith(q) for q in _QUESTION_STARTS):
        return []

    parts = re.split(r'->|→', msg)
    items = []
    for part in parts:
        part = part.strip(' -•*\n')
        m = re.search(r'^(.+?)\s*[=:]\s*(\d+)\s*([a-zA-Z]+)?', part)
        if not m:
            continue
        name = m.group(1).strip()
        qty = int(m.group(2))
        raw_unit = (m.group(3) or 'pcs').lower()
        unit = _UNIT_MAP.get(raw_unit, raw_unit)
        if len(name) < 2 or any(s in name.lower() for s in _SKIP_WORDS):
            continue
        items.append({'name': name, 'qty': qty, 'unit': unit})
    return items


def _handle_inventory_items(items):
    """Call tool executor for each parsed inventory item; return list of result strings."""
    inventory = _load("inventory.json")
    results = []
    for item in items:
        name_lower = item['name'].lower()
        # Try to find existing item by name — word-overlap match (e.g. "Hikvision camera" matches "Hikvision 4MP IP Camera")
        def _name_score(existing):
            words = set(name_lower.split())
            ex_words = set(existing['name'].lower().split())
            return len(words & ex_words) / max(len(words), 1)
        match = max(inventory, key=_name_score, default=None)
        if match and _name_score(match) < 0.4:
            match = None
        if match:
            res = _tool_executor("update_inventory_stock",
                                 {"sku": match["sku"], "new_stock": item['qty']})
        else:
            # Generate a clean SKU from the name
            sku = re.sub(r'[^A-Z0-9]', '-', item['name'].upper())[:20].strip('-')
            res = _tool_executor("add_inventory_item", {
                "name": item['name'],
                "sku": sku,
                "category": "General",
                "stock": item['qty'],
                "unit": item['unit'],
            })
        results.append(json.loads(res))
        inventory = _load("inventory.json")  # reload after each write
    return results


# ── AI Chat ──────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    message = body.get("message", "").strip()
    history = body.get("history", [])

    if not message:
        return jsonify({"error": "message required"}), 400

    # Direct inventory parsing — bypass AI tool selection entirely for inventory listings
    inv_items = _parse_inventory_message(message)
    if inv_items:
        def generate_inv():
            results = _handle_inventory_items(inv_items)
            ok = [r for r in results if r.get("ok")]
            fail = [r for r in results if not r.get("ok")]
            count = len(ok)
            confirm = f"Done! {count} item{'s' if count != 1 else ''} added to inventory."
            if fail:
                confirm += f" ({len(fail)} could not be saved.)"
            card = {
                "type": "inventory",
                "title": "Inventory Updated",
                "rows": [r.get("message", "") for r in ok],
                "errors": [r.get("message", "") for r in fail],
            }
            yield f"data: {json.dumps({'text': confirm})}\n\n"
            yield f"data: {json.dumps({'card': card})}\n\n"
            yield f"data: {json.dumps({'refresh': True})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(generate_inv()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    data_ctx = _build_data_context()

    # Track tool calls so we can emit a card after AI executes tools
    _tool_log = []
    def _tracked_executor(fn_name, args):
        result = _tool_executor(fn_name, args)
        parsed = json.loads(result)
        _tool_log.append({"fn": fn_name, "args": args, "result": parsed})
        return result

    _CARD_META = {
        "create_task":            ("task",      "Task Created"),
        "update_task_status":     ("task",      "Task Updated"),
        "log_activity":           ("activity",  "Activity Logged"),
        "update_inventory_stock": ("inventory", "Stock Updated"),
        "add_inventory_item":     ("inventory", "Item Added"),
    }

    def generate():
        for chunk in rag.chat_stream(message, history, tool_executor=_tracked_executor, data_context=data_ctx):
            if chunk == REFRESH_SENTINEL:
                if _tool_log:
                    ok_logs = [e for e in _tool_log if e["result"].get("ok")]
                    dup_logs = [e for e in _tool_log if e["result"].get("duplicate")]
                    if ok_logs:
                        fns = [e["fn"] for e in ok_logs]
                        card_type, card_title = _CARD_META.get(fns[0], ("general", "Done"))
                        if len(set(fns)) > 1:
                            card_title = "Changes Applied"
                        card = {
                            "type": card_type,
                            "title": card_title,
                            "rows": [e["result"].get("message", "") for e in ok_logs],
                            "errors": [e["result"].get("message", "") for e in _tool_log if not e["result"].get("ok")],
                        }
                        yield f"data: {json.dumps({'card': card})}\n\n"
                    elif dup_logs:
                        fn = dup_logs[0]["fn"]
                        card_type = "task" if fn in ("create_task", "update_task_status") else "activity"
                        item_type = "task" if card_type == "task" else "activity"
                        yield f"data: {json.dumps({'text': f'That {item_type} is already in your records.'})}\n\n"
                        card = {
                            "type": card_type,
                            "title": "Already Exists",
                            "rows": [],
                            "errors": [e["result"].get("message", "") for e in dup_logs],
                        }
                        yield f"data: {json.dumps({'card': card})}\n\n"
                yield f"data: {json.dumps({'refresh': True})}\n\n"
            else:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Activities ───────────────────────────────────────────────────────────────

@app.route("/api/activities", methods=["GET"])
def get_activities():
    activities = _load("activities.json")
    activities.sort(key=lambda x: x.get("date", ""), reverse=True)
    return jsonify(activities)

@app.route("/api/activities", methods=["POST"])
def create_activity():
    body = request.get_json(force=True)
    activities = _load("activities.json")

    new_id = f"act-{str(uuid.uuid4())[:8]}"
    activity = {
        "id": new_id,
        "date": body.get("date", datetime.now().strftime("%Y-%m-%d")),
        "activity": body.get("activity", "Work Activity"),
        "client": body.get("client", ""),
        "project": body.get("project", ""),
        "technician": body.get("technician", "Denni Alwin"),
        "items_used": body.get("items_used", []),
        "description": body.get("description", ""),
        "status": "completed",
        "hours": int(body.get("hours", 0)),
    }

    activities.insert(0, activity)
    _save("activities.json", activities)

    rag.upsert(
        f"activity-{new_id}",
        f"Activity: {activity['activity']} | Client: {activity['client']} | "
        f"Date: {activity['date']} | Description: {activity['description']}",
        {"type": "activity", "id": new_id},
    )

    return jsonify(activity), 201

@app.route("/api/activities/<activity_id>", methods=["DELETE"])
def delete_activity(activity_id):
    activities = _load("activities.json")
    activities = [a for a in activities if a["id"] != activity_id]
    _save("activities.json", activities)
    return jsonify({"ok": True})

# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    tasks = _load("tasks.json")
    status_filter = request.args.get("status")
    if status_filter:
        tasks = [t for t in tasks if t.get("status") == status_filter]
    return jsonify(tasks)

@app.route("/api/tasks", methods=["POST"])
def create_task():
    body = request.get_json(force=True)
    tasks = _load("tasks.json")

    new_id = f"task-{str(uuid.uuid4())[:8]}"
    task = {
        "id": new_id,
        "title": body.get("title", "New Task"),
        "client": body.get("client", ""),
        "project": body.get("project", ""),
        "priority": body.get("priority", "medium"),
        "status": body.get("status", "pending"),
        "due_date": body.get("due_date", ""),
        "assigned_to": body.get("assigned_to", "Denni Alwin"),
        "description": body.get("description", ""),
    }

    tasks.insert(0, task)
    _save("tasks.json", tasks)

    rag.upsert(
        f"task-{new_id}",
        f"Task: {task['title']} | Client: {task['client']} | Status: {task['status']}",
        {"type": "task", "id": new_id},
    )

    return jsonify(task), 201

@app.route("/api/tasks/<task_id>", methods=["PATCH"])
def update_task(task_id):
    body = request.get_json(force=True)
    tasks = _load("tasks.json")

    for task in tasks:
        if task["id"] == task_id:
            for key in ("status", "priority", "title", "description", "due_date"):
                if key in body:
                    task[key] = body[key]
            break

    _save("tasks.json", tasks)
    return jsonify({"ok": True})

@app.route("/api/tasks/<task_id>", methods=["DELETE"])
def delete_task(task_id):
    tasks = _load("tasks.json")
    tasks = [t for t in tasks if t["id"] != task_id]
    _save("tasks.json", tasks)
    return jsonify({"ok": True})

# ── Inventory ─────────────────────────────────────────────────────────────────

@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    inventory = _load("inventory.json")
    category = request.args.get("category")
    if category:
        inventory = [i for i in inventory if i.get("category") == category]
    return jsonify(inventory)

@app.route("/api/inventory", methods=["POST"])
def create_inventory():
    body = request.get_json(force=True)
    inventory = _load("inventory.json")

    new_id = body.get("sku", f"ITEM-{str(uuid.uuid4())[:6].upper()}")
    item = {
        "id": new_id,
        "name": body.get("name", "New Item"),
        "sku": body.get("sku", new_id),
        "category": body.get("category", "General"),
        "stock": int(body.get("stock", 0)),
        "unit": body.get("unit", "pcs"),
        "status": _stock_status(int(body.get("stock", 0)), int(body.get("min_stock", 5))),
        "min_stock": int(body.get("min_stock", 5)),
        "price": float(body.get("price", 0)),
        "movement": 0,
        "location": body.get("location", ""),
    }

    inventory.append(item)
    _save("inventory.json", inventory)
    rag.upsert(
        f"inventory-{new_id}",
        f"Inventory: {item['name']} | SKU: {item['sku']} | Stock: {item['stock']} {item['unit']}",
        {"type": "inventory", "id": new_id},
    )
    return jsonify(item), 201

@app.route("/api/inventory/<item_id>", methods=["PATCH"])
def update_inventory(item_id):
    body = request.get_json(force=True)
    inventory = _load("inventory.json")

    for item in inventory:
        if item["id"] == item_id:
            if "stock" in body:
                delta = int(body["stock"]) - item["stock"]
                item["stock"] = int(body["stock"])
                item["movement"] = delta
                item["status"] = _stock_status(item["stock"], item.get("min_stock", 5))
            for key in ("name", "category", "unit", "min_stock", "price", "location"):
                if key in body:
                    item[key] = body[key]
            break

    _save("inventory.json", inventory)
    return jsonify({"ok": True})

def _stock_status(stock: int, min_stock: int) -> str:
    if stock == 0:
        return "out_of_stock"
    if stock <= min_stock:
        return "critical" if stock <= max(1, min_stock // 2) else "low_stock"
    return "in_stock"

# ── Projects ──────────────────────────────────────────────────────────────────

@app.route("/api/projects", methods=["GET"])
def get_projects():
    return jsonify(_load("projects.json"))

# ── Search / Memory ───────────────────────────────────────────────────────────

@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    results = rag.search(query, n=8)
    return jsonify(results)

# ── Stats ─────────────────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    activities = _load("activities.json")
    tasks = _load("tasks.json")
    inventory = _load("inventory.json")
    projects = _load("projects.json")

    low_stock = [i for i in inventory if i.get("status") in ("low_stock", "critical", "out_of_stock")]
    total_hours = sum(a.get("hours", 0) for a in activities)
    clients = list({a["client"] for a in activities})

    task_counts = {"pending": 0, "in_progress": 0, "completed": 0}
    for t in tasks:
        s = t.get("status", "pending")
        if s in task_counts:
            task_counts[s] += 1

    return jsonify({
        "total_activities": len(activities),
        "total_tasks": len(tasks),
        "active_tasks": task_counts["in_progress"],
        "pending_tasks": task_counts["pending"],
        "completed_tasks": task_counts["completed"],
        "total_inventory": len(inventory),
        "low_stock_items": len(low_stock),
        "total_projects": len(projects),
        "active_projects": len([p for p in projects if p.get("status") in ("in_progress", "active")]),
        "total_hours": total_hours,
        "total_clients": len(clients),
        "vector_docs": rag.collection.count(),
        "ai_ready": rag.llm_ready,
    })

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 8080))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    print(f"[Kroniq] Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
