================================================================================
KRONIQ — AI BUSINESS MEMORY OS
Complete Project Documentation
================================================================================
Client     : ShipFactory (IT Services — CCTV, Networking, Servers, Cabling)
Built by   : Denni Alwin
Stack      : Python · Flask · ChromaDB · Ollama (Qwen2.5) · sentence-transformers
Date       : June 2026
================================================================================


TABLE OF CONTENTS
-----------------
1.  What Is Kroniq
2.  Core Technology Concepts
3.  Tech Stack (All Libraries and Tools)
4.  Project File Structure
5.  Module Breakdown — rag.py
6.  Module Breakdown — app.py
7.  Module Breakdown — index.html (Frontend SPA)
8.  Data Layer — JSON Files
9.  Complete Request Flow (End-to-End)
10. AI Chat Flow with Tool Calling (The Most Complex Path)
11. RAG Flow Explained Step by Step
12. Tool Calling: How AI Edits Real Data
13. API Endpoints Reference
14. Frontend Views and JavaScript Modules
15. How to Start the Server
16. Configuration and Environment Variables
17. Known Gotchas and Design Decisions


================================================================================
1. WHAT IS KRONIQ
================================================================================

Kroniq is an AI-powered business memory and operations platform built for
ShipFactory, an IT services company that does CCTV installations, network
setups, server maintenance, and cable infrastructure work.

The core idea: instead of using spreadsheets and paper to track jobs done,
inventory used, and tasks, you talk to an AI assistant (in plain English) and
it reads and writes your business data for you.

Key capabilities:
  - Log completed work activities (who, what, where, when, how long)
  - Track task/work orders with priorities and statuses
  - Manage inventory with real-time stock levels and movement tracking
  - Semantic memory search — "find all CCTV work we did at ABC Industries"
  - AI chat that can ACTUALLY edit data (not just answer questions)
  - Full dashboard with stats, charts, and alerts

The AI is not just a chatbot — it uses Tool Calling to directly create tasks,
update inventory, and log activities into the real data files when you ask it.


================================================================================
2. CORE TECHNOLOGY CONCEPTS
================================================================================

RAG (Retrieval-Augmented Generation)
--------------------------------------
RAG means the AI doesn't just answer from its own training — it first searches
your business data and injects the most relevant records into its context before
answering. This means the AI knows about your specific clients, jobs, stock, etc.

How it works in Kroniq:
  Step 1: User sends a message to the chat.
  Step 2: The message is converted to a vector (a list of 384 numbers) using
          the sentence-transformers model (all-MiniLM-L6-v2).
  Step 3: ChromaDB searches for the 5 most similar vectors in its store
          (using cosine similarity).
  Step 4: The matching business records (activities, tasks, inventory entries)
          are prepended to the user's message as context.
  Step 5: Ollama (Qwen2.5) receives the enriched message and responds.

Vector Embeddings
------------------
A vector embedding converts text to numbers. Sentences that mean similar things
produce similar vectors. "CCTV installation at ABC" and "camera setup for ABC
Industries" will be close in vector space. ChromaDB uses these to find relevant
records without exact keyword matching.

Tool Calling
-------------
Ollama supports function/tool calling — the model can respond by saying
"call this function with these arguments" instead of just text. Kroniq defines
5 tools: create_task, update_task_status, log_activity, update_inventory_stock,
and add_inventory_item. When you say "add 12 more Hikvision cameras", the AI
calls update_inventory_stock({sku: "HIK-DS-2CD2143G2-I", delta: 12}) and the
Python code actually writes to inventory.json.

Server-Sent Events (SSE)
--------------------------
The AI response streams character by character to the browser using SSE
(text/event-stream). This is what makes the response appear to "type out" in
real time. Each event is a JSON object: {"text": "chunk"} or {"refresh": true}.
The {"refresh": true} event tells the frontend to reload all data from the
server (triggered after a tool call modifies data).


================================================================================
3. TECH STACK (ALL LIBRARIES AND TOOLS)
================================================================================

BACKEND:
  Python 3.9+              - Language
  Flask 3.0.3              - Web framework (HTTP server, routing, SSE streaming)
  flask-cors 4.0.1         - Allows browser to call the API from any origin
  ollama 0.4.7             - Python client for local Ollama AI server
  chromadb 0.5.23          - Vector database (stores and searches embeddings)
  sentence-transformers 3.3.1 - Converts text to vector embeddings (all-MiniLM-L6-v2 model)
  python-dotenv 1.0.1      - Loads .env file into environment variables
  gunicorn 23.0.0          - Production WSGI server (alternative to Flask dev server)

AI MODEL:
  Ollama                   - Local AI server (runs on your machine at localhost:11434)
  qwen2.5:latest           - The LLM model (4.68GB, pulled via: ollama pull qwen2.5)
  all-MiniLM-L6-v2         - Embedding model (downloaded by sentence-transformers, ~80MB)

FRONTEND:
  Vanilla JavaScript       - No framework (React/Vue not used)
  HTML5 / CSS3             - Single-page app in one template file
  Fetch API                - For HTTP calls to the Flask backend
  ReadableStream API       - For consuming the SSE stream from /api/chat
  Google Fonts             - JetBrains Mono (monospace) + Space Grotesk (UI)

DATA STORAGE:
  JSON files               - data/activities.json, tasks.json, inventory.json, projects.json
  ChromaDB (persistent)    - ./chroma_db/ directory — stores vector embeddings for semantic search

DEVELOPMENT TOOLS:
  .env / .env.example      - Environment configuration
  start.sh                 - Convenience startup script


================================================================================
4. PROJECT FILE STRUCTURE
================================================================================

python-prod/
│
├── app.py                  ← Flask web server, API routes, tool executor
├── rag.py                  ← RAG engine, vector search, AI chat, tool definitions
├── requirements.txt        ← Python dependencies
├── start.sh                ← Startup script (loads .env, runs app.py)
├── .env.example            ← Environment variable template
│
├── templates/
│   └── index.html          ← Complete frontend SPA (HTML + CSS + JavaScript, ~1145 lines)
│
├── data/
│   ├── activities.json     ← Work activity history
│   ├── tasks.json          ← Tasks / work orders
│   ├── inventory.json      ← Equipment / stock inventory
│   └── projects.json       ← Client project records
│
└── chroma_db/              ← ChromaDB persistent vector store (auto-created)
    └── (binary files)


File roles summary:
  rag.py      The brain. Handles embeddings, vector search, and AI conversation.
  app.py      The server. Exposes HTTP endpoints, reads/writes JSON files, calls rag.py.
  index.html  The UI. A single-page app that calls app.py's API endpoints.
  data/*.json The database. Plain JSON files, human-readable and directly editable.
  chroma_db/  The memory index. ChromaDB's persistent storage for vector similarity search.


================================================================================
5. MODULE BREAKDOWN — rag.py
================================================================================

Purpose: Core intelligence layer. Manages vector embeddings, ChromaDB, and
         the AI conversation with tool calling support.

CONSTANTS:
----------
  REFRESH_SENTINEL = "\x00REFRESH\x00"
    A special string yielded by chat_stream() after a tool call modifies data.
    app.py detects this sentinel and sends {"refresh": true} to the frontend,
    which then reloads all views to show the updated data.

  OLLAMA_HOST = "http://localhost:11434"
    Where the local Ollama server is running.

  OLLAMA_MODEL = "qwen2.5"
    The model name to use for chat (changeable via OLLAMA_MODEL env var).

SYSTEM_PROMPT:
--------------
  A multi-line string sent as the first message in every conversation. It tells
  Qwen who it is (Kroniq), what tools it has, that it should call tools
  immediately when asked to create/update data, and that IDs/SKUs come from
  [System Data] in the user's message.

TOOLS list:
-----------
  A list of 5 tool definitions in Ollama's JSON schema format:

  1. create_task
     Creates a new task/work order.
     Required: title
     Optional: client, project, priority (low/medium/high), due_date (YYYY-MM-DD),
               description

  2. update_task_status
     Changes the status of an existing task.
     Required: task_id (exact ID from [System Data]), status (pending/in_progress/completed)

  3. log_activity
     Logs a completed work activity to business memory.
     Required: activity, client
     Optional: date, hours, project, description, items_used (array of strings)

  4. update_inventory_stock
     Updates stock of an EXISTING item.
     Required: sku (exact SKU from [System Data])
     Optional: delta (add/remove relative to current stock — "add 12" → delta=12)
               new_stock (set to absolute value — "set to 20" → new_stock=20)

  5. add_inventory_item
     Adds a completely NEW item to inventory.
     Required: name, sku, category, stock
     Optional: unit, min_stock, price

  NOTE: The descriptions are carefully worded to prevent the AI from calling
  the wrong tool. For example: "update_inventory_stock: NEVER use this to create
  a new item" and "add_inventory_item: Do NOT use for existing items."

CLASS: KroniqRAG
-----------------
  __init__(self):
    - Initializes ChromaDB persistent client at "./chroma_db"
    - Gets or creates the "kroniq_memory" collection (cosine similarity space)
    - Loads all-MiniLM-L6-v2 from sentence-transformers
    - Creates an Ollama client and checks connectivity
    - Sets self.llm_ready = True/False

  _embed(self, text: str) -> list:
    Converts a string to a 384-dimensional float vector using sentence-transformers.
    normalize_embeddings=True ensures vectors have unit length (required for cosine).

  upsert(self, doc_id: str, text: str, metadata: dict):
    Inserts or updates a document in ChromaDB.
    - Checks if doc_id exists (collection.get)
    - Updates if exists, adds if not
    - Always passes pre-computed embeddings (not query_texts) to avoid ChromaDB
      trying to download its own ONNX embedding model

  search(self, query: str, n: int = 5) -> list:
    Semantic search. Embeds the query, then asks ChromaDB for the n closest docs.
    Returns list of {"text": str, "meta": dict, "score": float 0-1}
    Score = 1 - cosine_distance (1.0 = identical, 0.0 = unrelated)

  index_all(self, activities, tasks, inventory, projects):
    Called once at startup. Converts all business records to text strings and
    upserts them into ChromaDB. Format examples:
      Activity: "Activity: CCTV Installation | Client: ABC Industries | Date: 2026-06-20..."
      Task:      "Task: Install cameras | Client: ABC | Status: pending | Priority: high..."
      Inventory: "Inventory: Hikvision 4MP IP Camera | SKU: HIK-DS-2CD2143G2-I | Stock: 20..."
      Project:   "Project: Security Upgrade 2026 | Client: ABC Industries | Status: in_progress..."

  chat_stream(self, message, history, tool_executor=None, data_context=""):
    THE MAIN AI FUNCTION. Yields text chunks (and possibly REFRESH_SENTINEL).
    See Section 10 for the full detailed flow.


================================================================================
6. MODULE BREAKDOWN — app.py
================================================================================

Purpose: Flask HTTP server. Exposes all API endpoints, manages JSON file I/O,
         defines the tool executor, and serves the frontend template.

GLOBAL SETUP:
-------------
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    Absolute path to the data/ directory. All JSON reads/writes use this.

  rag = KroniqRAG()
    The RAG engine is instantiated once at startup (loads model, connects ChromaDB).

  rag.index_all(...)
    Called immediately after rag is created. Seeds ChromaDB with all current data.

HELPER FUNCTIONS:
-----------------
  _load(filename) -> list:
    Reads a JSON file from DATA_DIR. Returns [] if file doesn't exist.

  _save(filename, data):
    Writes a list to a JSON file in DATA_DIR with indent=2 formatting.
    Creates DATA_DIR if it doesn't exist (os.makedirs with exist_ok=True).

  _stock_status(stock, min_stock) -> str:
    Computes the stock status badge:
      stock == 0                      -> "out_of_stock"
      stock <= max(1, min_stock // 2) -> "critical"
      stock <= min_stock              -> "low_stock"
      else                            -> "in_stock"

  _build_data_context() -> str:
    Builds a compact text block injected into every AI chat message.
    Includes the first 15 tasks (id, title, status, client) and first 15
    inventory items (SKU, name, stock, status). This is how the AI knows
    exact IDs and SKUs to pass to tool calls.
    Example output:
      [System Data — use exact IDs/SKUs when calling update tools]
      Tasks:
        task-4c7f42a1: CCTV Installation at ABC [in_progress] - ABC Industries
      Inventory:
        SKU=HIK-DS-2CD2143G2-I: Hikvision 4MP IP Camera stock=20pcs [in_stock]

TOOL EXECUTOR FUNCTION:
-----------------------
  _tool_executor(fn_name: str, args: dict) -> str (JSON):
    Called by rag.py's chat_stream() when the AI decides to use a tool.
    Wrapped in try/except — returns {"ok": false, "message": "..."} on any error.

    create_task:
      Generates UUID-based ID (task-XXXXXXXX).
      Inserts task dict at index 0 of tasks.json (newest first).
      Calls rag.upsert() to add the new task to vector memory.
      Returns {"ok": true, "id": "task-...", "message": "Task '...' created"}

    update_task_status:
      Finds task by ID in tasks.json.
      Updates its "status" field.
      Saves tasks.json.
      Returns {"ok": true, "message": "Task '...' updated to in_progress"}

    log_activity:
      Generates UUID-based ID (act-XXXXXXXX).
      Inserts activity dict at index 0 of activities.json.
      Calls rag.upsert() to add to vector memory.
      Returns {"ok": true, "id": "act-...", "message": "Activity '...' logged"}

    update_inventory_stock:
      Finds item by sku OR id field.
      If delta given: new_stock = max(0, old + delta)
      If new_stock given: stock = max(0, new_stock)
      Updates movement = new_stock - old_stock
      Recomputes status via _stock_status()
      Saves inventory.json.
      Returns {"ok": true, "message": "Stock for ... updated from X to Y"}

    add_inventory_item:
      SAFETY NET: if SKU already exists, redirects to update_inventory_stock
                  with new_stock (prevents duplicate entries).
      Otherwise: creates new item dict, appends to inventory.json.
      Calls rag.upsert() to add to vector memory.
      Returns {"ok": true, "message": "Item '...' added to inventory"}

AI CHAT ENDPOINT:
-----------------
  POST /api/chat
    Body: {"message": "...", "history": [...]}
    Response: text/event-stream (SSE)

    Events:
      data: {"text": "chunk"}     — AI text chunk (stream)
      data: {"refresh": true}     — Tool was used, reload all data
      data: [DONE]                — Stream finished

    Internally calls rag.chat_stream() with _tool_executor and data_context.
    The generate() inner function converts REFRESH_SENTINEL to {"refresh": true}.

REST API ENDPOINTS:
-------------------
  GET  /api/stats           — Dashboard stats (counts, totals, ai_ready flag)
  GET  /api/activities      — All activities sorted by date desc
  POST /api/activities      — Create new activity
  DELETE /api/activities/<id> — Delete activity
  GET  /api/tasks           — All tasks (optional ?status=pending filter)
  POST /api/tasks           — Create new task
  PATCH /api/tasks/<id>     — Update task fields
  DELETE /api/tasks/<id>    — Delete task
  GET  /api/inventory       — All inventory (optional ?category=CCTV filter)
  POST /api/inventory       — Add new inventory item
  PATCH /api/inventory/<id> — Update inventory item (stock, name, etc.)
  GET  /api/projects        — All projects
  GET  /api/search?q=...    — Semantic search via RAG (returns relevance-ranked results)
  GET  /                    — Serves the frontend (index.html)


================================================================================
7. MODULE BREAKDOWN — index.html (Frontend SPA)
================================================================================

Purpose: The complete frontend. A single HTML file (~1145 lines) containing all
         HTML structure, CSS styles, and JavaScript in one file. No build tool,
         no bundler, no framework — just browser-native code.

CSS DESIGN SYSTEM:
------------------
  Color palette (CSS custom properties):
    --bg: #16181E           (Dark background)
    --surface: #1E2029      (Card/panel background)
    --surface2: #252832     (Slightly lighter surface)
    --surface3: #2D3047     (Borders, hover states)
    --accent: #C8F24E       (Lime green — primary accent color)
    --text: #FFFFFF         (Primary text)
    --text2: #9CA3AF        (Secondary text)
    --text3: #6B7280        (Muted text)
    --red: #FF5B5B          (Danger, out-of-stock)
    --yellow: #F5C518       (Warning, low stock, pending)
    --green: #4ADE80        (Success, in-stock, completed)
    --blue: #60A5FA         (Info, in-progress)

  Typography:
    --mono: 'JetBrains Mono' (numbers, SKUs, dates, code-like data)
    --sans: 'Space Grotesk'  (all UI text)

  Layout: Flexbox sidebar + scrollable main area
    Sidebar: 220px fixed width, dark surface color
    Main: flex:1, overflow-y scroll

VIEWS (sections shown/hidden):
-------------------------------
  #view-workspace   — Dashboard home with stats, recent activities, open tasks, low stock alert
  #view-assistant   — Full AI chat interface with streaming response
  #view-tasks       — Task management with filter tabs and task cards
  #view-inventory   — Inventory table with search, category filter, stock bars
  #view-memory      — Activity timeline with semantic search panel
  #view-overview    — Analytics with client bar chart and top clients list

  showView(name):
    Hides all views, shows the selected one.
    Also triggers fresh data loads from the server (e.g., showView('inventory')
    calls loadInventory() so you always see current data).

JAVASCRIPT STATE:
-----------------
  let allActivities = []    — Cached activities from last API call
  let allTasks = []         — Cached tasks
  let allInventory = []     — Cached inventory
  let statsData = {}        — Cached stats from /api/stats
  let chatHistory = []      — Chat conversation history (sent to AI for context)
  let currentFilter = 'all' — Active task filter tab

CORE JAVASCRIPT FUNCTIONS:
---------------------------
  loadAll():
    Calls loadStats(), loadActivities(), loadTasks(), loadInventory() in parallel
    using Promise.all(). Called on DOMContentLoaded and after AI tool calls.

  api(path, opts):
    Thin wrapper around fetch() that sets Content-Type: application/json and
    auto-serializes the body with JSON.stringify.

  loadStats() → updates stat cards in DOM
  loadActivities() → sets allActivities, calls renderWorkspace() + renderTimeline()
  loadTasks() → sets allTasks, calls renderTasks() + renderWorkspace()
  loadInventory() → sets allInventory, calls renderInventory() + renderLowStock()

  sendChat():
    The main chat handler.
    1. Reads input, adds user bubble to chat UI.
    2. Appends to chatHistory.
    3. POSTs to /api/chat with message + last 12 history items.
    4. Reads SSE stream with fetch + ReadableStream.
    5. On {"text": chunk}: appends to fullText, updates the typing bubble.
    6. On {"refresh": true}: calls loadAll() to reload all data immediately.
    7. On [DONE]: enables send button.

  quickAsk():
    Same as sendChat() but displays response in the small "Quick Ask" box on
    the Workspace view (not the full chat panel). Useful for fast one-off queries.

  renderInventoryTable(items):
    Renders the inventory table. For each item:
    - Calculates stock percentage: stock / (min_stock * 2) capped at 100%
    - Renders a colored bar (green/yellow/red) using inline CSS width
    - Shows movement delta with +/- color coding

  toggleTask(id, currentStatus):
    Toggles a task between pending and completed via PATCH /api/tasks/<id>.
    Updates the local allTasks array and re-renders without a full reload.

  searchMemory():
    Calls GET /api/search?q=... and renders results with relevance percentages.
    This directly uses the ChromaDB semantic search, not keyword matching.

  openModal(name) / closeModal(name):
    Shows/hides modal overlays. Modals are always present in DOM but hidden.

  showToast(msg, type):
    Shows a temporary bottom-right notification for 3 seconds.

BADGE HELPERS:
--------------
  priorityBadge(p) → red (high), yellow (medium), gray (low)
  statusBadge(s)   → green (completed), blue (in_progress), yellow (pending)
  stockBadge(s)    → green (in_stock), yellow (low_stock), red (critical/out_of_stock)


================================================================================
8. DATA LAYER — JSON FILES
================================================================================

All business data is stored as plain JSON files in the data/ directory.
No database server is needed. Files are read on every API request and written
on every create/update/delete operation. This is intentional for the MVP — simple,
transparent, and zero infrastructure overhead.

data/inventory.json — Array of inventory item objects:
  {
    "id": "HIK-DS-2CD2143G2-I",    ← same as SKU (used as primary key)
    "name": "Hikvision 4MP IP Camera",
    "sku": "HIK-DS-2CD2143G2-I",
    "category": "CCTV",            ← CCTV | Networking | Cables | Servers | Power | Tools | Accessories
    "stock": 20,
    "unit": "pcs",                 ← pcs | rolls | meters | sets
    "status": "in_stock",          ← in_stock | low_stock | critical | out_of_stock
    "min_stock": 10,               ← threshold for low stock alert
    "price": 4500,                 ← unit price in ₹
    "movement": -4,                ← last stock change (positive = received, negative = used)
    "location": "Shelf A1"
  }

data/tasks.json — Array of task objects:
  {
    "id": "task-4c7f42a1",
    "title": "CCTV Installation at XYZ Corp",
    "client": "XYZ Corp",
    "project": "Network Overhaul Phase 2",
    "priority": "high",            ← low | medium | high
    "status": "pending",           ← pending | in_progress | completed
    "due_date": "2026-06-30",
    "assigned_to": "Denni Alwin",
    "description": "..."
  }

data/activities.json — Array of completed work activity objects:
  {
    "id": "act-5017b681",
    "date": "2026-06-20",
    "activity": "CCTV Installation",
    "client": "ABC Industries",
    "project": "Security Upgrade 2026",
    "technician": "Denni Alwin",
    "items_used": ["HIK-DS-2CD2143G2-I", "CAB-CAT6-100M"],
    "description": "Installed 4 cameras...",
    "status": "completed",
    "hours": 6
  }

data/projects.json — Array of project objects:
  {
    "id": "proj-001",
    "name": "Security Upgrade 2026",
    "client": "ABC Industries",
    "status": "in_progress",       ← planning | in_progress | completed | on_hold
    "start_date": "2026-05-15",
    "end_date": "2026-07-31",
    "budget": 250000,
    "activities": ["act-001", "act-006"],
    "description": "..."
  }

ChromaDB (chroma_db/ directory):
  Not JSON — ChromaDB's binary format. Stores one vector per record.
  Collection name: "kroniq_memory"
  Distance metric: cosine
  Each document has:
    id: "activity-act-001" | "task-task-001" | "inventory-HIK-..." | "project-proj-001"
    embedding: [384 floats]
    document: text representation of the record
    metadata: {"type": "activity", "id": "act-001"}


================================================================================
9. COMPLETE REQUEST FLOW (END-TO-END)
================================================================================

EXAMPLE: User opens the browser and goes to Inventory view.

1. Browser → GET http://localhost:8080/
   Flask returns templates/index.html

2. Browser executes DOMContentLoaded in index.html
   → Calls loadAll()
   → Fires 4 parallel fetch calls:
       GET /api/stats
       GET /api/activities
       GET /api/tasks
       GET /api/inventory

3. GET /api/inventory → app.py:get_inventory()
   → _load("inventory.json")
   → Reads data/inventory.json from disk
   → Returns JSON array

4. Browser receives inventory array
   → Sets allInventory = [...]
   → Calls renderInventory() → filterInventory() → renderInventoryTable(items)
   → Each item is rendered as a <tr> with stock bar, movement badge, etc.

5. User clicks "Edit" on a row → openStockEdit(id, name, stock) → modal opens

6. User enters new stock value → clicks "Update Stock"
   → submitStockEdit() fires
   → PATCH /api/inventory/<id> with body: {"stock": 25}

7. app.py:update_inventory() receives the PATCH
   → _load("inventory.json")
   → Finds item by id
   → Updates stock, calculates movement delta, recomputes status
   → _save("inventory.json", inventory)
   → Returns {"ok": true}

8. Browser receives {"ok": true}
   → Updates local allInventory array in place (no full reload needed)
   → Re-renders inventory table
   → Shows toast "Stock updated"


================================================================================
10. AI CHAT FLOW WITH TOOL CALLING (THE MOST COMPLEX PATH)
================================================================================

EXAMPLE: User types "add 12 more Hikvision 4MP IP Camera to stock"

--- BROWSER SIDE ---

1. User presses Enter in #chatInput
2. sendChat() runs:
   a. appendMessage('user', "add 12 more...")  — shows user bubble immediately
   b. chatHistory.push({role:'user', content:'add 12 more...'})
   c. appendMessage('ai', typing-dots, id='typing-1234')  — shows loading indicator
   d. fetch POST /api/chat with body:
      {
        "message": "add 12 more Hikvision 4MP IP Camera to stock",
        "history": [...]
      }
   e. Opens SSE stream reader, loops over chunks

--- SERVER SIDE: app.py ---

3. app.py /api/chat endpoint receives the request:
   a. Extracts message and history from JSON body
   b. Calls _build_data_context() — builds the [System Data] block:
      "[System Data...]\nInventory:\n  SKU=HIK-DS-2CD2143G2-I: Hikvision 4MP IP Camera stock=20pcs [in_stock]\n..."
   c. Calls rag.chat_stream(message, history, tool_executor=_tool_executor, data_context=data_ctx)
   d. Wraps it in a Flask SSE Response with generate() generator

--- SERVER SIDE: rag.py ---

4. chat_stream() begins:
   a. Calls self.search("add 12 more Hikvision 4MP IP Camera") — RAG lookup
      - Embeds the query to a 384-dim vector
      - ChromaDB finds 5 most similar records by cosine distance
      - Returns relevant records (probably the Hikvision inventory entry)
   b. Builds rag_ctx string from returned docs:
      "[Relevant Business Memory]\n• Inventory: Hikvision 4MP IP Camera | SKU: HIK-DS-2CD2143G2-I | Stock: 20..."
   c. Builds the messages array:
      [
        {"role": "system", "content": SYSTEM_PROMPT},
        ...last 8 chat history items...
        {"role": "user", "content":
          "add 12 more Hikvision 4MP IP Camera to stock"
          + rag_ctx
          + data_context   ← includes the [System Data] block with exact SKUs
        }
      ]

5. STAGE 1: Non-streaming API call to Ollama:
   response = ollama.chat(model="qwen2.5", messages=messages, tools=TOOLS)

   Qwen reads the message and [System Data], sees SKU=HIK-DS-2CD2143G2-I,
   and decides to call update_inventory_stock.

   response.message.tool_calls = [
     ToolCall(function=Function(
       name='update_inventory_stock',
       arguments={'sku': 'HIK-DS-2CD2143G2-I', 'delta': 12}
     ))
   ]

6. Tool calls detected → enter the tool execution branch:
   a. Build assistant_entry with the tool_calls
   b. Append assistant_entry to messages (conversation memory)
   c. For each tool call:
      result = tool_executor('update_inventory_stock', {'sku': 'HIK-DS-2CD2143G2-I', 'delta': 12})

      This calls _tool_executor() in app.py:
        - _load("inventory.json")
        - Finds Hikvision item (sku matches)
        - old = 20
        - new_stock = max(0, 20 + 12) = 32
        - item["movement"] = 32 - 20 = 12
        - item["status"] = _stock_status(32, 10) = "in_stock"
        - _save("inventory.json", inventory)  ← FILE IS WRITTEN HERE
        - returns '{"ok": true, "message": "Stock for Hikvision 4MP IP Camera updated from 20 to 32"}'

   d. Append tool result to messages:
      {"role": "tool", "content": '{"ok": true, "message": "...updated from 20 to 32"}'}

7. STAGE 2: Streaming call to Ollama with tool result in context:
   stream = ollama.chat(model="qwen2.5", messages=messages, stream=True)
   Qwen sees the tool result and generates a confirmation text:
   "I've added 12 more Hikvision 4MP IP Cameras to the stock. The stock has been
    updated from 20 to 32 units."

   Each chunk is yielded from chat_stream():
     yield "I've"
     yield " added"
     yield " 12"
     ...

8. After the stream ends:
   yield REFRESH_SENTINEL  ← "\x00REFRESH\x00"

--- BACK IN app.py ---

9. generate() receives REFRESH_SENTINEL from the generator:
   → yield f"data: {json.dumps({'refresh': True})}\n\n"
   → yield "data: [DONE]\n\n"

--- BACK IN BROWSER ---

10. SSE reader processes events:
    - {"text": "I've"} → fullText += "I've" → updateMessage(typingId, "I've")
    - {"text": " added"} → ... (types out progressively)
    - {"refresh": true} → loadAll()  ← RELOADS ALL DATA FROM SERVER
      → loadInventory() fetches /api/inventory
      → Renders inventory table with Hikvision stock now showing 32
    - [DONE] → enables send button

User sees: the AI typing out its confirmation, and simultaneously the
inventory page updates to show the new stock value.


================================================================================
11. RAG FLOW EXPLAINED STEP BY STEP
================================================================================

Setup (startup):
  1. app.py calls rag.index_all(activities, tasks, inventory, projects)
  2. For each record, a text string is built (human-readable description)
  3. sentence-transformers encodes the string to a 384-dim float vector
  4. ChromaDB stores: (doc_id, vector, text, metadata)
  5. When a new item is created via API, rag.upsert() is called immediately to
     keep the vector store in sync

Query (every chat message):
  1. User sends: "What CCTV work did we do for ABC Industries?"
  2. rag.search("What CCTV work...") embeds the query → 384-dim vector
  3. ChromaDB queries for 5 nearest vectors (cosine similarity)
  4. ChromaDB returns the closest activity and task records
  5. The text of these records is prepended to the user's message as context
  6. Qwen receives: user_message + relevant_memory + system_data → answers with context

Why this matters:
  Without RAG, Qwen knows nothing about ShipFactory's specific jobs, clients, and
  inventory. With RAG, the AI's answer is grounded in real business data — it can
  say "Yes, you installed 4 cameras at ABC Industries on June 20, 2026 using SKU
  HIK-DS-2CD2143G2-I as part of the Security Upgrade 2026 project."


================================================================================
12. TOOL CALLING: HOW AI EDITS REAL DATA
================================================================================

Why tool calling and not just prompting:
  Early versions of Kroniq told the AI to reply with instructions like
  "Here's how to add inventory: click the inventory page...". This was useless.
  Tool calling makes the AI an actor that directly writes to the database.

How it works technically:
  Ollama's tool calling follows the OpenAI function calling spec. You pass a
  list of "tools" (JSON schema describing function names and parameters).
  The model doesn't run Python code — instead, it returns a structured response
  saying "I want to call function X with arguments Y". Your code then actually
  calls the function and feeds the result back to the model.

Two-stage conversation:
  Stage 1 (non-streaming): Send message + tools → model either returns text OR
                            returns tool_calls (function name + args)
  Execute: Python code runs the tool function, writes to JSON file
  Stage 2 (streaming): Send updated conversation (with tool result) → model
                        generates human-readable confirmation text, streamed live

The safety net (add_inventory_item):
  Early in development, Qwen would sometimes call add_inventory_item when the
  item already existed (because it confused "add 12 more" with "add a new item").
  The safety net in _tool_executor() detects this: if the SKU already exists when
  add_inventory_item is called, it silently redirects to update_inventory_stock
  with the new absolute stock count. This prevents duplicate inventory rows.

Tool description engineering:
  The exact wording of tool descriptions matters. Bad description:
    "update_inventory_stock: Updates stock for an item"
  This caused the model to sometimes call add_inventory_item for existing items.

  Fixed description:
    "update_inventory_stock: Update stock of an EXISTING item already in [System Data].
     Use delta for relative changes ('add 12 more' → delta=12, 'remove 5' → delta=-5)
     or new_stock to set an absolute value ('set to 20' → new_stock=20).
     NEVER use this to create a new item."

  The examples in parentheses ("add 12 more" → delta=12) guide the model to use
  the right parameter type.

The delta vs new_stock distinction:
  delta: relative change. "Add 12 more" → delta=12. "Used 3" → delta=-3.
  new_stock: absolute value. "Set it to 50" → new_stock=50.
  The model picks the right one based on the user's phrasing.


================================================================================
13. API ENDPOINTS REFERENCE
================================================================================

All endpoints return JSON. POST/PATCH bodies are JSON.

STATS
  GET /api/stats
  Response: {
    total_activities, total_tasks, active_tasks, pending_tasks, completed_tasks,
    total_inventory, low_stock_items, total_projects, active_projects,
    total_hours, total_clients, vector_docs, ai_ready
  }

ACTIVITIES
  GET    /api/activities            → Array of all activities (sorted by date desc)
  POST   /api/activities            → Create activity, returns new activity object (201)
         Body: {activity, client, date, hours, project, description, items_used}
  DELETE /api/activities/<id>       → Delete activity, returns {"ok": true}

TASKS
  GET    /api/tasks                 → All tasks (optional: ?status=pending)
  POST   /api/tasks                 → Create task, returns new task object (201)
         Body: {title, client, project, priority, status, due_date, description}
  PATCH  /api/tasks/<id>            → Update task fields
         Body: any subset of {status, priority, title, description, due_date}
  DELETE /api/tasks/<id>            → Delete task

INVENTORY
  GET    /api/inventory             → All items (optional: ?category=CCTV)
  POST   /api/inventory             → Add item, returns new item object (201)
         Body: {name, sku, category, unit, stock, min_stock, price, location}
  PATCH  /api/inventory/<id>        → Update item fields
         Body: any subset of {stock, name, category, unit, min_stock, price, location}

PROJECTS
  GET    /api/projects              → All projects (read-only, no add/edit via API yet)

SEARCH
  GET    /api/search?q=<query>      → Semantic RAG search
  Response: [{text, meta: {type, id}, score}] (up to 8 results, sorted by relevance)

AI CHAT
  POST   /api/chat                  → Chat with AI (SSE streaming response)
  Body: {message: string, history: [{role: 'user'|'assistant', content: string}]}
  Content-Type response: text/event-stream
  Events:
    data: {"text": "chunk"}
    data: {"refresh": true}
    data: [DONE]


================================================================================
14. FRONTEND VIEWS AND JAVASCRIPT MODULES
================================================================================

WORKSPACE VIEW (#view-workspace)
  Loaded by: showView('workspace') or on DOMContentLoaded
  Data: stats, last 6 activities, open tasks, low stock items
  Functions: renderWorkspace(), renderLowStock()
  Also has: Quick Ask — small chat box for fast questions without going to Assistant

ASSISTANT VIEW (#view-assistant)
  Full-screen chat interface with streaming AI responses
  Left panel: chat messages
  Right sidebar: memory stats (counts from /api/stats)
  Functions: sendChat(), appendMessage(), updateMessage(), handleChatKey(), autoResize()
  Chat state: chatHistory[] array maintained across the session

TASKS VIEW (#view-tasks)
  Filter tabs: All | In Progress | Pending | Completed
  Task cards with checkbox, priority badge, status badge, delete button
  Functions: filterTasks(), renderTasks(), toggleTask(), deleteTask(), submitTask()
  Modal: #modal-addTask for creating new tasks

INVENTORY VIEW (#view-inventory)
  Table with search input + category dropdown filter
  Each row: product name, SKU, category, stock count, visual bar, movement delta, status
  Edit button opens #modal-editStock for quick stock updates
  Functions: renderInventory(), filterInventory(), renderInventoryTable(), openStockEdit(), submitStockEdit()
  Modal: #modal-addInventory for adding new items

MEMORY VIEW (#view-memory)
  Timeline of all activities (newest first)
  Semantic search bar that calls /api/search
  Results panel shows matching records with relevance score
  Functions: renderTimeline(), searchMemory(), clearSearch(), submitActivity()
  Modal: #modal-addActivity for manual activity logging

OVERVIEW VIEW (#view-overview)
  Stats row (hours, clients, tasks completed, vector docs)
  Bar chart showing activities per client
  Top clients list with job counts
  Functions: renderOverview() — builds chart from allActivities clientMap

MODALS (always present in DOM, shown/hidden with CSS class):
  #modal-addTask         — New task form
  #modal-addInventory    — Add inventory item form
  #modal-addActivity     — Log activity form
  #modal-editStock       — Quick stock update (just stock number + item name)


================================================================================
15. HOW TO START THE SERVER
================================================================================

Prerequisites:
  1. Python 3.9+ installed
  2. pip install -r requirements.txt  (installs all Python dependencies)
  3. Ollama installed and running: ollama serve
  4. Qwen model pulled: ollama pull qwen2.5

Start the server:

  Option A — using start.sh (recommended):
    chmod +x start.sh
    ./start.sh

  Option B — direct Python:
    python3 app.py

  Option C — with custom port:
    FLASK_PORT=8080 python3 app.py

  Option D — production with gunicorn:
    gunicorn -w 1 -b 0.0.0.0:8080 app:app
    NOTE: Use 1 worker only — the RAG model and ChromaDB are not safe to share
          across forked processes. Multiple workers would cause initialization issues.

Access:
  Open http://localhost:8080 in your browser

First run behavior:
  - sentence-transformers downloads all-MiniLM-L6-v2 (~80MB) on first run
  - ChromaDB creates the chroma_db/ directory
  - All mock data (activities, tasks, inventory, projects) is indexed into ChromaDB
  - Server is ready when you see: "[Kroniq] Starting server on http://localhost:8080"


================================================================================
16. CONFIGURATION AND ENVIRONMENT VARIABLES
================================================================================

Create a .env file (copy from .env.example):

  FLASK_PORT=8080         ← Port the server runs on (default: 8080)
  FLASK_ENV=development   ← Set to "development" to enable Flask debug mode
  OLLAMA_HOST=http://localhost:11434   ← Ollama server address
  OLLAMA_MODEL=qwen2.5                ← Model to use for chat

If you want to change the AI model:
  - Pull a different model: ollama pull llama3.2 (or any Ollama model with tool support)
  - Set OLLAMA_MODEL=llama3.2 in .env
  - Not all models support tool calling — qwen2.5 is tested and recommended


================================================================================
17. KNOWN GOTCHAS AND DESIGN DECISIONS
================================================================================

1. DUPLICATE INVENTORY ENTRIES
   Problem: Before the safety net was added in add_inventory_item, each "add more
   stock" chat command created a new row with the same SKU. The tool would only
   update the first row, making it look like nothing changed.
   Fix: Safety net in _tool_executor redirects add_inventory_item to
   update_inventory_stock if the SKU already exists. Also deduplication was
   run on data/inventory.json to clean existing duplicates.

2. CHROMADB NOT USING ONNX
   Problem: ChromaDB tries to use query_texts which downloads its own 79MB ONNX
   embedding model. This caused errors and slowness on first run.
   Fix: Always pass query_embeddings (pre-computed by sentence-transformers) to
   all ChromaDB operations. ChromaDB never downloads anything.

3. STALE UI DATA
   Problem: After AI tools ran, views showed old data because showView() only
   called render functions (from cached arrays), not fetch functions.
   Fix: showView() now calls load functions (loadInventory(), loadTasks(), etc.)
   which hit the server for fresh data. The {"refresh": true} SSE event also
   triggers loadAll() immediately after any tool modifies data.

4. TOOL SELECTION AMBIGUITY
   Problem: "add 12 more Hikvision cameras" would sometimes trigger
   add_inventory_item (creates new item with stock=12) instead of
   update_inventory_stock (adds 12 to existing stock).
   Fix: Clear tool descriptions with explicit examples, a "delta" parameter
   for relative changes, and the safety net redirect.

5. SINGLE JSON FILE PER COLLECTION
   Design decision: Using flat JSON files instead of SQLite or PostgreSQL.
   Pros: Zero infrastructure, human-readable, directly editable, easy backup.
   Cons: No concurrent write safety (Flask runs single-threaded), full file read
   on every operation, will slow down with thousands of records.
   For MVP with <1000 records per file: completely fine.

6. ONE GUNICORN WORKER
   The KroniqRAG object loads a 80MB ML model into memory. With multiple workers,
   each worker would load its own copy. With 1 worker, the model loads once and
   all requests share it. For a small internal tool this is acceptable.

7. NO AUTHENTICATION
   The app has no login system. It's designed for a single-user internal tool
   on a local network. If exposing publicly, add Flask-Login or a reverse proxy
   with basic auth.

8. OLLAMA MUST BE RUNNING SEPARATELY
   app.py does not start Ollama. If Ollama is not running, the server starts but
   rag.llm_ready = False, and the AI chat shows an error. Start Ollama with:
   ollama serve   (in a separate terminal)

9. HISTORY LIMIT
   Only the last 8 messages from chatHistory are sent to the AI (history[-8:]).
   This prevents the context window from filling up on long conversations.
   The RAG search provides relevant long-term memory, compensating for the
   short conversation window.

10. PORT CONFLICTS
    If port 8080 or 8082 is already in use (from a previous session),
    start the server will fail silently (Flask won't start, but the old server
    still handles requests). Always kill old processes first:
      lsof -ti :8080 | xargs kill -9


================================================================================
END OF DOCUMENTATION
================================================================================
Total files: 11 (excluding chroma_db binaries)
Lines of code: ~440 (app.py) + ~280 (rag.py) + ~1145 (index.html)
Data files: 4 JSON files in data/
Tech stack: 7 Python packages + 1 local AI server + vanilla browser JS
================================================================================
