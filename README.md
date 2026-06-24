<div align="center">

# ⚡ KRONIQ
### AI Business Memory OS

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-API-black?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-green?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-orange?style=for-the-badge)

**Turn business operations into searchable AI memory.**

</div>

---

## 🚀 Features

- 🧠 AI-Powered Business Memory (RAG)
- 🔍 Semantic Search with ChromaDB
- 📦 Inventory Management
- 📋 Task & Project Tracking
- ⚡ Real-Time AI Chat Streaming
- 🤖 Tool Calling (AI can update real data)

---

## 🏗 Architecture

```mermaid
flowchart LR
User --> Frontend
Frontend --> Flask
Flask --> RAG
RAG --> ChromaDB
RAG --> Ollama
Ollama --> Tools
Tools --> JSONData
```

---

## 🛠 Tech Stack

- Python + Flask
- ChromaDB
- Ollama (Qwen2.5)
- Sentence Transformers
- Vanilla JavaScript
- JSON Storage

---

## 📂 Structure

```bash
├── app.py
├── rag.py
├── templates/
│   └── index.html
├── data/
│   ├── activities.json
│   ├── tasks.json
│   ├── inventory.json
│   └── projects.json
└── chroma_db/
```

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt

ollama serve
ollama pull qwen2.5

python app.py
```

Open:

```text
http://localhost:8080
```

---

## 🤖 Example

**You:**  
> Add 12 more Hikvision cameras to stock

**Kroniq:**  
✔ Updates inventory automatically using AI Tool Calling.

---

<div align="center">

### Built for modern service businesses

**AI • RAG • Inventory • Tasks • Memory**

</div>
