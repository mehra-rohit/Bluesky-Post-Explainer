# Bluesky Explainer Agent

## 🚀 Overview
Social platforms like Bluesky often host highly specialized communities (e.g., AI Engineering, ATProto developers). Standard LLMs often lack the real-time context to explain viral memes or breaking technical shifts. This project implements a ReAct (Reason + Act) agent that retrieves current context to explain these posts accurately.

## ✨ Features
*   **Autonomous Reasoning**: Uses a ReAct loop to analyze posts and determine what information is missing.
*   **Real-Time Tool Use**: Integrates with search APIs (e.g. DuckDuckGo) to fetch the latest context from 2025-2026.
*   **Citations**: Every bullet point is grounded in retrieved data with source links provided.
*   **Evaluation Harness**: A robust testing module with 10+ benchmark cases to ensure factuality and utility.
*   **Image Understanding**: Leverages Vision models to interpret meme-based context.

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd bluesky-explainer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file or export your keys:

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

## 📂 Repository Structure

```text
.
├── agent/
│   ├── __init__.py
│   ├── core.py          # The ReAct reasoning loop
│   └── tools.py         # Search and Vision tool implementations
├── eval/
│   ├── harness.py       # Evaluation logic and scoring
│   └── cases.json       # Benchmark dataset (10+ posts)
├── main.py              # CLI Entry point
├── README.md            # Documentation and design decisions
└── requirements.txt
```

## 💻 Usage

### Running the Agent
To explain a specific Bluesky post:

```bash
python main.py --url "https://bsky.app/profile/user.bsky.social/post/example_id"
```

### Running the Evaluation Harness
To verify the agent's performance across the benchmark dataset:

```bash
python -m eval.harness
```

## 🧠 Design Decisions

### 1. Why a ReAct Loop?
Instead of a single-shot prompt, I implemented a Reasoning + Acting pattern. This allows the agent to:
*   **Think**: Identify entities like "Ralph Wiggum technique."
*   **Act**: Execute a targeted search.
*   **Observe**: Review the search results.
*   **Finalize**: If the results are insufficient, the agent can perform a second, refined search before summarizing.

### 2. Search Strategy
The agent is optimized to prioritize technical repositories (GitHub), developer blogs, and social archives. This ensures it captures the origin of memes (like the "Ralph Wiggum" bash-loop) rather than just general news.

### 3. The "Judge" Evaluation
The evaluation harness uses a separate "Judge LLM" to score the agent's output against the "Gold Standard" context. This provides an objective metric for Factuality and Utility.

### 4. Modular Architecture
The codebase is split into:
*   `core/agent.py`: The main reasoning logic.
*   `core/tools.py`: Search and Vision tool wrappers.
*   `eval/`: Test cases and scoring logic.

## 📊 Evaluation Results
The current agent achieves high utility scores across 10+ test cases, ranging from:
*   **Technical Jargon**: (e.g., PDS, ATProto, Vibe Coding)
*   **Community Memes**: (e.g., Ralph Wiggum, $RALPH token)
*   **Breaking News**: (e.g., New AI model releases or protocol forks)
