# LangChain Chains

This module demonstrates various **Chain** patterns in LangChain using the **LangChain Expression Language (LCEL)** — the `|` (pipe) operator that connects prompts, models, and parsers together.

---

## 📂 Files

### 1. `simple_chain.py` — Simple Chain
The most basic chain pattern: **Prompt → Model → Parser**.

- Uses `gemini-2.5-flash` via Google Generative AI.
- A single `PromptTemplate` asks about a topic in 5 points.
- `StrOutputParser` extracts the text from the model response.
- Prints the chain graph using `get_graph().print_ascii()`.

```
PromptTemplate → ChatGoogleGenerativeAI → StrOutputParser
```

### 2. `sequential_chain.py` — Sequential Chain
Chains **two prompts sequentially** — the output of the first feeds into the second.

- First prompt explains a topic, second prompt summarizes it in 4 lines.
- `StrOutputParser` between the two model calls converts the output to a plain string, making it compatible as input for the next prompt.

```
Prompt1 → Model → Parser → Prompt2 → Model → Parser
```

### 3. `parallel_chain.py` — Parallel Chain
Runs **two sub-chains in parallel** using `RunnableParallel`, then merges results.

- Uses **two models simultaneously**: `gemini-2.5-flash` (Google) and `Llama-3.1-8B` (HuggingFace).
- Parallel branch 1: Generates **notes** from a given text.
- Parallel branch 2: Generates **quizzes** from the same text.
- A merge chain combines both outputs into a single document.

```
         ┌─ Prompt1 → Model1 → Parser (notes) ─┐
Input ──►│                                       ├──► Prompt3 → Model1 → Parser
         └─ Prompt2 → Model2 → Parser (quiz)  ──┘
```

### 4. `conditional_chain.py` — Conditional (Branching) Chain
Uses `RunnableBranch` to **route execution** based on a condition.

- First classifies feedback sentiment (positive/negative) using a `PydanticOutputParser` with a `Sentiment` schema.
- `RunnableBranch` then routes to different prompts based on the detected sentiment.
- Falls back to a `RunnableLambda` for invalid sentiments.

```
Input → Classifier Chain → RunnableBranch
                              ├─ positive → Prompt2 → Model → Parser
                              ├─ negative → Prompt3 → Model → Parser
                              └─ default  → "Invalid sentiment"
```

---

## 🧠 Key Concepts

| Chain Type | LCEL Component | Purpose |
|---|---|---|
| Simple | `\|` (pipe) | Linear prompt → model → parser |
| Sequential | `\|` (pipe) | Multi-step processing pipeline |
| Parallel | `RunnableParallel` | Run multiple chains concurrently |
| Conditional | `RunnableBranch` | Route to different chains based on conditions |

---

## ⚙️ Setup

1. Create a `.env` file with your API tokens:
   ```
   GOOGLE_API_KEY=your_google_api_key
   HUGGINGFACEHUB_API_TOKEN=your_hf_token
   ```
2. Install dependencies:
   ```bash
   pip install langchain-core langchain-google-genai langchain-huggingface pydantic python-dotenv
   ```
3. Run any file:
   ```bash
   python simple_chain.py
   ```
