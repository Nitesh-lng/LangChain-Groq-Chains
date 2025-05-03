
# LangChain-Groq-Chains

This project demonstrates how to use [LangChain](https://www.langchain.com/) with the [Groq API](https://groq.com/) to build advanced language model pipelines using simple, sequential, parallel, and conditional execution strategies.

## 🚀 Features

- ✅ **Simple Chain**: Prompt → LLM → Output  
- 🔁 **Sequential Chain**: Multi-step reasoning (Long → Summary)
- 🔄 **Parallel Chain**: Simultaneous generation of notes and quiz → merged output
- ⚖️ **Conditional Chain**: Sentiment classification with branch logic

## 📂 Project Structure

```
chains/
│
├── simple_chain.py              # Basic prompt chain using ChatGroq
├── sequential_chain.py          # Multi-step sequential pipeline
├── parellel_chain.py            # Parallel prompt processing & merging
├── conditional_chain.py         # Branching logic using classification
├── requirements.txt             # All required dependencies
├── .env                         # Environment variables (add GROQ_API_KEY here)
```

## 🧠 LLMs Used

- `meta-llama/llama-4-scout-17b-16e-instruct`
- `deepseek-r1-distill-llama-70b`
- `llama3-8b-8192`

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Your `.env` file should contain:

```
GROQ_API_KEY=your_groq_api_key
```

## ▶️ Usage

Run any chain file directly:

```bash
python simple_chain.py
python sequential_chain.py
python parellel_chain.py
python conditional_chain.py
```

## 📌 Notes

- All LLMs are invoked via `langchain_groq.ChatGroq`
- Output parsing is handled with `StrOutputParser` and `PydanticOutputParser`
- Prompt templating is done via `PromptTemplate`

## 🧑‍💻 Author

**Nitesh Kumar Singh**  
*Data Scientist | AI Agent Systems | LLMs*

---

## 📃 License

This project is licensed under the MIT License.
