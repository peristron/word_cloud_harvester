# 🧠 The Unstructured Data Intel Engine
### Multi-File Word Cloud, Network Graph & Bayesian Analyzer

**A Streamlit-based intelligence platform for extracting qualitative insights from "dirty," unstructured text data.** 

This application moves beyond simple word counting. It employs **Network Graph Theory**, **Bayesian Inference**, and **Generative AI** to visualize context, measure sentiment confidence, and discover hidden topics in datasets ranging from single text files to massive 10M+ row corpora.

---

## 🚀 Key Features

### 1. Hybrid Streaming Architecture
The app automatically adapts to your data size:
*   **Small Files:** Instant visualization. See word clouds for individual files before they are merged.
*   **Large Files:** Streaming "Scan Mode." Reads files in chunks, extracts statistical "Sketches," and discards raw text to save memory (Ephemeral Processing).
*   **Enterprise Mode:** Import pre-computed sketches from offline servers for massive datasets.

### 2. 🔍 Bayesian Theme Discovery
Identify hidden topics using advanced probabilistic models:
*   **LDA (Latent Dirichlet Allocation):** Best for complex mixtures (essays, assignments).
*   **NMF (Non-negative Matrix Factorization):** Best for distinct categories (chat logs, support tickets).

### 3. 🕸️ Network Graph & Communities
*   **Concept Mapping:** Visualizes connections between words (Bigrams).
*   **Community Detection:** Automatically color-codes clusters of related terms to identify distinct conversation threads.

### 4. ⚖️ Bayesian Sentiment Inference
*   **Credible Intervals:** Uses a **Beta-Binomial** model to calculate the statistical confidence of sentiment. 
*   *Example:* "We are 95% confident the true positive rate is between 62% and 68%."

### 5. 🛠️ The Data Refinery (Utility)
*   A built-in tool to clean and split massive, messy CSVs. 
*   Upload a gigabyte-sized log file $\to$ Get a ZIP file of cleaned, Excel-ready CSV chunks.

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/unstructured-data-engine.git
    cd unstructured-data-engine
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

---

## 📖 How to Use

### Workflow A: Quick Analysis (Files < 200MB)
1.  Open the Sidebar.
2.  Drag & Drop your files (`CSV`, `XLSX`, `PDF`, `PPTX`, `TXT`, `VTT`).
3.  The app visualizes each file immediately.
4.  Scroll down to see the **Combined Analysis** (Graph, Sentiment, Topics).

### Workflow B: The Enterprise "Harvester" (10M+ Rows)
For data that is too large to upload or too sensitive to leave your secure server:
1.  Open the **"Enterprise Workflow"** expander in the app.
2.  Copy the `harvest.py` script provided there.
3.  Run it locally on your secure machine: `python harvest.py giant_dataset.csv`.
4.  Upload the resulting `.json` Sketch file to the app.

---

## 🔐 Privacy & Security
*   **Ephemeral Processing:** When scanning raw files, data is processed in memory chunks and immediately discarded. Only statistical summaries (counts) are retained in the session state.
*   **AI Privacy:** If you use the Generative AI features, only the *summary statistics* (top 100 words, graph clusters) are sent to the API. Your raw documents are never sent to the LLM.

---

## ⚙️ Requirements
*   Python 3.9+
*   `streamlit`
*   `pandas`, `numpy`, `scipy`
*   `scikit-learn` (for LDA/NMF)
*   `networkx` (for Graph Theory)
*   `nltk` (for Sentiment)
*   `openai` (Optional, for AI summaries)

---

## 📖 Use-cases
(Some)use-cases for this unstructured data intelligence engine; you'll likely think of more

- customer feedback and support analytics
- market and competitive intelligence
- academic and research applications
- internal knowledge mining
- compliance and risk monitoring
- content summarization and curation
- product dev and ux research
- crisis monitoring in real-time
- voice of the customer (voc) programs
- employee engagement analysis
- legal discovery and e-discovery
- healthcare: patient feedback and clinical notes analysis
- education: course feedback and academic research
- security: insider threat detection
- media and journalism analytics
- automated discovery of "unknown unknowns" in large, unstructured datasets
- trend detection over time (e.g., how topics or sentiment shift week-to-week)


---


## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
