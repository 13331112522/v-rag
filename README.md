
# <center> **Visual RAG** </center>

Build your own **Visual RAG** Application using less than 300 lines of code.

<demo>

* **Features**
    - Ingest your videos and pictures  with Multimodal LLM
    - Q&A with LLM about your visual files
    - Run locally without compromising your privacy
    - Locating the relevant resource with quotation
    - Extremely simple with only one file with less 200 lines of code
* **Process**
    - Parse videos or pictures in the folder into text with LLava, which run locally with ollama
    - Ingest the text into vectorDB
    - Query it with local LLM.
* **Setup**
    - **Create and activate virtual environment**
        
        ```bash
        python -m venv v-rag
        source v-rag/bin/activate
        ```
        
    - **Clone repo and install dependencies**
    
      ```bash
      git clone https://github.com/13331112522/v-rag.git
      cd v-rag
      python -m pip install -r requirements.txt
      cp example.env .env
      ```
    
    - **Get ready for models**
        - Put local LLM weights into model, supporting any GGUF format, and change the MODEL_PATH in .env for your model path.
        - Run MLLM. We employ the latest llava 1.6 for image and video parsing.
        
        ```bash
        ollama run llava
        ```
        
    - **Run**
        
        Path refers to the folder which contains all the images and videos you want to retrieve, and stride refers to the frame interval for video parse.
        
        ```bash
        python v-rag.py --path ./source --stride 25
        ```
        
* **To-do List**
    - WebUI
    - JSON support rather than txt
    - [Video](http://3.video) playback with the query result.
    - Evaluation on open Dataset
    - MultiModal RAG
    - Docker support
* **Acknowledgement**
    - llava 1.6
    - PrivateGPT
    - ollama
    - langchain
    - Llama.cpp
