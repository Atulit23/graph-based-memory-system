# Graph Based Memory System for LLMs


**What does this do?**

- As we increase the context provided to LLMs, there performance starts to deteriorate. This is the issue this system aims to resolve.
- So, the solution is that we give the LLM only those chats which are relevant to the current query.
- This is essentially a graph based system in which an engine creates a graph by treating each chat as a node and the edges between these chats are created based on semantic & symbolic similarity.

**How to run?**

- Firstly, go to chatgpt.com -> Settings -> Data controls -> Export data and get your chatgpt chats
- You will recieve an email with your data.
- Extract the archive, find conversations.json and store it with the code.
- Create a virtual environemnt (optional)
- Then, in your teriminal run these commands:
    
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
- Run **gpt-chats.py**, this will generate combined-chats.json
- Then run, **superfinal.py**, this will take some time and will generate your graph (a json file).
- After you have the graph, open **index.html** & select your graph.