# Pipeline

![pipeline](assets/pipeline.png)

# Folder structure
```
root/
├── main.py
├── config.py
├── requirements.txt
├── schemas.py
│
├── models/
│   ├──      
│   ├──          
│   ├──           
│   └── 
│     
├── pipeline/
│   ├── text_to_sql.py    
│   ├── parser.py 
│   ├── codebert.py
│   ├── error_checker.py
│   ├── input.py
│   ├── llm_checker.py        
│   └── encoder.py
│
└── artifacts/ (not committed in this repo but locally stored)
    ├──unixcoder
    │  ├──config.json
    │  ├──model.safetensors
    │  ├──tokenizer_config.json
    │  └──tokenizer.json     
    ├──  
```