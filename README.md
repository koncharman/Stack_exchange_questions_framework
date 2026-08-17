# Python Shiny APP for content analysis of Stack Exchange questions

This repository offers functionalities for text and tag analysis, along 
with options for evaluating topic popularity and difficulty. The application is
built using Python Shiny. The primary libraries are:
- shiny (front-end and server)
- scikit-learn (machine learning)
- tomotopy (topic modeling)
- nimfa (document clustering - Functionalities for Non-Negative Matrix Factorization NMF)
- nltk (text preprocessing)
- plotly/matplotlib/pyvis (data visualization)

## Overview

1. Load a dataset of questions using the APIs from https://data.stackexchange.com/.
2. Review post views, no answers and questions, and post dates
3. Conduct text preprocessing using post titles and body
4. Inspect frequent tags and words (title and body)
5. Run Topic Modeling (Titles + Body), Tag Clustering (Tags), Document Clustering (Titles + Body + Tags)
6. Evaluate Topic or Cluster popularity and difficulty
7. Multivariate analysis of post metrics based on text and tags using machine learning models

## Structure

```mermaid
flowchart TD
    LD["Load Dataset"]
    LD --> RPM["Review Post Metrics"]
    
    RPM --> RPM_P["Posts per year and month"]
    RPM --> RPM_B["Barplots of Views, Answers, Scores, Comments"]
    
    LD --> TXT["Text Analysis"]
    TXT --> TXT_P["Preprocessing"]
    TXT --> TXT_T['Topic Modeling']
    TXT_T --> TXT_E["Topic Evaluation"]
    TXT_T --> PO['Post Analysis']
    
    LD --> TAG['Tag Analysis']
    TAG --> TAG_C["Tag Clustering"]
    TAG_C --> CEV["Cluster Evaluation"]
    TAG_C --> PO
    
    LD --> DC['Document Analysis']
    DC --> DC_C["Document Clustering NMF"]
    DC_C --> CEV
    DC_C --> PO
    
    PO --> POP["Popularity Evaluation"]
    PO --> DIF["Difficulty Evaluation"]
    PO --> FET["Machine Learning and Feature Importance"]
    
    classDef in fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef out fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef sel fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;

    class LD,TXT_P,TXT_T,TAG_C,DC in;
    class RPM_P,RPM_B,PO,DIF,FET,TXT_E,_TAG_E,DOC_E out;
    class RPM,TXT,PO,TAG,DC sel;

```


## Screenshots
You can see some representative views of the current APP in screenshots/


```bash
pip install -r requirements.txt
```
