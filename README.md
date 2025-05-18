# Images to captions and labels 

Using https://github.com/salesforce/BLIP

And NER with SPACY

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
python3 -m spacy download en_core_web_sm
```

Add images to a directory called `images/`

## Usage

```
python3 generate.py
```

Outputs in `/results`