# generate_clean_text_output.py
This utility script converts out BioC JSON datasets into plain text files that we can feed to the LLM for relation extraction. 

For each document in the BioC JSON file, it produces one .txt file with:

```text
<title line>

<full text line>
```

The LLM will later only get those text files as input.


## Usage: 
If you want to regenerate the Input texts, run the following: 

```bash
uv run generate_clean_text_output.py --input <path to input json file> --output <path to output dir>
```


