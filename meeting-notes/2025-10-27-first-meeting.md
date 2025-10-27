
Problem to be solved: Create relational Triples from Text. Do this as accurately as possible. 

What we have to research: Which Prompting Techniques deliver the best results? 

Questions:
- Do we get any Tokens for using LLMs?
	- Open Router access 
	- gpu cluster access 
- Which models should we use? 
	- few different models qwenn models 8B models inference 
	- Some reasoning models as well gpt 5 mini reasoning etc 

- Should we use different models and compare them to each other? 
- How can we evaluate the performance of the generated responses? 
	- syntetic nlp metrics for renaming 
	- semantic checks 
	- bird score? 
	- graph edit distance
		- how many changes to arrive at the other graph 
	- comparing graphs 
	- 
- How big should the input text be at first and towards the end of the project? 
	- just the given text 
	- 
- Should we just stick with the default settings of the LLMs or should we also experiment with LLM settings and their outcomes like temperature etc. 
	- temprerature is not needed really 
	- prompt optimizer for gpt 5 
	- 

Our Approach: 
- Define which knowledge intense domain we want to choose
- Review and research popular engineering methods and summarize their principles 
- Define Data Pipeline: Knowledge Text -> (-> preprocessing, embedding generation, fine tuning) LLM -> Relational Triples (-> Testing, Validation, statistics) -> Graphs
- Define how we want to validate the Relationlal Triplets, how can we decide what was the best response? How many tries per PRompting Technique? 
- Define clear Data Structure and Code Pattern 
- Implement all Prompts and get started Testing 
iterative proposal of triples validating, multistep query valid grap etc 


What is relation extraction 
What is it used for 
Why ?

Goal for out project 

Prompting Techniques 

One demo Of input output what it looks like 

extract the relations 

Plan of implementation 

metrics for measurement 


## Presentation Structure 

Intro 30s (D)

Definition what is relation extraction (D) 2m
Whats it used for (D) 
(Motivation)

Goal (D) 1m

Prompting techniques (Definition, (how does it work), Example related to relation extraction): 
    I/O (D) 1m
    CoT (D) 2m
    RAG (S) 2-3m
    ReAct (S) 2m

Showcase Plan 1.5m (S)

Metrics we want to check/test use for validation 1-2m (S)

Conclusion 30s (S)



















