# Prompting Strategies Evaluation Report

## Results Summary

| Strategy | F1-Score | Exact Match | Accuracy |
|----------|----------|-------------|----------|
| basic | 0.7200 | 0.4000 | 0.4000 |
| few_shot | 0.5467 | 0.6000 | 0.6000 |
| persona | 0.3800 | 0.6000 | 0.6000 |

## Best Strategy: BASIC

**F1-Score**: 0.7200
**Exact Match**: 0.4000
**Accuracy**: 0.4000

## Hypothesis Analysis

**Why Basic Strategy Performed Best:**
- Simple and direct prompting format
- Minimal cognitive overhead for the model
- Clear task specification without distractions
- Effective for factual question-answering tasks

## Detailed Results

### BASIC Strategy

**Question 1**: Did the team , sedating Cleveland with nitrous oxide -LRB- laughing gas -RRB- , not remove his upper left jaw and portions of his hard palate ?
- True Answer: yes
- Predicted: Yes
- F1: 1.000 | EM: 1.0

**Retrieved Context:**
```
Under the guise of a vacation cruise, Cleveland, accompanied by lead surgeon Dr. Joseph Bryant, left for New York. Bryant, joined by his assistants Dr. John F. Erdmann, Dr. W.W. Keen Jr., Dr. Ferdinand Hasbrouck (dentist and anesthesiologist), and Dr. Edward Janeway, operated aboard E. C. Benedict's...
```

**Question 2**: Is English the official language?
- True Answer: yes
- Predicted: Yes
- F1: 1.000 | EM: 1.0

**Retrieved Context:**
```
Canada's two official languages are  English and French. Official Bilingualism in Canada is law, defined in the Canadian Charter of Rights and Freedoms, the Official Languages Act, and Official Language Regulations; it is applied by the Commissioner of Official Languages. English and French have equ...
```

**Question 3**: Where is Wilson buried?
- True Answer: He was buried in Washington National Cathedral
- Predicted: Washington National Cathedral
- F1: 0.600 | EM: 0.0

**Retrieved Context:**
```
Wilson died in his S Street home on February 3, 1924. Because his plan for the League of Nations ultimately failed, he died feeling that he had lied to the American people and that his motives for joining the war had been in vain.  He was buried in Washington National Cathedral....
```

**Question 4**: Does the economy not depend heavily on exports refining imported goods , especially in manufacturing ?
- True Answer: No
- Predicted: Yes
- F1: 0.000 | EM: 0.0

**Retrieved Context:**
```
Finland has a highly industrialised, free-market economy with a per capita output equal to that of other western economies such as Sweden, the UK, France and Germany. The largest sector of the economy is services at 65.7 percent, followed by manufacturing and refining at 31.4 percent. Primary produc...
```

**Question 5**: Is Ford one of only four former Presidents to live to 90 or more years of age ?
- True Answer: Yes.
- Predicted: Yes
- F1: 1.000 | EM: 0.0

**Retrieved Context:**
```
*Ford is one of only four former Presidents to live to 90 or more years of age. The others are Ronald Reagan (93), John Adams (90) and Herbert Hoover (90)....
```

### FEW_SHOT Strategy

**Question 1**: Did the team , sedating Cleveland with nitrous oxide -LRB- laughing gas -RRB- , not remove his upper left jaw and portions of his hard palate ?
- True Answer: yes
- Predicted: Yes, they did remove his upper left jaw and portions of his hard palate.
- F1: 0.133 | EM: 1.0

**Retrieved Context:**
```
Under the guise of a vacation cruise, Cleveland, accompanied by lead surgeon Dr. Joseph Bryant, left for New York. Bryant, joined by his assistants Dr. John F. Erdmann, Dr. W.W. Keen Jr., Dr. Ferdinand Hasbrouck (dentist and anesthesiologist), and Dr. Edward Janeway, operated aboard E. C. Benedict's...
```

**Question 2**: Is English the official language?
- True Answer: yes
- Predicted: Yes
- F1: 1.000 | EM: 1.0

**Retrieved Context:**
```
Canada's two official languages are  English and French. Official Bilingualism in Canada is law, defined in the Canadian Charter of Rights and Freedoms, the Official Languages Act, and Official Language Regulations; it is applied by the Commissioner of Official Languages. English and French have equ...
```

**Question 3**: Where is Wilson buried?
- True Answer: He was buried in Washington National Cathedral
- Predicted: Washington National Cathedral.
- F1: 0.600 | EM: 0.0

**Retrieved Context:**
```
Wilson died in his S Street home on February 3, 1924. Because his plan for the League of Nations ultimately failed, he died feeling that he had lied to the American people and that his motives for joining the war had been in vain.  He was buried in Washington National Cathedral....
```

**Question 4**: Does the economy not depend heavily on exports refining imported goods , especially in manufacturing ?
- True Answer: No
- Predicted: Yes
- F1: 0.000 | EM: 0.0

**Retrieved Context:**
```
Finland has a highly industrialised, free-market economy with a per capita output equal to that of other western economies such as Sweden, the UK, France and Germany. The largest sector of the economy is services at 65.7 percent, followed by manufacturing and refining at 31.4 percent. Primary produc...
```

**Question 5**: Is Ford one of only four former Presidents to live to 90 or more years of age ?
- True Answer: Yes.
- Predicted: Yes.
- F1: 1.000 | EM: 1.0

**Retrieved Context:**
```
*Ford is one of only four former Presidents to live to 90 or more years of age. The others are Ronald Reagan (93), John Adams (90) and Herbert Hoover (90)....
```

### PERSONA Strategy

**Question 1**: Did the team , sedating Cleveland with nitrous oxide -LRB- laughing gas -RRB- , not remove his upper left jaw and portions of his hard palate ?
- True Answer: yes
- Predicted: Yes, they did remove his upper left jaw and portions of his hard palate.
- F1: 0.133 | EM: 1.0

**Retrieved Context:**
```
Under the guise of a vacation cruise, Cleveland, accompanied by lead surgeon Dr. Joseph Bryant, left for New York. Bryant, joined by his assistants Dr. John F. Erdmann, Dr. W.W. Keen Jr., Dr. Ferdinand Hasbrouck (dentist and anesthesiologist), and Dr. Edward Janeway, operated aboard E. C. Benedict's...
```

**Question 2**: Is English the official language?
- True Answer: yes
- Predicted: Yes, English is one of the two official languages of Canada.
- F1: 0.167 | EM: 1.0

**Retrieved Context:**
```
Canada's two official languages are  English and French. Official Bilingualism in Canada is law, defined in the Canadian Charter of Rights and Freedoms, the Official Languages Act, and Official Language Regulations; it is applied by the Commissioner of Official Languages. English and French have equ...
```

**Question 3**: Where is Wilson buried?
- True Answer: He was buried in Washington National Cathedral
- Predicted: Washington National Cathedral.
- F1: 0.600 | EM: 0.0

**Retrieved Context:**
```
Wilson died in his S Street home on February 3, 1924. Because his plan for the League of Nations ultimately failed, he died feeling that he had lied to the American people and that his motives for joining the war had been in vain.  He was buried in Washington National Cathedral....
```

**Question 4**: Does the economy not depend heavily on exports refining imported goods , especially in manufacturing ?
- True Answer: No
- Predicted: Yes.
- F1: 0.000 | EM: 0.0

**Retrieved Context:**
```
Finland has a highly industrialised, free-market economy with a per capita output equal to that of other western economies such as Sweden, the UK, France and Germany. The largest sector of the economy is services at 65.7 percent, followed by manufacturing and refining at 31.4 percent. Primary produc...
```

**Question 5**: Is Ford one of only four former Presidents to live to 90 or more years of age ?
- True Answer: Yes.
- Predicted: Yes.
- F1: 1.000 | EM: 1.0

**Retrieved Context:**
```
*Ford is one of only four former Presidents to live to 90 or more years of age. The others are Ronald Reagan (93), John Adams (90) and Herbert Hoover (90)....
```
