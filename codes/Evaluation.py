import rouge
from rouge import Rouge

# Reference summary from CNN daily mail
reference_summary = '''
Beckham has agreed to a five-year contract with Los Angeles Galaxy . 
New contract took effect July 1, 2007 . 
Former English captain to meet press, unveil new shirt number Friday . 
CNN to look at Beckham as footballer, fashion icon and global phenomenon .'''

# Evaluate the summary 
rouge = Rouge()
scores = rouge.get_scores(reference_summary, summary)
print("ROUGE score: ",scores)