import random

# Predefined responses
philosophy_responses = [
    "Biased data sets/We are flawed, so machines learn/Our shared healing path",        
    "Algorithm's choice/Affects the whole community/Ubuntu's warning",                   
    "Code without conscience/Needs our collective wisdom/Ubuntu guides truth",           
    "Data sets exclude/Voices silenced become gaps/We must include all",                
    "Digital divide/Separates us from ourselves/Ubuntu bridges all",                    
    "Synthetic voices/Echo our unspoken truths/We are accountable"                      
]

# Random response
random_haiku = random.choice(philosophy_responses)
print("\nUbuntu Inspired Haiku:\n")
lines = random_haiku.split("/")
for line in lines:
    print(line)


