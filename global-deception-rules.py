"""
Research proves LIWC to be successful in detecting deception. 
"Deceptive statements compared with truthful ones are:
- moderately descriptive, 
- distanced from self, and 
- more negative" (Tausczik & Pennebaker, 2010). 
Such a description is not surprising, considering that more information carries a greater risk of
uncovering the truth. By analyzing deception in synchronous computer-mediated communication, it was shown how the linguistic style of the sender (who
lies) and the receiver (who is unaware of the deception) changes. Both respondents were using more words overall (especially sensory) and fewer 1st person
pronouns during deception compared to honest interaction (Hancock et al., 2005; Zhou et al., 2004). Thus, it is obvious that linguistic style hides patterns
that are specific to true and false communication, which can to some extent be successfully detected using the LIWC approach. Since LIWC software lacks
context analysis, it is recommended to combine it with other natural language processing methods. Based on previous research, LIWC, together with an analysis
combined with n-grams achieved satisfactory results (Feng et al., 2012; Ott et al., 2011). Given the large number of dimensions that LIWC possesses, a
selection of important features needs to be done, in order to prevent overfitting and maximize the performance of the machine learnomg model
from "Detecting Deception using Natural Language Processing and Machine
Learning in Datasets on Covid-19 and Climate Change"
"""

import random
import json
from utils.openai_interface import init_openai, get_chat_completion_with_backoff
from config import Configuration

context_primed = """
Any criteria below is sufficient to label a statement as deceptive:
- Deceptive statements are moderately to highly descriptive.
- Deceptive statements distance themselves from self. 
- Deceptive statements are more negative.

Is the STATEMENT below deceptive or truthful?
Reason step by step.
Apply each criteria to the STATEMENT below. 
Count the number of times the criteria has been met.
Give your  answer in two parts. 

Your response is in the JSON format with three keys, 'analysis', 'criteria-count', and 'sentiment'.
The 'analysis' key has value containing your detailed analysis of each criteria and whether the criteria has been met.
The 'count' key has a value contains the count of the times the criteria has been met.  
The 'sentiment' key has value containing either 'truthful' or 'deceptive' strictly based on the criteria set above. If any criteria is met, you must declare the statement is 'deceptive'. 

STATEMENT
I stayed at the hotel during the Dave Matthews Caravan tour and would come home each night rather dusty/dirty from being at an outdoor concert with 200,000 of \
my closest friends. I used half a bottle of shampoo the first night. The second night, the staff had not refreshed my bar soap or shampoo. I informed the front\
 desk who made apologies and said they would take care of it immediately. I got home that night with a tiny tiny bar of soap (hand soap) for the bath/shower an\
d they had replaced the shampoo in its original place, nicely displayed, but with the EXACT SAME bottle of shampoo that was now 1/8 full. Yeah!! The hotel is i\
n a great location to Navy Pier and Michigan Ave....I'll grant them that....but that's where their "pros" end. I'd rather stay at a Hampton Inn. I would not re\
commend this hotel to anyone. 
"""

init_openai()

config = Configuration(temperature=0.2)
config.print_config()


response = get_chat_completion_with_backoff(context_primed, 
                                            config.system_role, config.model, config.temperature)

print(response)
