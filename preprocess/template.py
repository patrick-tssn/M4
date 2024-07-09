from dataclasses import dataclass

# pasts, currents, nexts 

@dataclass
class Templates:
    pasts = [
        "What was I doing earlier?",
        "Can you summarize what I've accomplished so far?",
        "What tasks have I completed?",
        "What was the outcome of my previous action?",
        "Did I succeed in what I was trying to do?",
        "What were the steps I took to achieve that?",
        "Can you recall the last thing I did?",
        "How many tasks have I finished today?",
        "What was the most challenging part of what I did?",
        "Did I make any mistakes?",
        "What did I learn from my previous experience?",
        "Can you show me the history of my actions?",
        "What was the goal of my previous task?",
        "Did I meet my expectations?",
        "What were the resources I used?",
        "Can you analyze my performance?",
        "What were the key takeaways from what I did?",
        "How does what I did relate to my overall goal?",
        "Can you provide feedback on my approach?",
        "What would you have done differently?",
        "Were there any obstacles I overcame?",
        "How did I handle the challenges?",
        "What were the benefits of what I did?",
        "Can you recommend improvements?",
        "What were the consequences of my actions?",
        "Did I follow the correct procedure?",
        "Can you verify the accuracy of my work?",
        "What were the key metrics I was tracking?",
        "How does what I did impact the bigger picture?",
        "Can you provide a summary of my actions?",
        "What were the dependencies involved?",
        "Did I collaborate with anyone?",
        "What was the scope of my project?",
        "Can you evaluate the quality of my work?",
        "What were the risks involved?",
        "How did I mitigate them?",
        "What were the opportunities I seized?",
        "Can you highlight the most important aspects?",
        "What were the lessons I learned?",
        "Can you create a report of my progress?",
        "What were the milestones I achieved?",
        "How does what I did align with my goals?",
        "Can you provide a summary of my accomplishments?",
        "What were the skills I demonstrated?",
        "Can you identify areas for improvement?",
        "What were the tools I used?",
        "Can you recreate the steps I took?",
        "What were the outcomes I expected?",
        "What done?",
        "please summarize."
    ]
    currents = [
        "What am I doing right now?",
        "Can you describe my current task?",
        "What is my current goal?",
        "How does what I'm doing now relate to my previous task?",
        "What are the key steps involved in what I'm doing?",
        "Can you break down my current task into smaller steps?",
        "What are the resources I'm using right now?",
        "What are the potential obstacles I might face?",
        "Can you provide guidance on what I'm doing?",
        "What are the benefits of what I'm doing now?",
        "How does what I'm doing now impact the bigger picture?",
        "Can you analyze my current approach?",
        "What are the key metrics I should be tracking?",
        "How does what I'm doing now align with my goals?",
        "Can you provide feedback on my current progress?",
        "What are the dependencies involved in what I'm doing?",
        "Am I on the right track?",
        "Can you identify potential risks?",
        "How can I improve what I'm doing now?",
        "What are the opportunities I should seize?",
        "Can you highlight the most important aspects of what I'm doing?",
        "What are the skills I'm demonstrating?",
        "Can you recreate the steps I'm taking?",
        "What are the outcomes I expect from what I'm doing?",
        "Can you reflect on my decision-making process?",
        "What are the consequences of what I'm doing?",
        "Did I follow the correct procedure?",
        "Can you verify the accuracy of my work?",
        "What are the key takeaways from what I'm doing?",
        "How does what I'm doing now relate to my overall goal?",
        "Can you provide a summary of my current task?",
        "What were the prerequisites for what I'm doing?",
        "Can you identify areas for improvement?",
        "What are the tools I'm using?",
        "Can you show me an example of what I'm doing?",
        "How does what I'm doing now compare to what I did before?",
        "What are the benefits of taking this approach?",
        "Can you provide an alternative solution?",
        "What are the trade-offs involved?",
        "Can you evaluate the quality of my work?",
        "What are the risks involved in what I'm doing?",
        "How can I mitigate them?",
        "What are the opportunities I should consider?",
        "Can you highlight the most critical aspects?",
        "What are the skills I need to develop?",
        "Can you provide a roadmap for what I'm doing?",
        "What are the milestones I should achieve?",
        "How does what I'm doing now impact my future tasks?",
        "Now?",
        "current?"
    ]
    nexts = [
        "What should I do next?",
        "Can you recommend the next step?",
        "What is the most important task I should focus on?",
        "How does what I'm doing now lead to what I should do next?",
        "What are the prerequisites for the next task?",
        "Can you provide a roadmap for what I should do next?",
        "What are the key metrics I should track for the next task?",
        "How does what I should do next align with my goals?",
        "Can you identify potential obstacles for the next task?",
        "How can I overcome them?",
        "What are the benefits of doing what I should do next?",
        "How does what I should do next impact the bigger picture?",
        "Can you analyze my approach for the next task?",
        "What are the resources I'll need for the next task?",
        "What are the dependencies involved in the next task?",
        "Can you provide guidance on what I should do next?",
        "What are the opportunities I should seize in the next task?",
        "Can you highlight the most important aspects of what I should do next?",
        "What are the skills I'll need to demonstrate?",
        "Can you recreate the steps I'll need to take?",
        "What are the outcomes I should expect from the next task?",
        "Can you reflect on my decision-making process for the next task?",
        "What are the consequences of what I should do next?",
        "Did I follow the correct procedure for the next task?",
        "Can you verify the accuracy of my work for the next task?",
        "What are the key takeaways from what I should do next?",
        "How does what I should do next relate to my overall goal?",
        "Can you provide a summary of the next task?",
        "What were the prerequisites for the next task?",
        "Can you identify areas for improvement in the next task?",
        "What are the tools I'll need for the next task?",
        "Can you show me an example of what I should do next?",
        "How does what I should do next compare to what I did before?",
        "What are the benefits of taking this approach for the next task?",
        "Can you provide an alternative solution for the next task?",
        "What are the trade-offs involved in the next task?",
        "Can you evaluate the quality of my work for the next task?",
        "What are the risks involved in the next task?",
        "How can I mitigate them?",
        "What are the opportunities I should consider in the next task?",
        "Can you highlight the most critical aspects of the next task?",
        "What are the skills I need to develop for the next task?",
        "Can you provide an instruction for the next task?",
        "What are the milestones I should achieve in the next task?",
        "How does what I should do next impact my future tasks?",
        "Can you reflect on my progress so far?",
        "What are the next steps I should take?",
        "Can you recommend a course of action?",
        "Next?",
        "nexts?"
    ]
    casuals = ['...'] * 50
    queries = pasts + currents + nexts + casuals
