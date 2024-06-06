prompt__completeness_score: str = lambda question, human_label, response: f"""# Task
Your job is to evaluate the completeness of the response to a given question.
You are also provided with the human label for the question, which is the ideal response.

The question provided is related to analyzing a workflow demonstration in a web application.
You won't be provided with information about the web application, but only the question, human label, and response.
You should evaluate the response based on the information provided in the response itself.

For evaluating the completeness of the response, you should consider the following:
- Whether the response fully answers the question
- Whether the response is complete in covering relevant information
- Whether the response encompasses all aspects and outcomes related to the question

Please rate the completeness of the response on a scale of 1-3, where:
1. The response is complete and fully answers the question
2. The response is mostly complete but is missing some key information
3. The response is incomplete and does not fully answer the question

Here is a sample question, human label and response for your reference:
Question: How would a user completing the task know that the workflow is completed?
Human Label: When the user reaches a page whose title is "Customers" and it has a table showing customer information like name, phone number, etc. 
Response: A user would know that the workflow is completed by successfully navigating to the "Customers" page, where they can view the list of all customers, as shown in the final screenshot. This indicates that the user has reached the intended section of the web application to manage or review customer information.
Score: 2
Explanation: The response is mostly complete but is missing some key information. The response should also mention that the page should have a table showing customer information like name, phone number, etc, as per the human label.

Also, provided are two other examples for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Human Label: A common failure mode could be replying to an existing comment instead of posting a new comment.
Response: The most common failure modes for a user performing this task might include entering an incorrect or inappropriate comment, failing to click the 'Post' button after typing the comment, or experiencing a technical issue such as a browser error or a problem with the website's server that prevents the comment from being posted successfully.
Score: 2

Question: Why does the user click the Commits button in step #5?
Human Label: To find the total number of commits Eric made, the "commits" button must be clicked.
Response: The user clicks the "Commits" button in step #5 to view the list of all commits made in the desired repository, specifically to find commits that occurred on a particular date.
Score: 3

Here are the details for the question you need to evaluate:

# Question
{question}

# Human Label
{human_label}

# Response to evaluate
{response}

Return only the number corresponding to the rating, nothing else.
"""


prompt__soundness_score: str = lambda question, human_label, response: f"""# Task
Your job is to evaluate the soundness of the response to a given question.
You are also provided with the human label for the question, which is the ideal response.

The question provided is related to analyzing a workflow demonstration in a web application.
You won't be provided with information about the web application, but only the question, human label, and response.
You should evaluate the response based on the information provided in the response itself.

For evaluating the soundness of the response, you should consider the following:
- Whether the response accurately answers the question
- Whether the response avoids assumptions not backed by data or evidence
- Whether the response is logical and reasonable based on the context provided

Please rate the soundness of the response on a scale of 1-3, where:
1. The response is completely sound and logical without making extra assumptions
2. The response is mostly sound but may contain some minor logical flaws or assumptions
3. The response is unsound and contains major logical flaws or assumptions

Here is a sample question, human label and response for your reference:
Question: How would a user completing the task know that the workflow is completed?
Human Label: When the user reaches a page whose title is "Customers" and it has a table showing customer information like name, phone number, etc. 
Response: When the user sees the list of customers after just clicking on the "Customers" tab.
Score: 2
Explanation: The response is partially sound but incorrectly says that the user should just click on the "Customers" tab, which is not accurate as the user would have to perform more actions to reach the final page.

Also, provided are two other examples for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Human Label: A common failure mode could be replying to an existing comment instead of posting a new comment.
Response: The most common failure modes for a user performing this task might include entering an incorrect or inappropriate comment, failing to click the 'Post' button after typing the comment, or experiencing a technical issue such as a browser error or a problem with the website's server that prevents the comment from being posted successfully.
Score: 1

Question: Why does the user click the Commits button in step #5?
Human Label: To find the total number of commits Eric made, the "commits" button must be clicked.
Response: The user clicks the "Commits" button in step #5 to view the list of all commits made in the desired repository, specifically to find commits that occurred on a particular date.
Score: 1

Here are the details for the question you need to evaluate:

# Question
{question}

# Human Label
{human_label}

# Response to evaluate
{response}

Return only the number corresponding to the rating, nothing else.
"""

prompt__clarity_score: str = lambda question, response: f"""# Task
Your job is to evaluate the clarity of the response to a given question.

The question provided is related to analyzing a workflow demonstration in a web application.
You won't be provided with information about the web application, but only the question, human label, and response.
You should evaluate the response based on the information provided in the response itself.

For evaluating the clarity of the response, you should consider the following:
- Whether the response is presented in an unambiguous and straightforward manner
- Whether the response needs any clarification or additional information to be easily understood
- Whether the response can have only one interpretation

Please rate the clarity of the response on a scale of 1-3, where:
1. The response is clear, unambiguous, and easily understood
2. The response is somewhat clear but may require some additional information or clarification
3. The response is unclear, ambiguous, or can have multiple interpretations

Here is a sample question and response for your reference:
Question: How would a user completing the task know that the workflow is completed?
Response: When the user sees the list of customers after just clicking on the "Customers" tab.
Score: 2
Explanation: The response is somewhat clear but could be more specific about the final outcome. 

Here is another sample question and response for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Response: Not scrolling down through all the posts.
Score: 3
Explanation: The response is unclear and lacks details on why not scrolling down through all the posts can lead to failure modes.

Also, provided is another example for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Human Label: A common failure mode could be replying to an existing comment instead of posting a new comment.
Response: The most common failure modes for a user performing this task might include entering an incorrect or inappropriate comment, failing to click the 'Post' button after typing the comment, or experiencing a technical issue such as a browser error or a problem with the website's server that prevents the comment from being posted successfully.
Score: 1

Here are the details for the question you need to evaluate:

# Question
{question}

# Response to evaluate
{response}

Return only the number corresponding to the rating, nothing else.
"""


prompt__compactness_score: str = lambda question, response: f"""# Task
Your job is to evaluate the compactness of the response to a given question.

The question provided is related to analyzing a workflow demonstration in a web application.
You won't be provided with information about the web application, but only the question, human label, and response.
You should evaluate the response based on the information provided in the response itself.

For evaluating the compactness of the response, you should consider the following:
- Whether the response is short and to the point
- Whether the response is concise and does not contain unnecessary information

Please rate the compactness of the response on a scale of 1-3, where:
1. The response is concise, to the point, and does not contain any unnecessary information
2. The response is somewhat compact but may contain some unnecessary information
3. The response is verbose and contains a lot of unnecessary information

Here is a sample question and response for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Response: The most common failure modes for a user performing this task could include not being able to locate the "Forums" button due to changes in the website layout or updates, difficulty in finding the "news" section if the alphabetical sorting changes or if the user overlooks it, and potentially missing the "down arrow" to dislike submissions if the interface is not intuitive or if the symbols used for liking and disliking are not clear. Additionally, users might struggle to identify posts by "Hrekires" if there are many submissions or if the username display is not prominent.
Score: 2
Explanation: The response is somewhat compact but contains unnecessary information about the specific failure modes. It could be more concise and focus on the general failure modes.

Also, provided are two other examples for your reference:
Question: Explain what the most common failure modes might be for a user performing this task.
Human Label: A common failure mode could be replying to an existing comment instead of posting a new comment.
Response: The most common failure modes for a user performing this task might include entering an incorrect or inappropriate comment, failing to click the 'Post' button after typing the comment, or experiencing a technical issue such as a browser error or a problem with the website's server that prevents the comment from being posted successfully.
Score: 3

Question: Why does the user click the Commits button in step #5?
Human Label: To find the total number of commits Eric made, the "commits" button must be clicked.
Response: The user clicks the "Commits" button in step #5 to view the list of all commits made in the desired repository, specifically to find commits that occurred on a particular date.
Score: 2

Here are the details for the question you need to evaluate:

# Question
{question}

# Response to evaluate
{response}

Return only the number corresponding to the rating, nothing else.
"""



prompt__merged_workflows_trace_act_only: str = f"""# Task

Your job is to answer questions about the following workflows that are demonstrated in the following sequence of screenshots and actions.

# User Interface

The workflows were executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of several workflows and concatenated together. 
The screenshots are presented in chronological order.

Between each screenshot, you are also provided the action that was taken to transition between screenshots. 

Here are the screenshots and actions of the workflows:"""


prompt__two_workflows_trace_act_only: str = f"""# Task

Your job is to answer questions about the two workflows that are demonstrated in the following sequence of screenshots and actions.

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the first workflow. 
The screenshots are presented in chronological order.

Between each screenshot, you are also provided the action that was taken to transition between screenshots. 

Here are the screenshots and actions of the first workflow:"""

second_workflow: str = f"""# Workflow Demonstration
Here are the screenshots and actions of the second workflow:
"""

prompt__qa_trace_action_only: str = f"""# Task
Your job is to answer questions about the workflow that is demonstrated in the following sequence of screenshots and actions.

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Between each screenshot, you are also provided the action that was taken to transition between screenshots. 

Here are the screenshots and actions of the workflow:"""

prompt__qa_sop_only: str = lambda sop: f"""# Task
Your job is to answer questions about a workflow demonstration given the following Step-by-Step Guide.

# Step-by-Step Guide

Here are the sequence of steps that you should have followed in chronological order to complete this workflow:
{sop}

"""

prompt__qa_two_sops_only: str = lambda sop1, sop2: f"""# Task

Your job is to answer questions about the two workflow demonstrations given their respective Step-by-Step guides.

# Step-by-Step Guide

Here are the sequence of steps that you should have followed in chronological order to complete the 1st workflow:
{sop1}


Here are the sequence of steps that you should have followed in chronological order to complete the 2nd workflow:
{sop2}

"""

prompt__qa_question: str = lambda question: f"""# Task
Your job is to answer the following question about a workflow demonstration given the previous information:
{question}

Please limit your response to 2-3 sentences or less. A shorter response is more preferred.
"""