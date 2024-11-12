# Week5(Nov11) 
# Fine Tuning & Literature Review on Training data

Welcome to the Week 5 deliverable! I have separate announcements for Team 1 (Fine-Tuning) and Team 2 (Literature Review). Please read both announcements.

Team 1
After considering both our team’s goal of trustworthy content moderation by debiasing a large language model and the feasibility of building our own classifier, I am leaning towards integrating a BERT model provided by Vertex AI, supplemented with debiasing techniques from the literature. However, a significant challenge is that the flexibility to apply debiasing techniques—such as adding parameters, layers, attention heads, or even custom layers—is notably limited when using Vertex AI’s managed BERT models, which have a fixed architecture.
Given this constraint, three approaches (not mutually exclusive) could be considered:
Adding an adversarial component during training to expose the model to a more nuanced and sensitive dataset.
Fine-tuning within the parameters available in the Vertex AI settings.
Using embeddings from the BERT model without modification and building an additional layer that takes these embeddings as input.
I suggest starting with Option 2, as it is the most straightforward and aligns with the direction of the Vertex AI lab. This does not rule out Options 1 and 3; Option 1 could follow further literature review, and Option 3 may be challenging at this stage as it requires moving outside the Vertex AI setting.
This week, I’d like Team 1 to familiarize themselves with the different BERT models available through Vertex AI, including their inputs, outputs, and the parameters available for fine-tuning. Consider whether each parameter is relevant to our debiasing goals. A good starting point is this documentation: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/bert-base?hl=id&project=tmi-reddit-content-moderation&chat=true
Also, I want you to think about which approach you’d like to take, as the above is just my initial idea. I am open to hearing and incorporating your opinions on how we should proceed.


