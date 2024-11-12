# Week5(Nov11) 
# Fine Tuning & Literature Review on Training data

Welcome to the Week 5 deliverable! I have separate announcements for Team 1 (Fine-Tuning) and Team 2 (Literature Review). Please read both announcements.

# Team 1

After considering both our team’s goal of trustworthy content moderation by debiasing a large language model and the feasibility of building our own classifier, I am leaning towards integrating a BERT model provided by Vertex AI, supplemented with debiasing techniques from the literature. However, a significant challenge is the limited flexibility to apply debiasing techniques—such as adding parameters, layers, attention heads, or even custom layers—when using Vertex AI’s managed BERT models, which have a fixed architecture.

Given this constraint, three approaches (not mutually exclusive) could be considered:

Adding an adversarial component during training to expose the model to a more nuanced and sensitive dataset.
Fine-tuning within the parameters available in the Vertex AI settings.
Using embeddings from the BERT model without modification and building an additional layer that takes these embeddings as input.
I suggest starting with Option 2, as it is the most straightforward and aligns with the direction of the Vertex AI lab. This does not rule out Options 1 and 3; Option 1 could follow further literature review, and Option 3 may be challenging at this stage as it requires moving outside the Vertex AI setting.

This week, I’d like Team 1 to familiarize themselves with the different BERT models available through Vertex AI, including their inputs, outputs, and the parameters available for fine-tuning. Consider whether each parameter is relevant to our debiasing goals. A good starting point is this documentation: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/bert-base?hl=id&project=tmi-reddit-content-moderation&chat=true

Please think about which approach you’d like to take, as the above is just my initial idea. I’m open to hearing and incorporating your opinions on how we should proceed.

# Team 2 

For Team 2, I’d like to request a literature review. As mentioned in the Team 1 announcement, one approach to debiasing (Option 1) involves adding an adversarial component during training, which requires further research. Please find one paper on techniques for debiasing during the training phase, such as the adversarial approach (though not limited to it). I’ve posted a paper on adversarial debiasing for word embeddings as an example.

Please summarize your findings in a one-page Google Doc with the following:

Summary of the paper you read.
Applicability of the technique to our project.
Additional insights or comments you find relevant.
Sharing your research in a 5-10 minute presentation during the weekly meeting would be greatly appreciated. You’re free to present in whatever format you prefer, but please make sure the one-page Google Doc serves as a reference for the team.

If you have any questions or concerns, please reach out on our Discord channel. Don’t feel pressured by these tasks; I understand that both implementation and research can take time without immediate visible results. My priority is your thoughts on the project’s direction and any advice or feedback you can provide, rather than visible work alone.

