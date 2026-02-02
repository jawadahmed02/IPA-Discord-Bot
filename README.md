# IPA_Discbot
Discord Bot for LLM-based Interactive Planning Agent

Users can add this discord bot and customize it to use various LLMs to aide in generating a variety of different plans based on their needs. 
The bot converts user inputted natural language to PDDL, which is then sent through and MCP to return optimal plans.

Features:
- View and change LLM models - bot uses python llm module to access and use various providers and models
- Conversation logging - the bot will remember conversations with different users for context
- HITL PDDL editing - the user can change certain parts of the returned plan to tweak it as they see fit
