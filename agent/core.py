
import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel

from agent.tools import get_tools

# --- Pydantic Models for Structured Output (Optional but recommended for robustness) ---
# For this simplified implementation, we'll use strong prompt engineering + parsing 
# to keep it flexible, or we could use `response_format` if the model supports it well.
# We will use a dedicated method to parse the LLM's "Thought: ... Action: ..." format.

class BlueskyAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.tools = get_tools()
        self.max_steps = 5
        self.history: List[Dict[str, str]] = []

    def _construct_system_prompt(self) -> str:
        tool_descriptions = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
        
        return f"""You are an advanced AI assistant designed to explain Bluesky posts, memes, and technical jargon to a general audience.
You have access to the following tools:

{tool_descriptions}

You function as a ReAct (Reason + Act) agent. For each step, you must follow this format STRICTLY:

Thought: [Your reasoning about what information is missing or what to do next]
Action: [The name of the tool to use, e.g., 'search' or 'vision']
Action Input: [The input for the tool, e.g., a search query or image URL]

Once you receive an Observation, you will repeat the cycle until you have enough information.
When you are ready to provide the final answer, use this format:

Thought: I have sufficient information.
Final Answer:
* [Point 1] (Source)
* [Point 2] (Source)
* ...

PROTOCOL (Prioritized Decision Tree):

1. **MISSING CONTENT CHECK**:
   - Is the post content missing but a URL is provided?
   - If YES: Use `bluesky_fetch` to retrieve the text.
   - Then proceed to Step 2 with the fetched text.

2. **CHECK FOR IMAGES (CONTEXT ASSEMBLY)**:
   - Does the post (or fetched content) contain an image URL?
   - If YES: **Use the `vision` tool immediately** to get a description.
   - Combine the Post Text + Image Description for the next steps.

3. **ANALYZE CONTENT SIMPLICITY**: 
   - Is the combined context (Text + Image) strictly about simple, common knowledge?
   - If YES: **Do NOT search.** Provide a final Answer immediately.
   - If NO: Proceed to step 4.

4. **CHECK FOR JARGON/MEMES**:
   - Does the post contain technical jargon, memes, or slang that isn't explained by the image?
   - If YES: Use the `search` tool to find definitions or origins.

5. **FINALIZE**:
   - Once you have enough context, provide the Final Answer.

RULES:
- "Action" must be one of {list(self.tools.keys())}.
- If no search results are found, try a different query.
- **Output the Final Answer as a list of bullet points.**
"""

    def run(self, post_content: str, post_url: str = "") -> str:
        """Runs the ReAct loop to explain the post."""
        
        # Reset history for a new run
        self.history = [
            {"role": "system", "content": self._construct_system_prompt()},
            {"role": "user", "content": f"Please explain this Bluesky post:\nURL: {post_url}\nContent: {post_content}"}
        ]

        step_count = 0
        
        while step_count < self.max_steps:
            print(f"--- Step {step_count + 1} ---")
            
            # 1. Call LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                stop=["Observation:"] # Stop before the agent hallucinates an observation
            )
            
            llm_output = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": llm_output})
            print(f"{llm_output}")

            # 2. Parse Output
            if "Final Answer:" in llm_output:
                return llm_output.split("Final Answer:")[1].strip()

            if "Action:" in llm_output and "Action Input:" in llm_output:
                # Extract Action and Action Input
                try:
                    action_line = [line for line in llm_output.split('\n') if line.startswith("Action:")][0]
                    action_name = action_line.split("Action:")[1].strip().lower()
                    
                    input_line = [line for line in llm_output.split('\n') if line.startswith("Action Input:")][0]
                    action_input = input_line.split("Action Input:")[1].strip()
                    
                    # 3. Execute Tool
                    if action_name in self.tools:
                        if action_name == 'search':
                            tool_result = self.tools[action_name].execute(query=action_input)
                        elif action_name == 'bluesky_fetch':
                            tool_result = self.tools[action_name].execute(url=action_input)
                        else:
                            # Default to image_url for vision or others
                            tool_result = self.tools[action_name].execute(image_url=action_input)
                            
                        observation = f"Observation: {tool_result}"
                        print(f"Observation: {tool_result[:200]}...") # Truncate for logging
                    else:
                        observation = f"Observation: Error: Tool '{action_name}' not found."

                except Exception as e:
                    observation = f"Observation: Error parsing action: {str(e)}"
            else:
                 observation = "Observation: I need to specify an Action and Action Input to use a tool, or provide a Final Answer."

            # 4. Feed Observation back
            self.history.append({"role": "user", "content": observation})
            step_count += 1

        return "Error: Maximum steps reached without a final answer."
