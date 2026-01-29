
import argparse
import os
from dotenv import load_dotenv
from agent.core import BlueskyAgent

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Bluesky Explainer Agent")
    parser.add_argument("--url", type=str, help="URL of the Bluesky post to explain")
    parser.add_argument("--content", type=str, help="Content of the post (optional if URL provided, but helpful)")
    
    args = parser.parse_args()
    
    if not args.url and not args.content:
        print("Error: Please provide --url or --content.")
        return

    # In a real scenario, we might scrape the URL to get the content if not provided.
    # For simplicity, if content is missing, we'll ask the agent to try and find it or just use the URL context.
    post_content = args.content if args.content else "Content not provided, please use variable search to find the post content if needed."
    post_url = args.url if args.url else ""

    print(f"Initializing Bluesky Agent...\nURL: {post_url}\nContent: {post_content}\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    try:
        agent = BlueskyAgent()
        explanation = agent.run(post_content=post_content, post_url=post_url)
        
        print("\n=== Agent Explanation ===\n")
        print(explanation)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
