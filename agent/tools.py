import os
import json
from typing import List, Dict, Optional, Any
from openai import OpenAI
from ddgs import DDGS

class Tool:
    """Base class for all tools."""
    name: str = "base_tool"
    description: str = "Base tool description"

    def execute(self, **kwargs) -> Any:
        raise NotImplementedError

class SearchTool(Tool):
    """Tool to search the web using DuckDuckGo."""
    name = "search"
    description = "Useful for finding current information, technical documentation, or explaining memes. Input should be a search query."

    def execute(self, query: str, max_results: int = 5) -> str:
        """Executes a search query and returns the results as a string."""
        print(f"SEARCHING : {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
            formatted_results = []
            for r in results:
                formatted_results.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}")
            
            return "\n\n".join(formatted_results) if formatted_results else "No results found."
        except Exception as e:
            return f"Error performing search: {str(e)}"

class VisionTool(Tool):
    """Tool to analyze images using an external vision model (Placeholder)."""
    name = "vision"
    description = "Useful for describing the content of an image from a URL. Input should be the image URL."

    def execute(self, image_url: str) -> str:
        """Analyzes an image and returns a description using GPT-4o."""
        print(f"ANALYZING IMAGE: {image_url}")
        try:
             client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
             response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail. Identify any text, memes, objects, or context relevant to social media."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
             return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

class BlueskyTool(Tool):
    """Tool to fetch post content from a Bluesky URL."""
    name = "bluesky_fetch"
    description = "Useful for fetching the actual text content of a Bluesky post given its URL. Input should be the full Bluesky post URL."

    def execute(self, url: str) -> str:
        """Fetches post content using atproto."""
        print(f"FETCHING BLUESKY POST: {url}")
        try:
            from atproto import Client, IdResolver
            
            # Anonymous public access (if supported) or unauthenticated fetching
            # For public posts, we often don't need a login if using the proper XRPC endpoint,
            # but atproto client usually expects login for many actions. 
            # However, `get_post_thread` often works publicly on public PDS instances.
            
            # Simple workaround: Try unauthenticated client or just parse the URI if possible.
            # actually atproto Client needs login usually.
            # Let's try to use the public API without auth if possible, or fall back to error.
            # For this demo, we'll try a public instance access pattern if library permits.
            
            # Parsing the URL to get repo (handle) and rkey
            # URL format: https://bsky.app/profile/{handle}/post/{rkey}
            if "bsky.app/profile/" not in url or "/post/" not in url:
                return "Error: Invalid Bluesky URL format."

            parts = url.split("bsky.app/profile/")[1].split("/post/")
            handle = parts[0]
            rkey = parts[1].split("/")[0] # handle trailing slashes
            
            # To fetch without auth, we can use the public interface of a PDS like bsky.social
            client = Client("https://public.api.bsky.app") 
            
            # Resolve handle to DID
            did = client.resolve_handle(handle).did
            
            # Get post
            uri = f"at://{did}/app.bsky.feed.post/{rkey}"
            post_thread = client.get_post_thread(uri=uri)
            
            if hasattr(post_thread.thread, 'post'):
                post = post_thread.thread.post
                record = post.record
                content = f"Post Content by @{handle}:\n{record.text}"
                
                # Check for images in the post view embed
                if hasattr(post, 'embed') and post.embed:
                    # Check if it's a generic view with images (app.bsky.embed.images#view)
                    if hasattr(post.embed, 'images'):
                        for img in post.embed.images:
                            if hasattr(img, 'fullsize'):
                                content += f"\nImage URL: {img.fullsize}"
                    # Check if it's a record with media (app.bsky.embed.recordWithMedia#view)
                    elif hasattr(post.embed, 'media') and hasattr(post.embed.media, 'images'):
                         for img in post.embed.media.images:
                            if hasattr(img, 'fullsize'):
                                content += f"\nImage URL: {img.fullsize}"
                                
                return content
            else:
                return "Error: Could not retrieve post content. Post might be deleted or private."

        except Exception as e:
            return f"Error fetching Bluesky post: {str(e)}"

def get_tools() -> Dict[str, Tool]:
    """Returns a dictionary of available tools."""
    return {
        "search": SearchTool(),
        "vision": VisionTool(),
        "bluesky_fetch": BlueskyTool()
    }
