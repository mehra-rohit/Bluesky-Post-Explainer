
import json
import os
import argparse
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from agent.core import BlueskyAgent

load_dotenv()

class EvaluationHarness:
    def __init__(self, cases_path: str = "eval/cases.json"):
        with open(cases_path, "r") as f:
            self.cases = json.load(f)
        self.agent = BlueskyAgent()
        self.judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def evaluate_case(self, case: Dict) -> Dict:
        """Runs the agent on a case and uses a Judge LLM to score it."""
        print(f"\nRunning case {case['id']}: {case.get('post_content', 'Fetching URL...')[:50]}...")
        
        # 1. Run Agent
        try:
            # Default content to empty string if not present, to trigger fetch
            content_input = case.get('post_content', "")
            agent_output = self.agent.run(post_content=content_input, post_url=case['post_url'])
        except Exception as e:
            agent_output = f"Error: {e}"

        # 2. Judge Ouptut
        score = self._judge_output(case['post_content'] if 'post_content' in case else "Contentfetched from URL", case['gold_standard'], agent_output)
        
        return {
            "id": case['id'],
            "agent_output": agent_output,
            "scores": score
        }

    def _judge_output(self, original_post: str, gold_standard: str, agent_output: str) -> Dict:
        """Uses an LLM to score the output."""
        prompt = f"""
        You are an impartial judge evaluating an AI agent's ability to explain social media posts.
        
        Original Post: {original_post}
        Gold Standard Context: {gold_standard}
        
        Agent Explanation: {agent_output}
        
        Evaluate the Agent Explanation on two metrics (1-5 scale):
        1. Factuality: Is the information true and consistent with the Gold Standard?
        2. Utility: Is the explanation helpful, clear, and does it provide necessary context?
        
        Output valid JSON only:
        {{
            "factuality": <int>,
            "utility": <int>,
            "reasoning": "<short explanation>"
        }}
        """
        
        try:
            response = self.judge_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an evaluation judge."}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Judging error: {e}")
            return {"factuality": 0, "utility": 0, "reasoning": "Error in judging."}

    def run_benchmark(self, models: List[str] = ["gpt-4o", "gpt-4o-mini"]):
        """Runs the evaluation harness across multiple models."""
        benchmark_results = {}

        for model in models:
            print(f"\n\n====== Benchmarking Model: {model} ======")
            self.agent = BlueskyAgent(model_name=model)
            
            model_results = []
            for case in self. cases:
                result = self.evaluate_case(case)
                model_results.append(result)
            
            # Calculate averages
            avg_fact = sum(r['scores']['factuality'] for r in model_results) / len(model_results)
            avg_util = sum(r['scores']['utility'] for r in model_results) / len(model_results)
            
            benchmark_results[model] = {
                "avg_factuality": avg_fact,
                "avg_utility": avg_util,
                "details": model_results
            }

        print("\n\n====== Final Component Benchmark Results ======")
        print(f"{'Model':<15} | {'Factuality':<12} | {'Utility':<12}")
        print("-" * 45)
        for model, stats in benchmark_results.items():
            print(f"{model:<15} | {stats['avg_factuality']:<12.2f} | {stats['avg_utility']:<12.2f}")

        # Save results to JSON
        with open("eval/benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\nDetailed results saved to eval/benchmark_results.json")

        # Save readable report to Markdown
        with open("eval/benchmark_report.md", "w") as f:
            f.write("# Benchmark Results\n\n")
            for model, stats in benchmark_results.items():
                f.write(f"* **{model}**\n")
                f.write(f"    * Factuality: {stats['avg_factuality']:.2f}\n")
                f.write(f"    * Utility: {stats['avg_utility']:.2f}\n")
        print(f"Readable report saved to eval/benchmark_report.md")

if __name__ == "__main__":
    harness = EvaluationHarness()
    harness.run_benchmark(models=["gpt-4o", "gpt-4o-mini"])
