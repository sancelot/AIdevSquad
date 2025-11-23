
import json

class CostTracker:
    def __init__(self, token_counter, model_prices_data=None):
        """
        Initializes the tracker with a reference to LlamaIndex's TokenCountingHandler.
        """
        if token_counter is None:
            raise ValueError("A TokenCountingHandler is required.")
        
        self.token_counter = token_counter
 
        self.processed_calls = 0  

        if model_prices_data is None:
            # Default pricing data if none is provided
            self.model_prices_data = {
                "source_date": "2025-10-24", # Date when these prices were last checked
                "prices": {
                    "models/gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50},
                    "gpt-4o": {"input": 2.5, "output": 10.00},
                    "claude-4.1-opus": {"input": 15.00, "output": 75.00},
                    "qwen2:7b": {"input": 0.0, "output": 0.0} # Local models are free
                }
            }
        else:
            self.model_prices_data = model_prices_data
            
        self.total_costs_by_model = {model: 0.0 for model in self.model_prices_data["prices"]}

    def calculate_and_print_cost(self, model_name_used):
        
        """
        Calculates the cost for the most recent call based on LlamaIndex's counters.
        """
        # --- NEW ROBUST LOGIC ---
        new_completed_calls = len(self.token_counter.llm_token_counts)
        
        # If no new completed calls since last time, do nothing.
        if new_completed_calls <= self.processed_calls:
            return

        # Get the most recent completed event
        last_event = self.token_counter.llm_token_counts[-1]

        # Defensive check: ensure the event is complete before processing
        last_call_input = getattr(last_event, 'prompt_token_count', 0)
        last_call_output = getattr(last_event, 'completion_token_count', 0)

        # If token counts are zero, it might be an incomplete event or an error.
        # We can choose to skip it to be safe.
        if last_call_input == 0 and last_call_output == 0:
            print("📊 [Usage] Skipping cost calculation for an event with zero tokens (possibly a failed call).")
            print(last_event)
            return
        print(last_call_input,last_call_output)
        self.processed_calls = new_completed_calls # Mark this call as processed
        print(f"📊 [Usage] Call #{self.processed_calls}: Model Used='{model_name_used}', Input={last_call_input} tokens, Output={last_call_output} tokens")
        
        # --- 3. Comparative Cost Calculation ---
        for model, prices in self.model_prices_data["prices"].items():
            input_cost = (last_call_input / 1_000_000) * prices["input"]
            output_cost = (last_call_output / 1_000_000) * prices["output"]
            call_cost = input_cost + output_cost
            self.total_costs_by_model[model] += call_cost
            
            # Print the cost for this specific call for the model that was actually used
            if model == model_name_used:
                print(f"    -> Call Cost for {model}: ${call_cost:.6f}")
        
    def get_summary(self):
        """Returns the final comparative summary report."""
        summary = "\n" + "="*70 + "\n"
        summary += "📊 FINAL USAGE AND COMPARATIVE COST REPORT 📊\n"
        summary += "="*70 + "\n"

        total_calls = getattr(self.token_counter, 'total_llm_token_count', 0)
        total_prompt_tokens = getattr(self.token_counter, 'prompt_llm_token_count', 0)
        total_completion_tokens = getattr(self.token_counter, 'total_completion_tokens', 0)


        # for attr in dir(self.token_counter):
        #     if not attr.startswith('__'):
        #         print(f"{attr} = {getattr(self.token_counter, attr)}")

        summary += f"Total successful LLM calls : {total_calls}\n"
        summary += f"Total input tokens         : {total_prompt_tokens:,}\n"
        summary += f"Total output tokens        : {total_completion_tokens:,}\n"
        summary += "-"*70 + "\n"
        summary += f"COMPARATIVE COST ANALYSIS (Prices as of {self.model_prices_data['source_date']}):\n\n"
        
        # --- 4. Formatted Comparative Table ---
        # Sort models by total cost for a nice leaderboard
        sorted_costs = sorted(self.total_costs_by_model.items(), key=lambda item: item[1], reverse=True)
        
        for model, total_cost in sorted_costs:
            summary += f"  - {model:<35}: ${total_cost:10.6f}\n"

        summary += "="*70 + "\n"
        return summary

    def to_dict(self):
        """Exports the tracker's state."""
        return {
            "total_costs_by_model": self.total_costs_by_model,
            "processed_calls": self.processed_calls
        }

    def from_dict(self, data):
        """Loads the tracker's state."""
        self.total_costs_by_model = data.get("total_costs_by_model", self.total_costs_by_model)
        self.processed_calls = data.get("processed_calls", 0)
        print("CostTracker state loaded.")