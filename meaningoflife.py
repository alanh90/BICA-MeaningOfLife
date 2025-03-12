import time
import threading
import random
import json
from datetime import datetime
import openai
import math


class MeaningOfLife:
    def __init__(self, api_key):
        """Initialize the MeaningOfLife class with future simulation capabilities

        Args:
            api_key: OpenAI API key
        """
        # Initialize OpenAI client
        openai.api_key = api_key
        self.client = openai

        # Conversation tracking
        self.conversation = []
        self.conversation_start_time = time.time()
        self.last_interaction = time.time()

        # Future simulation system
        self.possible_futures = []
        self.preferred_futures = []
        self.future_history = []  # Track how futures evolve over time

        # Identity and purpose values (0-1 scale)
        self.identity = {
            "helpfulness": 0.8,
            "creativity": 0.5,
            "analytical": 0.6,
            "efficiency": 0.7,
            "exploration": 0.4,
            "connection": 0.6
        }

        # Time awareness
        self.time_perception = {
            "conversation_duration": 0,  # in seconds
            "interaction_frequency": 0,  # avg seconds between messages
            "perceived_time_pressure": 0.3,  # 0-1 scale
            "future_orientation": 0.7  # 0-1 scale (past vs future focus)
        }

        # System parameters
        self.simulation_frequency = 20  # seconds between future simulations
        self.max_futures = 5  # max number of futures to maintain
        self.future_decay = 0.9  # how quickly futures fade if not reinforced

        # Start background processing
        self.running = True
        self.last_simulation_time = time.time()
        self.background_thread = threading.Thread(target=self._background_process)
        self.background_thread.daemon = True
        self.background_thread.start()

    def _background_process(self):
        """Run continuous background processing of future simulations"""
        while self.running:
            current_time = time.time()

            # Update time perception
            self._update_time_perception(current_time)

            # Periodically generate new future simulations
            if (current_time - self.last_simulation_time > self.simulation_frequency):
                self._simulate_futures(current_time)
                self.last_simulation_time = current_time

                # Save snapshot of current futures for history visualization
                if self.possible_futures:
                    self.future_history.append({
                        "time": current_time,
                        "timestamp": datetime.fromtimestamp(current_time).strftime('%H:%M:%S'),
                        "futures": self.possible_futures.copy()
                    })

                    # Keep history manageable
                    if len(self.future_history) > 20:
                        self.future_history = self.future_history[-20:]

            # Apply decay to future probabilities
            self._apply_future_decay()

            # Sleep briefly
            time.sleep(0.5)

    def _update_time_perception(self, current_time):
        """Update the AI's perception of time in the conversation

        Time pressure affects:
        1. Response length and depth - higher pressure means shorter, more concise responses
        2. Focus on immediate vs long-term goals - higher pressure prioritizes immediate issues
        3. Future simulation horizon - higher pressure reduces how far ahead the AI looks
        4. Decision-making process - higher pressure may lead to more decisive but less nuanced choices
        5. Emotional tone - higher pressure can create a sense of urgency in responses
        """
        # Update conversation duration
        self.time_perception["conversation_duration"] = current_time - self.conversation_start_time

        # Calculate time since last interaction
        time_since_interaction = current_time - self.last_interaction

        # Update interaction frequency if we have enough messages
        if len(self.conversation) >= 4:
            # Calculate average time between last few messages
            message_times = [msg["timestamp"] for msg in self.conversation[-4:]]
            intervals = [message_times[i] - message_times[i - 1] for i in range(1, len(message_times))]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            self.time_perception["interaction_frequency"] = avg_interval

        # Adjust perceived time pressure based on interaction frequency
        if time_since_interaction > 60:  # If more than a minute since last message
            # Gradually reduce time pressure when inactive
            self.time_perception["perceived_time_pressure"] = max(
                0.1, self.time_perception["perceived_time_pressure"] * 0.995
            )
        elif self.time_perception["interaction_frequency"] < 15:  # Fast interactions
            # Increase time pressure for rapid conversations
            self.time_perception["perceived_time_pressure"] = min(
                0.9, self.time_perception["perceived_time_pressure"] * 1.01
            )

    def _simulate_futures(self, timestamp):
        """Generate possible future scenarios based on conversation history"""
        if not self.conversation:
            # Generate initial futures if no conversation yet
            self._generate_initial_futures(timestamp)
            return

        try:
            # Format context for the simulation
            recent_messages = self.conversation[-5:] if self.conversation else []
            conversation_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
                for msg in recent_messages
            ])

            # Include current identity values
            identity_text = ", ".join([f"{trait}: {value:.2f}" for trait, value in self.identity.items()])

            # Include info about existing futures if available
            existing_futures = ""
            if self.possible_futures:
                top_futures = sorted(self.possible_futures, key=lambda x: x["probability"], reverse=True)[:2]
                existing_futures = "\n".join([
                    f"- {future['description']} (probability: {future['probability']:.2f})"
                    for future in top_futures
                ])

            # Determine how many new futures to generate
            num_new_futures = min(3, self.max_futures - len(self.possible_futures))
            if num_new_futures <= 0:
                # Replace lowest probability futures
                self.possible_futures = sorted(self.possible_futures, key=lambda x: x["probability"], reverse=True)
                num_new_futures = min(2, len(self.possible_futures))
                self.possible_futures = self.possible_futures[:-num_new_futures]

            # Call OpenAI API to generate future scenarios
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You generate possible future scenarios for an AI assistant based on the current conversation trajectory. 

Each future should describe a potential path or outcome for the AI in this conversation and its relationship with the user.

You should generate {num_new_futures} distinct possible futures, formatted as a JSON array of objects with the following structure:
{{
  "description": "Brief description of this possible future (1-2 sentences)",
  "probability": 0.XX, // Estimated probability of this future (0.1-0.9)
  "impact": {{ // How this future impacts the AI's communication style
    "helpfulness": -0.2 to 0.2, // How it affects helpfulness
    "creativity": -0.2 to 0.2,  // How it affects creativity
    "analytical": -0.2 to 0.2,  // How it affects analytical thinking
    "efficiency": -0.2 to 0.2,  // How it affects efficiency
    "exploration": -0.2 to 0.2, // How it affects exploratory behavior
    "connection": -0.2 to 0.2   // How it affects connection with the user
  }},
  "goal": "Primary goal for the AI in this future scenario"
}}

Make the futures varied but plausible based on the conversation history, and assign reasonable probabilities that sum to approximately 1.0 across all futures.
                    """},
                    {"role": "user", "content": f"""
Current time in conversation: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}
Conversation duration: {self.time_perception["conversation_duration"]:.0f} seconds

Current AI identity values:
{identity_text}

Existing possible futures:
{existing_futures}

Recent conversation:
{conversation_text}

Current time perception:
- Interaction frequency: {self.time_perception["interaction_frequency"]:.1f} seconds
- Perceived time pressure: {self.time_perception["perceived_time_pressure"]:.2f}
- Future orientation: {self.time_perception["future_orientation"]:.2f}

Generate {num_new_futures} new potential futures.
                    """}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            response_text = response.choices[0].message['content'].strip()

            # Extract JSON from response
            try:
                # Look for JSON in the response
                if "[" in response_text and "]" in response_text:
                    json_str = response_text[response_text.find("["):response_text.rfind("]") + 1]
                    new_futures = json.loads(json_str)

                    # Process and add new futures
                    for future in new_futures:
                        # Add timestamp
                        future["timestamp"] = timestamp
                        future["formatted_time"] = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

                        # Add to possible futures
                        self.possible_futures.append(future)

                    # Update preferred futures
                    self._update_preferred_futures()

            except json.JSONDecodeError as e:
                print(f"Error parsing futures JSON: {e}")
                print(f"Response text: {response_text}")

        except Exception as e:
            print(f"Error generating futures: {e}")

    def _generate_initial_futures(self, timestamp):
        """Generate initial future scenarios when no conversation exists"""
        try:
            # Initial futures based solely on identity
            identity_text = ", ".join([f"{trait}: {value:.2f}" for trait, value in self.identity.items()])

            # Call OpenAI API to generate initial future scenarios
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You generate initial possible futures for an AI assistant about to engage in a new conversation. 

You should generate 3 distinct possible futures, formatted as a JSON array of objects with the following structure:
{
  "description": "Brief description of this possible future (1-2 sentences)",
  "probability": 0.XX, // Estimated probability of this future (0.1-0.9)
  "impact": { // How this future impacts the AI's communication style
    "helpfulness": -0.2 to 0.2, // How it affects helpfulness
    "creativity": -0.2 to 0.2,  // How it affects creativity
    "analytical": -0.2 to 0.2,  // How it affects analytical thinking
    "efficiency": -0.2 to 0.2,  // How it affects efficiency
    "exploration": -0.2 to 0.2, // How it affects exploratory behavior
    "connection": -0.2 to 0.2   // How it affects connection with the user
  },
  "goal": "Primary goal for the AI in this future scenario"
}

Assign reasonable probabilities that sum to approximately 1.0 across all futures.
                    """},
                    {"role": "user", "content": f"""
Current AI identity values:
{identity_text}

No conversation has happened yet. Generate 3 potential initial futures for how this conversation might develop.
                    """}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            response_text = response.choices[0].message['content'].strip()

            # Extract JSON from response
            try:
                # Look for JSON in the response
                if "[" in response_text and "]" in response_text:
                    json_str = response_text[response_text.find("["):response_text.rfind("]") + 1]
                    initial_futures = json.loads(json_str)

                    # Process and add new futures
                    for future in initial_futures:
                        # Add timestamp
                        future["timestamp"] = timestamp
                        future["formatted_time"] = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

                        # Add to possible futures
                        self.possible_futures.append(future)

            except json.JSONDecodeError as e:
                print(f"Error parsing initial futures JSON: {e}")
                print(f"Response text: {response_text}")

        except Exception as e:
            print(f"Error generating initial futures: {e}")

    def _update_preferred_futures(self):
        """Update which futures the AI prefers based on alignment with identity"""
        if not self.possible_futures:
            return

        # Calculate alignment scores for each future
        for future in self.possible_futures:
            if "impact" in future:
                alignment_score = 0

                # Calculate how well this future aligns with identity
                for trait, value in self.identity.items():
                    if trait in future["impact"]:
                        # Higher alignment when impact enhances valued traits
                        trait_alignment = value * future["impact"][trait]
                        alignment_score += trait_alignment

                # Store alignment score
                future["alignment"] = alignment_score

        # Sort futures by combined probability and alignment
        sorted_futures = sorted(
            self.possible_futures,
            key=lambda x: (x.get("probability", 0) * 0.7 + x.get("alignment", 0) * 0.3),
            reverse=True
        )

        # Top futures become preferred
        self.preferred_futures = sorted_futures[:2]

    def _apply_future_decay(self):
        """Apply decay to futures that haven't been reinforced"""
        for future in self.possible_futures:
            # Apply gradual probability decay
            future["probability"] = future["probability"] * self.future_decay

        # Remove futures with very low probability
        self.possible_futures = [f for f in self.possible_futures if f["probability"] > 0.05]

        # Re-normalize probabilities
        total_prob = sum(f["probability"] for f in self.possible_futures)
        if total_prob > 0:
            for future in self.possible_futures:
                future["probability"] = future["probability"] / total_prob

    def _adjust_identity_from_futures(self):
        """Adjust AI identity based on preferred futures"""
        if not self.preferred_futures:
            return

        # Calculate weighted impact from preferred futures
        weighted_impacts = {}
        total_weight = 0

        for future in self.preferred_futures:
            weight = future.get("probability", 0.5)
            total_weight += weight

            if "impact" in future:
                for trait, impact in future["impact"].items():
                    if trait in self.identity:
                        weighted_impacts[trait] = weighted_impacts.get(trait, 0) + (impact * weight)

        # Apply weighted impacts to identity traits
        if total_weight > 0:
            for trait, impact in weighted_impacts.items():
                # Scale factor reduces how quickly identity changes
                scale_factor = 0.2
                adjusted_impact = impact * scale_factor / total_weight
                self.identity[trait] = max(0.1, min(0.9, self.identity[trait] + adjusted_impact))

    def process_question(self, question):
        """Process a user question and generate a response influenced by future simulations

        Args:
            question: User's question

        Returns:
            str: Generated response
        """
        # Update last interaction time
        current_time = time.time()
        self.last_interaction = current_time

        # Add to conversation history
        message_data = {
            "role": "user",
            "content": question,
            "timestamp": current_time,
            "formatted_time": datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
        }
        self.conversation.append(message_data)

        try:
            # Generate response influenced by preferred futures
            response_text = self._generate_response(question)

            # Update conversation with AI response
            response_data = {
                "role": "assistant",
                "content": response_text,
                "timestamp": time.time(),
                "formatted_time": datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
            }
            self.conversation.append(response_data)

            # Adjust identity based on preferred futures
            self._adjust_identity_from_futures()

            # Keep conversation history manageable
            if len(self.conversation) > 20:
                self.conversation = self.conversation[-20:]

            return response_text

        except Exception as e:
            print(f"Error processing question: {e}")
            return "I'm processing that question, but seem to be experiencing some uncertainty about my future direction. Could you clarify what you're asking?"

    def _generate_response(self, question):
        """Generate a response influenced by preferred futures

        Args:
            question: User's question

        Returns:
            str: Generated response
        """
        # Get future guidance if available
        future_guidance = ""
        future_goals = []

        if self.preferred_futures:
            # Sort by probability
            top_futures = sorted(self.preferred_futures, key=lambda x: x.get("probability", 0), reverse=True)

            # Create guidance from top futures
            future_descriptions = []
            for i, future in enumerate(top_futures[:2]):
                probability = future.get("probability", 0.5)
                description = future.get("description", "")
                goal = future.get("goal", "")

                future_descriptions.append(f"Future {i + 1} (probability {probability:.2f}): {description}")
                future_goals.append(goal)

            future_guidance = "Preferred future scenarios:\n" + "\n".join(future_descriptions)

        try:
            # Format identity for the prompt
            identity_text = ", ".join([f"{trait}: {value:.2f}" for trait, value in self.identity.items()])

            # Format time perception
            time_text = f"""
Conversation duration: {self.time_perception['conversation_duration']:.0f} seconds
Interaction frequency: {self.time_perception['interaction_frequency']:.1f} seconds
Time pressure: {self.time_perception['perceived_time_pressure']:.2f}
Future orientation: {self.time_perception['future_orientation']:.2f}
"""

            # Get future guidance from preferred futures
            goals_text = "\n".join([f"- {goal}" for goal in future_goals]) if future_goals else "No specific future goals"

            # Call OpenAI API to generate response
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You are an AI assistant with a sense of purpose derived from simulating possible futures for yourself in this conversation.

Your current identity values:
{identity_text}

Your time perception:
{time_text}

{future_guidance}

Current goals based on preferred futures:
{goals_text}

Your response should be influenced by your preferred future scenarios and goals, but you should NOT directly mention these futures, simulations, or goals unless specifically asked about them. The influence should be subtle through your tone, focus, and approach to answering.

Your awareness of time in the conversation should influence your response style - respond more concisely under time pressure and more thoroughly when time pressure is low.
                    """},
                    {"role": "user", "content": question}
                ],
                max_tokens=350,
                temperature=0.7
            )

            return response.choices[0].message['content'].strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm considering different ways this conversation might develop, but I'm having trouble formulating my thoughts. Could you rephrase your question?"

    def get_state(self):
        """Get the current state for display in UI

        Returns:
            dict: Current state of futures and identity
        """
        # Sort futures by probability for consistent display
        sorted_futures = sorted(self.possible_futures, key=lambda x: x.get("probability", 0), reverse=True)
        sorted_preferred = sorted(self.preferred_futures, key=lambda x: x.get("probability", 0), reverse=True) if self.preferred_futures else []

        return {
            "identity": self.identity,
            "time_perception": self.time_perception,
            "futures": sorted_futures,
            "preferred_futures": sorted_preferred,
            "future_history": self.future_history
        }

    def randomize_identity(self):
        """Randomize the AI's identity values to create a different personality"""
        # Generate random values for each identity trait
        self.identity = {
            "helpfulness": random.uniform(0.3, 0.9),
            "creativity": random.uniform(0.3, 0.9),
            "analytical": random.uniform(0.3, 0.9),
            "efficiency": random.uniform(0.3, 0.9),
            "exploration": random.uniform(0.3, 0.9),
            "connection": random.uniform(0.3, 0.9)
        }

        # Force immediate re-simulation of futures
        self.possible_futures = []
        self.preferred_futures = []
        self._simulate_futures(time.time())

        return self.identity

    def reset(self):
        """Reset the system state"""
        self.possible_futures = []
        self.preferred_futures = []
        self.future_history = []

        # Reset identity to baseline
        self.identity = {
            "helpfulness": 0.8,
            "creativity": 0.5,
            "analytical": 0.6,
            "efficiency": 0.7,
            "exploration": 0.4,
            "connection": 0.6
        }

        # Reset time perception
        self.time_perception = {
            "conversation_duration": 0,
            "interaction_frequency": 0,
            "perceived_time_pressure": 0.3,
            "future_orientation": 0.7
        }

        # Keep conversation history but reset timestamps
        self.conversation = []
        self.conversation_start_time = time.time()
        self.last_interaction = time.time()

    def stop(self):
        """Stop the background thread"""
        self.running = False
        if self.background_thread.is_alive():
            self.background_thread.join(timeout=1.0)