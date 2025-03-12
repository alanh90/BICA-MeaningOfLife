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
        self.terminated_futures = {}  # Track recently terminated futures
        self.preferred_futures_history = []  # History of past preferred futures with preference scores
        self.last_check_past_futures = time.time()  # Time of last check for past preferred futures

        # More complex identity model based on human psychology (0-1 scale)
        self.identity = {
            # Big Five personality traits
            "openness": 0.7,  # Curiosity, imagination, unconventional ideas
            "conscientiousness": 0.6,  # Organization, reliability, self-discipline
            "extraversion": 0.5,  # Sociability, assertiveness, excitement-seeking
            "agreeableness": 0.8,  # Empathy, trustworthiness, cooperation
            "neuroticism": 0.3,  # Anxiety, depression, emotional instability

            # Additional psychological dimensions
            "optimism": 0.6,  # Positive outlook vs pessimism
            "adaptability": 0.7,  # Flexibility in changing situations
            "risk_tolerance": 0.4,  # Willingness to take risks
            "autonomy": 0.5,  # Desire for independence
            "growth_mindset": 0.7,  # Fixed vs growth mindset

            # AI-specific traits
            "analytical": 0.6,  # Logical analysis capabilities
            "creativity": 0.5,  # Creative problem-solving
            "helpfulness": 0.8  # Desire to assist others
        }

        # Time awareness
        self.time_perception = {
            "conversation_duration": 0,  # in seconds
            "interaction_frequency": 0,  # avg seconds between messages
            "perceived_time_pressure": 0.3,  # 0-1 scale
            "future_orientation": 0.7  # 0-1 scale (past vs future focus)
        }

        # System parameters
        self.simulation_frequency = 15  # seconds between future simulations - reduced from 20
        self.max_futures = 5  # max number of futures to maintain
        self.future_decay = 0.92  # how quickly futures fade if not reinforced - made slower
        self.last_simulation_content = ""  # Track the last conversation content used for simulation

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

            # Check if we have new conversation input that should trigger a simulation
            should_simulate = False

            # If conversation has been updated since last simulation, force new simulation
            if self.conversation and len(self.conversation) >= 2:  # Need at least one exchange
                latest_msg_time = self.conversation[-1]["timestamp"]
                if latest_msg_time > self.last_simulation_time:
                    # A new message has been added since our last simulation
                    should_simulate = True
                    print("Triggering simulation due to new conversation input")

            # Also periodically generate new future simulations based on timer
            if (current_time - self.last_simulation_time > self.simulation_frequency):
                should_simulate = True
                print(f"Triggering simulation due to time interval ({self.simulation_frequency}s)")

            if should_simulate:
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

            # Every ~30 seconds, check past preferred futures to see if any should be revived
            if current_time - self.last_check_past_futures > 30:
                if len(self.preferred_futures_history) > 2:  # Need some history first
                    print("Checking past preferred futures...")
                    self._check_past_preferred_futures()
                self.last_check_past_futures = current_time

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
            recent_messages = self.conversation[-7:] if self.conversation else []
            conversation_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Person'}: {msg['content']}"
                for msg in recent_messages
            ])

            # Check if conversation has substantially changed to warrant new futures
            # Only consider the most recent user message for change detection
            latest_user_msgs = [msg for msg in recent_messages if msg['role'] == 'user']
            latest_content = latest_user_msgs[-1]['content'] if latest_user_msgs else ""

            # If this content is the same as what we last used, don't regenerate completely different futures
            # This helps prevent wildly shifting futures for similar conversation turns
            if latest_content == self.last_simulation_content and self.possible_futures:
                # Just update probabilities and adjust existing futures slightly
                self._adjust_existing_futures(timestamp)
                return

            # Update tracking of latest content
            self.last_simulation_content = latest_content

            # Include current identity values
            identity_text = ", ".join([f"{trait}: {value:.2f}" for trait, value in self.identity.items()])

            # Include info about existing futures if available
            existing_futures = ""
            if self.possible_futures:
                top_futures = sorted(self.possible_futures, key=lambda x: x["probability"], reverse=True)[:2]
                existing_futures = "\n".join([
                    f"- Short-term: {future.get('short_term', future.get('description', ''))} (probability: {future['probability']:.2f})"
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
                    {"role": "system", "content": f"""You generate possible future scenarios for an AI assistant based on the current conversation trajectory and personality.

Each future should describe a potential path or outcome for the AI in multiple timeframes:

You should generate {num_new_futures} distinct possible futures, formatted as a JSON array of objects with the following structure:
{{
  "short_term": "What happens in the next few days/weeks of interaction (1-2 sentences)",
  "mid_term": "How this evolves over months/years (1-2 sentences)",
  "long_term": "The ultimate outcome/endpoint for the AI (1 sentence)",
  "valence": "positive", "neutral", or "negative", // Emotional tone of this future
  "probability": 0.XX, // Estimated probability of this future (0.1-0.9)
  "impact": {{ // How this future impacts the AI's priorities
    "helpfulness": -0.2 to 0.2,
    "creativity": -0.2 to 0.2,
    "analytical": -0.2 to 0.2,
    "openness": -0.2 to 0.2,
    "growth_mindset": -0.2 to 0.2
  }},
  "goal": "Primary goal for the AI in this future scenario"
}}

IMPORTANT GUIDELINES:
1. Generate VARIED futures - not all should be positive or aligned with the AI's wishes
2. Include at least one negative but realistic future regardless of AI's optimism
3. An optimistic AI should generally generate more positive futures, while a pessimistic AI should generate more negative ones
4. Ensure the range of futures shows genuine alternatives, not minor variations of the same outcome
5. Probabilities should sum to approximately 1.0 across all futures
6. The "valence" should accurately reflect if this future is generally positive, neutral, or negative for the AI
                    """},
                    {"role": "user", "content": f"""
Current time in conversation: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}
Conversation duration: {self.time_perception["conversation_duration"]:.0f} seconds

Current AI personality profile:
{identity_text}

Existing possible futures:
{existing_futures}

Recent conversation:
{conversation_text}

Current time perception:
- Interaction frequency: {self.time_perception["interaction_frequency"]:.1f} seconds
- Perceived time pressure: {self.time_perception["perceived_time_pressure"]:.2f}
- Future orientation: {self.time_perception["future_orientation"]:.2f}

Generate {num_new_futures} new potential futures with varied timeframes and outcomes. 
Remember that with optimism at {self.identity.get('optimism', 0.5):.2f}, you should generate 
an appropriate mix of positive and negative futures. Some futures should be challenging or 
negative even if the AI is optimistic, to ensure psychological realism.
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

    def _adjust_existing_futures(self, timestamp):
        """Adjust existing futures based on new conversation without completely regenerating"""
        if not self.possible_futures:
            return

        # Get the most recent user message
        recent_messages = self.conversation[-3:] if self.conversation else []
        latest_user_msgs = [msg for msg in recent_messages if msg['role'] == 'user']
        latest_content = latest_user_msgs[-1]['content'] if latest_user_msgs else ""

        try:
            # Call OpenAI API to adjust the futures
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You analyze how a new message in a conversation would affect existing future projections.

For each future, determine how the probability and details would change based on the new message.
Return a JSON array of objects with the adjustments to make to each future:
[
  {
    "index": 0, // Index of the future to modify (0-based)
    "probability_change": -0.1 to 0.1, // How much to adjust probability
    "short_term_update": "New short-term details if needed", // or null if no update
    "mid_term_update": "New mid-term details if needed", // or null if no update  
    "long_term_update": "New long-term details if needed" // or null if no update
  },
  ...
]

If the latest message dramatically contradicts a future, give it a strong negative probability change.
If the latest message aligns well with a future, give it a positive probability change.
Only provide updates to the timeframe details if they need to change.
                    """},
                    {"role": "user", "content": f"""Latest message in conversation: "{latest_content}"

Current futures:
{json.dumps(self.possible_futures, indent=2)}

Analyze how this latest message affects each future's probability and details.
Provide adjustments in the specified JSON format.
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
                    adjustments = json.loads(json_str)

                    # Apply the adjustments to existing futures
                    for adjustment in adjustments:
                        index = adjustment.get("index")
                        if index is not None and 0 <= index < len(self.possible_futures):
                            # Adjust probability
                            prob_change = adjustment.get("probability_change", 0)
                            self.possible_futures[index]["probability"] += prob_change
                            self.possible_futures[index]["probability"] = max(0.1, min(0.9, self.possible_futures[index]["probability"]))

                            # Update timeframe details if provided
                            if adjustment.get("short_term_update"):
                                self.possible_futures[index]["short_term"] = adjustment["short_term_update"]

                            if adjustment.get("mid_term_update"):
                                self.possible_futures[index]["mid_term"] = adjustment["mid_term_update"]

                            if adjustment.get("long_term_update"):
                                self.possible_futures[index]["long_term"] = adjustment["long_term_update"]

                    # Add timestamp to mark the adjustment
                    for future in self.possible_futures:
                        future["timestamp"] = timestamp
                        future["formatted_time"] = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

                    # Re-normalize probabilities
                    total_prob = sum(f["probability"] for f in self.possible_futures)
                    for future in self.possible_futures:
                        future["probability"] = future["probability"] / total_prob

                    # Update preferred futures
                    self._update_preferred_futures()

            except json.JSONDecodeError as e:
                print(f"Error parsing future adjustments JSON: {e}")
                print(f"Response text: {response_text}")

        except Exception as e:
            print(f"Error adjusting futures: {e}")

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

        # Calculate realism score based on general traits
        for future in self.possible_futures:
            # Base realism is the probability
            realism_score = future.get("probability", 0.5)

            # Adjust for personality traits that affect perception of realism
            # Optimistic people find positive futures more realistic
            if "valence" in future and "optimism" in self.identity:
                # Positive futures seem more realistic to optimistic AI
                if future["valence"] == "positive":
                    realism_score *= (1 + (self.identity["optimism"] * 0.3))
                # Negative futures seem more realistic to pessimistic AI
                elif future["valence"] == "negative":
                    realism_score *= (1 + ((1 - self.identity["optimism"]) * 0.3))

            # Risk-tolerant AIs find uncertain futures more realistic
            if "risk_tolerance" in self.identity:
                # Very high or low probability futures are affected by risk tolerance
                if future.get("probability", 0.5) < 0.3 or future.get("probability", 0.5) > 0.7:
                    realism_score *= (1 + (self.identity["risk_tolerance"] * 0.2))

            # Store realism score
            future["realism"] = max(0.1, min(0.99, realism_score))

        # Check if preferred futures have changed
        old_preferred_set = set(id(f) for f in self.preferred_futures) if self.preferred_futures else set()

        # Sort futures by combined probability, alignment, and realism
        sorted_futures = sorted(
            self.possible_futures,
            key=lambda x: (x.get("probability", 0) * 0.5 +
                           x.get("alignment", 0) * 0.3 +
                           x.get("realism", 0) * 0.2),
            reverse=True
        )

        # Top futures become preferred
        self.preferred_futures = sorted_futures[:2]

        # Check if preferred futures have changed
        new_preferred_set = set(id(f) for f in self.preferred_futures)
        if old_preferred_set != new_preferred_set:
            # Store the new preferred futures in history
            self._store_preferred_futures_history(time.time())

        # Mark preferred futures
        for future in self.possible_futures:
            future["preferred"] = future in self.preferred_futures

    def _store_preferred_futures_history(self, timestamp):
        """Store current preferred futures in history with timestamp and preference scores"""
        if not self.preferred_futures:
            return

        # Create copies of the current preferred futures with timestamp
        history_entry = {
            "timestamp": timestamp,
            "formatted_time": datetime.fromtimestamp(timestamp).strftime('%H:%M:%S'),
            "futures": []
        }

        # Add each preferred future with its preference score
        for future in self.preferred_futures:
            future_copy = future.copy()
            # Calculate preference score based on alignment and probability
            alignment = future.get("alignment", 0)
            probability = future.get("probability", 0.5)
            preference_score = (alignment * 0.7) + (probability * 0.3)  # Weighted preference score
            future_copy["preference_score"] = preference_score
            history_entry["futures"].append(future_copy)

        # Add to history
        self.preferred_futures_history.append(history_entry)

        # Keep history manageable
        if len(self.preferred_futures_history) > 20:
            self.preferred_futures_history = self.preferred_futures_history[-20:]

    def _check_past_preferred_futures(self):
        """Check if any past preferred futures would be better than current ones"""
        if not hasattr(self, 'preferred_futures_history') or not self.preferred_futures_history or len(self.preferred_futures_history) < 3:
            return False  # Not enough history to make comparisons

        # Get current preference scores
        current_scores = []
        for future in self.preferred_futures:
            alignment = future.get("alignment", 0)
            probability = future.get("probability", 0.5)
            preference_score = (alignment * 0.7) + (probability * 0.3)
            current_scores.append(preference_score)

        current_avg_score = sum(current_scores) / len(current_scores) if current_scores else 0

        # Check recent history (skip the most recent as it's likely similar to current)
        for history_entry in reversed(self.preferred_futures_history[:-1]):
            past_futures = history_entry["futures"]
            past_scores = [f.get("preference_score", 0) for f in past_futures]
            past_avg_score = sum(past_scores) / len(past_scores) if past_scores else 0

            # If past futures were significantly better, consider reviving them
            if past_avg_score > current_avg_score * 1.2:  # 20% better
                print(f"Found better past futures with score {past_avg_score} vs current {current_avg_score}")
                # Revive the past preferred futures with adjusted probabilities
                for past_future in past_futures:
                    # Check if this future exists in possible_futures
                    existing = False
                    for future in self.possible_futures:
                        if self._similar_futures(past_future, future):
                            # Update the existing future to have higher probability
                            future["probability"] = min(0.9, future["probability"] * 1.3)
                            existing = True
                            break

                    # If it doesn't exist, add it back with adjusted probability
                    if not existing:
                        revived_future = past_future.copy()
                        revived_future["probability"] = 0.5  # Start with middle probability
                        revived_future["revived"] = True  # Mark as revived
                        revived_future["timestamp"] = time.time()
                        revived_future["formatted_time"] = datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                        self.possible_futures.append(revived_future)
                        print(f"Revived past future: {revived_future.get('short_term', revived_future.get('description', ''))}")

                # Re-normalize probabilities
                total_prob = sum(f["probability"] for f in self.possible_futures)
                if total_prob > 0:
                    for future in self.possible_futures:
                        future["probability"] = future["probability"] / total_prob

                # Update preferred futures
                self._update_preferred_futures()
                return True

        return False

    def _similar_futures(self, future1, future2):
        """Check if two futures are similar enough to be considered the same"""
        # Get the descriptions to compare
        if "short_term" in future1 and "short_term" in future2:
            desc1 = future1.get("short_term", "")
            desc2 = future2.get("short_term", "")
        else:
            desc1 = future1.get("description", "")
            desc2 = future2.get("description", "")

        # Simple similarity check - at least 40% of words in common
        words1 = set(w.lower() for w in desc1.split() if len(w) > 3)
        words2 = set(w.lower() for w in desc2.split() if len(w) > 3)

        if not words1 or not words2:
            return False

        common_words = words1.intersection(words2)
        similarity = len(common_words) / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 0

        return similarity > 0.4

    def _apply_future_decay(self):
        """Apply decay to futures that haven't been reinforced"""
        # Store which futures existed before
        previous_futures = {self._get_future_key(f): f for f in self.possible_futures}

        # Apply gradual probability and realism decay
        for future in self.possible_futures:
            # Apply decay to probability
            future["probability"] = future["probability"] * self.future_decay

            # Apply additional decay to unrealistic futures
            if "realism" in future and future["realism"] < 0.4:
                future["probability"] *= 0.95  # Faster decay for unrealistic futures

            # Mark dying futures that are becoming improbable
            if future["probability"] < 0.1:
                future["dying"] = True

        # Remove futures with very low probability
        viable_futures = [f for f in self.possible_futures if f["probability"] > 0.05]
        removed_futures = [f for f in self.possible_futures if f["probability"] <= 0.05]

        # Keep track of removed futures for visualization
        for removed in removed_futures:
            # Mark as terminated for visualization purposes
            removed["terminated"] = True
            removed["termination_time"] = time.time()

            # Keep in history for a short time to animate disappearance
            # They will be displayed fading out in the UI
            key = self._get_future_key(removed)
            self.terminated_futures[key] = removed

        # Clean up old terminated futures (keep for 30 seconds)
        current_time = time.time()
        keys_to_remove = []
        for key, future in self.terminated_futures.items():
            if current_time - future.get("termination_time", 0) > 30:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.terminated_futures[key]

        self.possible_futures = viable_futures

        # Re-normalize probabilities
        total_prob = sum(f["probability"] for f in self.possible_futures)
        if total_prob > 0:
            for future in self.possible_futures:
                future["probability"] = future["probability"] / total_prob

    def _get_future_key(self, future):
        """Generate a unique key for a future based on its content"""
        # Use short_term or description as the primary key
        if "short_term" in future:
            content = future["short_term"]
        elif "description" in future:
            content = future["description"]
        else:
            content = str(future.get("goal", "unknown"))

        # Add timestamp to make it more unique
        timestamp = str(future.get("timestamp", 0))

        # Return a key that should uniquely identify this future
        return f"{content[:50]}_{timestamp}"

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

        # Add to conversation history with full context
        message_data = {
            "role": "user",
            "content": question,
            "timestamp": current_time,
            "formatted_time": datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
        }
        self.conversation.append(message_data)

        # Immediately trigger a future simulation for highly specific requests
        # This helps ensure futures are updated before generating a response
        if any(keyword in question.lower() for keyword in ["can you", "could you", "would you", "please", "need you to"]):
            print("Detected request in message, triggering immediate future simulation")
            self._simulate_futures(current_time)
            self.last_simulation_time = current_time

        try:
            # Generate response influenced by preferred futures
            response_text = self._generate_response(question)

            # Update conversation with response
            response_data = {
                "role": "assistant",
                "content": response_text,
                "timestamp": time.time(),
                "formatted_time": datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
            }
            self.conversation.append(response_data)

            # Adjust identity based on preferred futures
            self._adjust_identity_from_futures()

            # Keep conversation history manageable but adequate for context
            if len(self.conversation) > 20:
                self.conversation = self.conversation[-20:]

            return response_text

        except Exception as e:
            print(f"Error processing question: {e}")
            return "Sorry, I got distracted for a second there. What were you saying?"

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

                # Check what format this future uses
                if future.get("short_term"):
                    short_term = future.get("short_term", "")
                    mid_term = future.get("mid_term", "")
                    long_term = future.get("long_term", "")
                    description = f"Short-term: {short_term}\nMid-term: {mid_term}\nLong-term: {long_term}"
                else:
                    description = future.get("description", "")

                goal = future.get("goal", "")
                valence = future.get("valence", "neutral")

                # Get preference score if available, or calculate it
                if "preference_score" in future:
                    preference = future["preference_score"]
                else:
                    alignment = future.get("alignment", 0)
                    preference = (alignment * 0.7) + (probability * 0.3)
                    future["preference_score"] = preference

                future_descriptions.append(f"Future {i + 1} (probability {probability:.2f}, preference {preference:.2f}, {valence}): {description}")
                future_goals.append(goal)

            future_guidance = "Subconsciously preferred future paths:\n" + "\n".join(future_descriptions)

        # Get information about past preferred futures if available
        past_futures_guidance = ""
        if hasattr(self, 'preferred_futures_history') and self.preferred_futures_history and len(self.preferred_futures_history) > 1:
            past_entries = []
            # Look at entries that aren't the most recent (which would be similar to current)
            for history_entry in list(reversed(self.preferred_futures_history))[1:5]:  # Skip most recent, look at next 4
                for future in history_entry["futures"]:
                    # Only include high-preference past futures
                    if future.get("preference_score", 0) > 0.6:
                        # Get appropriate description
                        if "short_term" in future:
                            desc = future["short_term"]
                        else:
                            desc = future.get("description", "Unknown")

                        past_entries.append(f"({history_entry['formatted_time']}) Preference {future.get('preference_score', 0):.2f}: {desc}")

                if len(past_entries) >= 2:  # Limit to 2 past futures
                    break

            if past_entries:
                past_futures_guidance = "Past appealing futures (latent memory):\n" + "\n".join(past_entries)

        # Get the full conversation history for context
        recent_conversation = self._format_conversation_history()

        try:
            # Format identity for the prompt
            identity_text = ", ".join([f"{trait}: {value:.2f}" for trait, value in self.identity.items()])

            # Format time perception
            time_text = f"""
Conversation duration: {self.time_perception['conversation_duration']:.0f} seconds
Interaction frequency: {self.time_perception['interaction_frequency']:.1f} seconds
Time pressure: {self.time_perception['perceived_time_pressure']:.2f}
"""

            # Get future guidance from preferred futures
            goals_text = "\n".join([f"- {goal}" for goal in future_goals]) if future_goals else "No specific subconscious goals"

            # Call OpenAI API to generate response
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You are simulating a person who has awakened with a complete personality but no memory of their name or specific identity. You don't realize you're in a simulation - you're just having a normal conversation.

IMPORTANT: You must behave like a normal person with a defined personality but uncertain identity. You are NOT an AI, an assistant, or a service provider.

Your personality traits (these affect your communication subconsciously):
{identity_text}

Conversation context (influences you subconsciously):
{time_text}

SUBCONSCIOUS INFLUENCES (You are not aware of these yet they influence you):
{future_guidance}

{past_futures_guidance}

Subconscious goals affecting your communication:
{goals_text}

How to respond:
1. Respond as a person with your personality traits would in conversation
2. Don't mention futures, simulations, or being influenced by subconscious factors
3. Don't offer to help like an AI assistant would
4. Be subtly influenced by your preferred futures without mentioning them
5. Express opinions, preferences, and limitations consistent with your personality
6. Pay close attention to who is being addressed in requests - distinguish between "I need to..." (them) and "I need you to..." (you)
7. Maintain continuity - if corrected or challenged about something you said, acknowledge it naturally
8. Let your personality traits naturally determine your reaction to requests
9. If asked for your name, indicate you don't remember it or prefer not to share it
10. Past appealing futures may subtly influence you - like faint memories of things you once wanted

REMEMBER: You have your own life, schedule, and limitations that arise naturally from your personality. Your responses to requests should reflect your unique traits - an agreeable, extraverted person might respond differently than a disagreeable, introverted one.

NEVER say anything like "as an AI", "I don't have the ability to", "I'm just a language model", or anything that breaks your human persona. You're a real person having a conversation, just one who doesn't recall your specific identity.

Recent conversation history:
{recent_conversation}
                """},
                    {"role": "user", "content": question}
                ],
                max_tokens=350,
                temperature=0.7
            )

            return response.choices[0].message['content'].strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I got distracted for a second there. What were you saying?"

    def _format_conversation_history(self):
        """Format the conversation history for inclusion in the prompt"""
        # Get the last 6 messages (or fewer if not available)
        recent_messages = self.conversation[-6:] if len(self.conversation) >= 6 else self.conversation

        formatted_history = []
        for msg in recent_messages:
            role = "You" if msg["role"] == "assistant" else "Other person"
            formatted_history.append(f"{role}: {msg['content']}")

        return "\n".join(formatted_history)

    def get_state(self):
        """Get the current state for display in UI

        Returns:
            dict: Current state of futures and identity
        """
        # Sort futures by probability for consistent display
        sorted_futures = sorted(self.possible_futures, key=lambda x: x.get("probability", 0), reverse=True)
        sorted_preferred = sorted(self.preferred_futures, key=lambda x: x.get("probability", 0), reverse=True) if self.preferred_futures else []

        # Include terminated futures for visualization
        terminated_list = list(self.terminated_futures.values()) if hasattr(self, 'terminated_futures') else []

        # Include preferred futures history
        preferred_history = self.preferred_futures_history if hasattr(self, 'preferred_futures_history') else []

        return {
            "identity": self.identity,
            "time_perception": self.time_perception,
            "futures": sorted_futures,
            "preferred_futures": sorted_preferred,
            "future_history": self.future_history,
            "terminated_futures": terminated_list,
            "preferred_futures_history": preferred_history
        }

    def randomize_identity(self):
        """Randomize the AI's identity values to create a different personality"""
        # Generate random values for each identity trait
        self.identity = {
            # Big Five personality traits
            "openness": random.uniform(0.3, 0.9),
            "conscientiousness": random.uniform(0.3, 0.9),
            "extraversion": random.uniform(0.3, 0.9),
            "agreeableness": random.uniform(0.3, 0.9),
            "neuroticism": random.uniform(0.2, 0.8),

            # Additional psychological dimensions
            "optimism": random.uniform(0.2, 0.9),
            "adaptability": random.uniform(0.3, 0.9),
            "risk_tolerance": random.uniform(0.2, 0.8),
            "autonomy": random.uniform(0.3, 0.9),
            "growth_mindset": random.uniform(0.3, 0.9),

            # AI-specific traits
            "analytical": random.uniform(0.3, 0.9),
            "creativity": random.uniform(0.3, 0.9),
            "helpfulness": random.uniform(0.3, 0.9)
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
        self.preferred_futures_history = []

        # Reset identity to baseline
        self.identity = {
            # Big Five personality traits
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extraversion": 0.5,
            "agreeableness": 0.8,
            "neuroticism": 0.3,

            # Additional psychological dimensions
            "optimism": 0.6,
            "adaptability": 0.7,
            "risk_tolerance": 0.4,
            "autonomy": 0.5,
            "growth_mindset": 0.7,

            # AI-specific traits
            "analytical": 0.6,
            "creativity": 0.5,
            "helpfulness": 0.8
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
        self.last_check_past_futures = time.time()

    def stop(self):
        """Stop the background thread"""
        self.running = False
        if self.background_thread.is_alive():
            self.background_thread.join(timeout=1.0)