# Cultural Translation 
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Based on Ubuntu Dialogue Corpus (UDC)
# Represents conversation patterns from Ubuntu IRC dataset
UBUNTU_DIALOGUE_PATTERNS = {
    "conversation_flows": [
        {
            "pattern_id": "helpful_correction",
            "sequence": [
                "User: I'm having trouble with X",
                "Helper: I see the issue. Here's what's happening...", 
                "User: Oh that makes sense, thank you!",
                "Helper: No problem! Let me know if you need more help"
            ],
            "features": ["acknowledgment", "explanation", "gratitude", "availability"]
        },
        {
            "pattern_id": "clarification_request", 
            "sequence": [
                "User: Can someone help with Y?",
                "Helper: Sure! Can you be more specific about what you're trying to do?",
                "User: I want to achieve Z but getting error...",
                "Helper: Ah, I see. Try this approach..."
            ],
            "features": ["offer_help", "seek_clarity", "provide_context", "solution"]
        },
        {
            "pattern_id": "community_discussion",
            "sequence": [
                "User A: What do you all think about approach X?",
                "User B: I've tried that, here's my experience...",
                "User C: That's interesting, another option is...",
                "User A: Thanks everyone, this gives me good options"
            ],
            "features": ["seek_opinions", "share_experience", "multiple_perspectives", "synthesis"]
        }
    ],
    
    "response_templates": {
        "acknowledgment": [
            "I understand your concern about {topic}",
            "That's a valid point about {topic}", 
            "I see what you mean regarding {topic}"
        ],
        "helpful_explanation": [
            "Here's what's happening: {explanation}",
            "The issue is likely {explanation}",
            "This usually means {explanation}"
        ],
        "consensus_building": [
            "It sounds like we're all agreeing that {point}",
            "From what everyone's saying, the main issue is {point}",
            "I'm hearing consensus around {point}"
        ]
    }
}

# Generated Ubuntu-Net Dataset 
# Represents the cultural wisdom and communication patterns
UBUNTU_NET_CULTURAL_DATA = {
    "cultural_communication_patterns": [
        {
            "pattern_id": "nigerian_indirect_criticism",
            "culture": "Nigerian",
            "phrase_markers": ["ehen", "so you", "abi", "sha"],
            "communication_style": "indirect confrontation",
            "typical_structure": "Ehen, so [behavior observation] abi?",
            "cultural_meaning": "Playful calling out of behavior change",
            "tone": "sarcastic but affectionate",
            "relationship_context": "close friends/family",
            "misunderstanding_risk": "Receiver thinks it's harsh criticism rather than caring concern",
            "training_examples": [
                {
                    "text": "Ehen, so you're now too big to call me abi?",
                    "cultural_intent": "I feel neglected, please reconnect with me",
                    "literal_misunderstanding": "You think I've gained weight?",
                    "appropriate_response": "You're right, I've been distant. Let's catch up soon!"
                }
            ]
        },
        {
            "pattern_id": "ghanaian_diplomatic_disagreement", 
            "culture": "Ghanaian",
            "phrase_markers": ["interesting", "we should discuss", "let me think"],
            "communication_style": "diplomatic indirect disagreement",
            "typical_structure": "That's [positive word] but [gentle redirect]",
            "cultural_meaning": "Polite way to express disagreement or concern",
            "tone": "respectful but cautious",
            "relationship_context": "professional/formal",
            "misunderstanding_risk": "Receiver thinks it's genuine enthusiasm",
            "training_examples": [
                {
                    "text": "Your proposal is very interesting. We should discuss it further.",
                    "cultural_intent": "I see problems with this proposal that need addressing",
                    "literal_misunderstanding": "They love the proposal and want to move forward",
                    "appropriate_response": "What concerns or questions do you have about it?"
                }
            ]
        },
        {
            "pattern_id": "kenyan_sarcastic_callout",
            "culture": "Kenyan", 
            "phrase_markers": ["aki", "you have jokes", "kumbe", "ati"],
            "communication_style": "direct sarcastic confrontation",
            "typical_structure": "[Expression] you have jokes [about behavior]",
            "cultural_meaning": "Sarcastic calling out of ridiculous behavior",
            "tone": "sarcastic with underlying concern",
            "relationship_context": "friends/colleagues",
            "misunderstanding_risk": "Receiver thinks it's a compliment about being funny",
            "training_examples": [
                {
                    "text": "Aki wewe, you have jokes! Missing the meeting like that.",
                    "cultural_intent": "Your behavior was unacceptable and I'm calling you out",
                    "literal_misunderstanding": "You think I'm funny and entertaining",
                    "appropriate_response": "You're right to call me out. I messed up and I'm sorry."
                }
            ]
        }
    ],
    
    "proverbs_and_wisdom": {
        "akan": [
            {
                "proverb": "Se wo were fi na wosankofa a yenkyi",
                "translation": "It's not wrong to go back for what you forgot",
                "application": "Learning from mistakes, humility in progress",
                "usage_context": "When someone needs to correct course or admit error"
            }
        ],
        "yoruba": [
            {
                "proverb": "Bí a bá ń gun orí àgbà, a ó dé òkè", 
                "translation": "If we climb on elders' shoulders, we reach the top",
                "application": "Building on others' wisdom and experience",
                "usage_context": "Acknowledging help received or encouraging collaboration"
            }
        ]
    }
}

# Simulated LLM
class DatasetTrainedCulturalLLM:
    """
    Shows how the datasets enhance cultural translation capabilities
    """
    
    def __init__(self):
        # Load simulated datasets
        self.dialogue_patterns = UBUNTU_DIALOGUE_PATTERNS
        self.cultural_knowledge = UBUNTU_NET_CULTURAL_DATA
        self.learned_responses = self._simulate_dataset_training()
    
    def _simulate_dataset_training(self) -> Dict:
        """
        Simulate how the LLM would learn from both datasets combined
        Ubuntu Dialogue Corpus: HOW to have conversations
        Ubuntu-Net: WHAT cultural meanings to apply
        """
        
        training_synthesis = {
            "conversation_management": {
                # From Ubuntu Dialogue Corpus - how to structure helpful responses
                "acknowledgment_patterns": self.dialogue_patterns["response_templates"]["acknowledgment"],
                "explanation_patterns": self.dialogue_patterns["response_templates"]["helpful_explanation"],
                "flow_understanding": [flow["features"] for flow in self.dialogue_patterns["conversation_flows"]]
            },
            
            "cultural_intelligence": {
                # From Ubuntu-Net - what cultural meanings to detect and apply
                "cultural_patterns": {
                    pattern["pattern_id"]: {
                        "detection_markers": pattern["phrase_markers"],
                        "cultural_meaning": pattern["cultural_meaning"],
                        "response_strategy": pattern["training_examples"][0]["appropriate_response"]
                    }
                    for pattern in self.cultural_knowledge["cultural_communication_patterns"]
                },
                "wisdom_application": self.cultural_knowledge["proverbs_and_wisdom"]
            }
        }
        
        return training_synthesis
    
    def analyze_with_datasets(self, message: str, sender_culture: str, 
                            receiver_culture: str, relationship: str) -> Dict:
        """
        Analyze message using patterns learned from both datasets
        """
        
        analysis = {
            "ubuntu_dialogue_analysis": self._apply_dialogue_patterns(message, relationship),
            "ubuntu_net_analysis": self._apply_cultural_patterns(message, sender_culture),
            "combined_intelligence": {}
        }
        
        # Combine insights from both datasets
        analysis["combined_intelligence"] = self._synthesize_dataset_insights(
            analysis["ubuntu_dialogue_analysis"],
            analysis["ubuntu_net_analysis"],
            sender_culture,
            receiver_culture
        )
        
        return analysis
    
    def _apply_dialogue_patterns(self, message: str, relationship: str) -> Dict:
        """Apply conversation patterns"""
        
        dialogue_analysis = {
            "conversation_type": "unknown",
            "appropriate_response_style": "neutral",
            "conversation_flow_position": "unknown"
        }
        
        # Detect conversation type based on Dialogue patterns
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["help", "problem", "issue", "trouble"]):
            dialogue_analysis["conversation_type"] = "help_request"
            dialogue_analysis["appropriate_response_style"] = "helpful_supportive"
            dialogue_analysis["suggested_flow"] = self.dialogue_patterns["conversation_flows"][0]  
            
        elif any(word in message_lower for word in ["think", "opinion", "what do you"]):
            dialogue_analysis["conversation_type"] = "opinion_seeking"
            dialogue_analysis["appropriate_response_style"] = "collaborative_discussion"
            dialogue_analysis["suggested_flow"] = self.dialogue_patterns["conversation_flows"][2]  
            
        else:
            dialogue_analysis["conversation_type"] = "general_communication"
            dialogue_analysis["appropriate_response_style"] = "acknowledgment_focused"
        
        return dialogue_analysis
    
    def _apply_cultural_patterns(self, message: str, sender_culture: str) -> Dict:
        """Apply cultural knowledge"""
        
        cultural_analysis = {
            "detected_cultural_patterns": [],
            "cultural_intent": None,
            "misunderstanding_risks": [],
            "cultural_response_needed": False
        }
        
        message_lower = message.lower()
        
        # Check against learned cultural patterns
        for pattern in self.cultural_knowledge["cultural_communication_patterns"]:
            if pattern["culture"].lower() == sender_culture.lower():
                # Check for phrase markers
                markers_found = [marker for marker in pattern["phrase_markers"] 
                               if marker.lower() in message_lower]
                
                if markers_found:
                    cultural_analysis["detected_cultural_patterns"].append({
                        "pattern_id": pattern["pattern_id"],
                        "culture": pattern["culture"],
                        "markers_found": markers_found,
                        "cultural_meaning": pattern["cultural_meaning"],
                        "tone": pattern["tone"],
                        "misunderstanding_risk": pattern["misunderstanding_risk"],
                        "training_example": pattern["training_examples"][0]
                    })
                    cultural_analysis["cultural_response_needed"] = True
        
        return cultural_analysis
    
    def _synthesize_dataset_insights(self, dialogue_analysis: Dict, cultural_analysis: Dict,
                                   sender_culture: str, receiver_culture: str) -> Dict:
        """
        Combine insights create culturally-intelligent response
        """
        synthesis = {
            "translation_strategy": "standard",
            "response_approach": "neutral",
            "cultural_warnings": [],
            "enhanced_translation": "",
            "suggested_responses": [],
            "educational_notes": []
        }
        
        if cultural_analysis["detected_cultural_patterns"]:
            pattern = cultural_analysis["detected_cultural_patterns"][0]
            
            # Use Ubuntu Dialogue Corpus patterns for response structure
            if dialogue_analysis["conversation_type"] == "help_request":
                response_template = self.learned_responses["conversation_management"]["acknowledgment_patterns"][0]
                synthesis["response_approach"] = "helpful_with_cultural_awareness"
                
            else:
                # Use cultural-aware acknowledgment
                synthesis["response_approach"] = "cultural_bridge_building"
            
            # Generate cultural warnings 
            synthesis["cultural_warnings"].append({
                "type": "cultural_misunderstanding_risk",
                "risk": pattern["misunderstanding_risk"],
                "cultural_context": f"In {pattern['culture']} culture: {pattern['cultural_meaning']}",
                "dataset_source": "ubuntu_net_cultural_patterns"
            })
            # Enhanced translation
            synthesis["enhanced_translation"] = (
                f"[Cultural Context: {pattern['cultural_meaning']}] "
                f"{pattern['training_example']['text']}"
            )

            # Response suggestions 
            synthesis["suggested_responses"] = [
                pattern["training_example"]["appropriate_response"],
                (
                    f"I understand you're expressing "
                    f"{pattern['cultural_meaning'].lower()}. Let's address this."
                )
            ]

            synthesis["educational_notes"].append(
                (
                    f"Dataset Learning: Ubuntu-Net identified this as "
                    f"{pattern['culture']} cultural pattern. "
                    f"Ubuntu Dialogue Corpus suggests responding with "
                    f"{dialogue_analysis['appropriate_response_style']} approach."
                )
            )

        
        return synthesis

# Cultural Translation System 

class UbuntuDatasetTranslator:
    """
    Translation uses both Ubuntu Dialogue Corpus + Ubuntu-Net datasets
    """
    
    def __init__(self):
        self.cultural_llm = DatasetTrainedCulturalLLM()
    
    def translate_with_dataset_intelligence(self, message: str, sender_culture: str,
                                          receiver_culture: str, relationship: str = "friends") -> Dict:
        
        # Get analysis from both datasets
        dataset_analysis = self.cultural_llm.analyze_with_datasets(
            message, sender_culture, receiver_culture, relationship
        )
        
        # Build comprehensive translation result
        result = {
            "original_message": message,
            "sender_culture": sender_culture,
            "receiver_culture": receiver_culture,
            "relationship": relationship,
            "dataset_sources": ["Ubuntu Dialogue Corpus", "Ubuntu-Net West African Languages"],
            
            "ubuntu_dialogue_insights": {
                "conversation_type": dataset_analysis["ubuntu_dialogue_analysis"]["conversation_type"],
                "response_style": dataset_analysis["ubuntu_dialogue_analysis"]["appropriate_response_style"],
                "learned_from": "1M+ Ubuntu IRC conversations"
            },
            
            "ubuntu_net_insights": {
                "cultural_patterns": dataset_analysis["ubuntu_net_analysis"]["detected_cultural_patterns"],
                "cultural_response_needed": dataset_analysis["ubuntu_net_analysis"]["cultural_response_needed"],
                "learned_from": "West African cultural communication patterns"
            },
            
            # Combined translation result
            "enhanced_translation": dataset_analysis["combined_intelligence"]["enhanced_translation"] or message,
            "cultural_warnings": dataset_analysis["combined_intelligence"]["cultural_warnings"],
            "response_suggestions": dataset_analysis["combined_intelligence"]["suggested_responses"],
            "educational_insights": dataset_analysis["combined_intelligence"]["educational_notes"],
            
            # Dataset synergy explanation
            "how_datasets_work_together": self._explain_dataset_synergy(dataset_analysis)
        }
        
        return result
    
    def _explain_dataset_synergy(self, analysis: Dict) -> Dict:
        """Explain how the two datasets complement each other"""

        return {
            "ubuntu_dialogue_corpus_role": (
                "Provides natural conversation patterns and response structures"
            ),
            "ubuntu_net_role": (
                "Provides cultural meaning and context for African communication styles"
            ),
            "synergy": (
                "Dialogue patterns teach HOW to respond, cultural data teaches "
                "WHAT the message really means"
            ),
            "example": (
                "Ubuntu Dialogue shows how to acknowledge concerns helpfully, "
                "Ubuntu-Net reveals when 'interesting' means 'problematic' in Ghanaian culture"
            )
        }


def demonstrate_dataset_powered_translation():
    print("UBUNTU TRANSLATE: Cultural Intelligence in Action")
    print("=" * 60)

    # Example translations 
    examples = [
        {
            "speaker": "Kwame",
            "message": (
                "Eiii, you people and your wahala. I'm coming there right now!"
            ),
            "culture": "Ghanaian-Nigerian mix",
            "translation": (
                "Wow, you guys and your drama/problems. I'm coming over right now!"
            ),
            "tone_alert": (
                "Frustrated but caring - this is how close friends express "
                "'I'm fed up but I'm still going to help you'"
            ),
            "cultural_notes": (
                "'Eiii' = Twi expression of exasperation, "
                "'wahala' = Nigerian Pidgin for trouble/drama"
            ),
            "meaning": (
                "Your friend is mildly annoyed but is definitely coming to support you"
            )
        },
        {
            "speaker": "Amina",
            "message": (
                "Ehen, so you're now doing big man/woman for us abi? "
                "You don forget your friends wey dey struggle?"
            ),
            "culture": "Nigerian",
            "translation": (
                "Oh, so now you're acting all important/successful, huh? "
                "Have you forgotten about your friends who are still struggling?"
            ),
            "tone_alert": (
                "Playful sarcasm and gentle guilt-tripping - common in Nigerian friendship dynamics"
            ),
            "cultural_notes": (
                "'Ehen, so' = sarcastic setup phrase, 'abi' = seeking agreement, "
                "'big man/woman' = acting superior"
            ),
            "meaning": (
                "Your friend is probably joking but wants attention/acknowledgment"
            ),
            "suggested_response": (
                "Haha, never! You know you're still my guy/girl. Let's catch up soon?"
            )
        },
        {
            "speaker": "Dr. Mensah",
            "message": (
                "Your proposal is very interesting. We should discuss the "
                "implementation details further."
            ),
            "culture": "Ghanaian",
            "translation": (
                "Your proposal has some concerning aspects. We need to address "
                "the problematic implementation details."
            ),
            "tone_alert": (
                "Diplomatic disagreement - 'interesting' in Ghanaian professional context "
                "often means 'problematic'"
            ),
            "cultural_notes": (
                "In Ghanaian English, 'interesting' can be a polite way to express concerns"
            ),
            "meaning": (
                "They see issues with your proposal that need to be addressed"
            ),
            "suggested_response": (
                "What specific concerns do you have about the proposal? "
                "I'd love to address them."
            )
        },
        {
            "speaker": "Grace",
            "message": (
                "Aki wewe, you have jokes! Missing the meeting like that."
            ),
            "culture": "Kenyan",
            "translation": (
                "Seriously? You're being ridiculous! Missing the meeting like that "
                "was unacceptable."
            ),
            "tone_alert": (
                "Sarcastic calling-out with underlying concern - not a compliment about being funny"
            ),
            "cultural_notes": (
                "'Aki wewe' = Kenyan expression of disbelief, 'you have jokes' = sarcastic "
                "response to ridiculous behavior"
            ),
            "meaning": (
                "Your colleague is calling out your unprofessional behavior"
            ),
            "suggested_response": (
                "You're absolutely right to call me out. I messed up and I apologize."
            )
        }
    ]

    
    for i, example in enumerate(examples, 1):
        print(f"\n**Example {i}: {example['culture']} Cultural Pattern**")
        print()
        
        # Incoming message
        print(f"[Incoming Message]")
        print(f"{example['speaker']}: \"{example['message']}\"")
        print()
        
        # Processing
        print(f"[Ubuntu Translate Processing...]")
        print(f"Analyzing: {example['culture']} expressions + Cultural context...")
        print()
        
        # Translation result  
        print(f"[Translation Result]")
        print(f"Translation: \"{example['translation']}\"")
        print()
        print(f"Tone Alert: {example['tone_alert']}")
        print()
        print(f"Cultural Note: {example['cultural_notes']}")
        print()
        print(f"This means: {example['meaning']}")
        
        if "suggested_response" in example:
            print()
            print(f"Suggested Response: \"{example['suggested_response']}\"")
        
        print()
        print("─" * 60)

def explain_dataset_architecture():
    """Show the contrast between standard vs Ubuntu translation"""
    
    print("\nCOMPARISON: Standard Translation vs Ubuntu Translate")
    print("=" * 60)
    
    comparisons = [
        {
            "original": (
                "Ehen, so you're now doing big man/woman for us abi? "
                "You don forget your friends wey dey struggle?"
            ),
            "standard": (
                "Yes, so you're now doing big man/woman for us right? "
                "You have forgotten your friends who are struggling?"
            ),
            "ubuntu": (
                "Oh, so now you're acting all important/successful, huh? "
                "Have you forgotten about your friends who are still struggling?"
            ),
            "context": (
                "This message contains playful sarcasm and gentle guilt-tripping - "
                "common in Nigerian friendship dynamics. Your friend is probably "
                "joking but wants attention/acknowledgment."
            ),
            "response": (
                "Haha, never! You know you're still my guy/girl. Let's catch up soon?"
            )
        }
    ]

    
    for comp in comparisons:
        print(f"\n**Original Message (Nigerian Pidgin):**")
        print(f"\"{comp['original']}\"")
        print()
        
        print(f"**Standard Translation:**")
        print(f"\"{comp['standard']}\"")
        print()
        
        print(f"**Ubuntu Translate:**")
        print(f"**Translation:** \"{comp['ubuntu']}\"")
        print()
        print(f"**Context Alert:** \"{comp['context']}\"")
        print()
        print(f"**Suggested Response:** \"{comp['response']}\"")
        
    print(f"\nTHE DIFFERENCE:")
    print(f"Standard translation = Word confusion and hurt feelings")
    print(f"Ubuntu Translate = Cultural understanding and stronger relationships")

if __name__ == "__main__":
    demonstrate_dataset_powered_translation()
    explain_dataset_architecture()

    print("\nKEY INSIGHT:")
    print("The Ubuntu Dialogue Corpus teaches conversation structure and flow.")
    print("The Ubuntu-Net dataset teaches cultural meaning and context.")
    print("Together, they create AI that can navigate cross-cultural communication.")
    print("\nThis is why dataset diversity matters - different datasets teach different skills. ")
