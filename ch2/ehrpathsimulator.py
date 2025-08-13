import matplotlib.pyplot as plt

# Initial patient data (3 days)
initial_record = [
    {"day": 1, "fever": 101.5, "cough": True, "notes": "Mild symptoms"},
    {"day": 2, "fever": 102.0, "cough": True, "notes": "Fever rising"},
    {"day": 3, "fever": 101.8, "cough": True, "notes": "Lab test ordered"},
]

# AI Agent 
class EHRCarePathSimulator:
    def __init__(self, patient_record):
        self.record = patient_record
        self.simulated_days = 0

    # Recorded patient data
    def get_text_window(self, days=3):
        return self.record[-days:]
    
    # Capture historical pattern
    def summarize_temporal_context(self):
        total_days = len(self.record)
        fever_days = sum(1 for r in self.record if r.get("fever", 0) > 100)
        return {
            "total_days": total_days,
            "fever_days": fever_days
        }
    
    # Predict patient progress
    def predict_next_step(self, text_window, summary): 
        last_day = text_window[-1]
        next_day = {}
        
        if last_day.get("fever", 0) > 101 and last_day.get("cough", False):
            next_day = {
                "fever": last_day["fever"] + 0.5,
                "cough": True,
                "notes": "Condition may be worsening"
            }
        elif last_day.get("fever", 0) <= 100:
            next_day = {
                "fever": 98.6,
                "cough": False,
                "notes": "Patient improving"
            }
        else:
            next_day = {
                "fever": last_day["fever"] - 0.5,
                "cough": last_day.get("cough", False),
                "notes": "Monitor condition"
            }
        
        return next_day
        
    # Simulation path of health journey 
    def simulate_until_recovery(self, max_days=5):
      
        current_day = len(self.record)
        
        while self.simulated_days < max_days:
            text_window = self.get_text_window()
            summary = self.summarize_temporal_context()
            prediction = self.predict_next_step(text_window, summary)
            
            current_day += 1
            prediction["day"] = current_day
            self.record.append(prediction)
            self.simulated_days += 1
            
            # Check for recovery
            if prediction["fever"] <= 98.6 and not prediction["cough"]:
                break
        
        return self.record

# Run the simulation
agent = EHRCarePathSimulator(initial_record)
final_record = agent.simulate_until_recovery()

# Visualization data
days = [entry["day"] for entry in final_record]
fevers = [entry["fever"] for entry in final_record]

# Plot fever trends
plt.figure(figsize=(10, 5))
plt.plot(days, fevers, marker='o', linestyle='-', linewidth=2)
plt.title("Simulated Patient Fever Over Time")
plt.xlabel("Day")
plt.ylabel("Fever (°F)")
plt.grid(True)
plt.xticks(days)
plt.ylim(97, max(fevers) + 1)
plt.show()

# Print the predicted journey
print("Complete Patient Journey:")
for entry in final_record:
    print(f"Day {entry['day']}: {entry['fever']}°F, Cough: {entry.get('cough', False)}, {entry['notes']}")


