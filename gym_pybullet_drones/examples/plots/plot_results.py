import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/eya/Desktop/gym-pybullet-drones/gym_pybullet_drones/examples/results_LFruns.csv")

# Success rate
success_rate = df["found"].mean()

# Time-to-find (only successful runs)
ttf = df[df["found"] == 1]["ttf_sec"]

print(f"Runs: {len(df)}")
print(f"Success rate: {success_rate*100:.1f}%")
if len(ttf) > 0:
    print(f"TTF mean: {ttf.mean():.2f}s")
    print(f"TTF std: {ttf.std(ddof=1):.2f}s")

# Plot
plt.figure()
plt.hist(ttf, bins=10)
plt.xlabel("Time to Find (s)")
plt.ylabel("Count")
plt.title("Leader Follower SAR – Time to Find Victim")
plt.grid(True)
plt.show()
