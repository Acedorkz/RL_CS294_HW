import pickle
with open('expert_data/Humanoid-v2.pkl', 'rb') as f:
    data = pickle.loads(f.read())

print(data['observations'].shape)
print(data['actions'].shape)
# (2000, 376)
# (2000, 1, 17)