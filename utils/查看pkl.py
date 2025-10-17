import pickle

with open('test_bci_output.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys:", data.keys())
print("\nStats:", data['stats'])
print("\nBCIs总数:", sum(len(layer) for layer in data['bcis']))
print("Contours总数:", sum(len(layer) for layer in data['contours']))

# 查看第一个BCI
bci = data['bcis'][2][0]  # Layer 2的第一个BCI
print(f"\n示例BCI: Level={bci.level}, 邻居数={len(bci.nei_pts)}")