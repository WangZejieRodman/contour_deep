import numpy as np

data = np.load('test_bev_output.npz')
print("Keys:", data.files)
print("BEV shape:", data['bev_layers'].shape)
print("VCD shape:", data['vcd'].shape)
print("BEV range:", data['bev_layers'].min(), data['bev_layers'].max())
print("VCD range:", data['vcd'].min(), data['vcd'].max())

# 可视化某一层
import matplotlib.pyplot as plt

plt.imshow(data['bev_layers'][0])  # 显示第1层
plt.title("BEV Layer 0")
plt.colorbar()
plt.savefig('bev_layer0.png')

plt.imshow(data['bev_layers'][1])  # 显示第2层
plt.title("BEV Layer 1")
plt.savefig('bev_layer1.png')

plt.imshow(data['bev_layers'][2])  # 显示第3层
plt.title("BEV Layer 2")
plt.savefig('bev_layer2.png')

plt.imshow(data['bev_layers'][3])  # 显示第4层
plt.title("BEV Layer 3")
plt.savefig('bev_layer3.png')

plt.imshow(data['bev_layers'][4])  # 显示第5层
plt.title("BEV Layer 4")
plt.savefig('bev_layer4.png')

plt.imshow(data['bev_layers'][5])  # 显示第6层
plt.title("BEV Layer 5")
plt.savefig('bev_layer5.png')

plt.imshow(data['bev_layers'][6])  # 显示第7层
plt.title("BEV Layer 6")
plt.savefig('bev_layer6.png')

plt.imshow(data['bev_layers'][7])  # 显示第8层
plt.title("BEV Layer 7")
plt.savefig('bev_layer7.png')

plt.imshow(data['vcd'])  # 显示垂直复杂度
plt.colorbar()
plt.title("BEVs vcd")
plt.savefig('bevs_vcd.png')

