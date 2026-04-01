# Neural-VISM: 基于神经网络的隐式溶剂模型架构



## 1. 输入数据定义
系统输入涵盖了分子的几何、化学及物理属性。

### 1.1 基础输入参数
* **原子坐标 (coords):** $[B, N, 3]$ 
* **原子类型 (atom_types):** $[B,N]$（用于 one-hot 编码）
* **物理参数:** 包含范德华半径 ($radii$) Lennard-Jones 参数 ($\epsilon, \sigma$) 以及部分电荷 ($charges$) 。
* **查询点 (query_points):** $[B, Q, 3]$ 

### 1.2 输入组织分类
建议将输入分为以下三类以适配网络：
1. **Query-level 输入:** 查询点 $x$ 的绝对坐标、相对于局部邻居原子的相对坐标 $x-r_i$ 及其距离 $||x-r_i||$ 。
2. **Atom-level 输入:** 原子类型 embedding、半径 $R_i$、LJ 参数 $\epsilon_i, \sigma_i$，以及可选的疏水性或极化率特征 。

---

## 2. 核心架构：局部-全局条件化隐式 SDF 网络
架构 A 旨在通过分子环境特征来预测空间中任意点的 SDF 值 。

### 2.1 形式化表达
$$\phi_{\theta}(x)=f_{\theta}(x,z_{local}(x,\mathcal{M}),z_{global}(\mathcal{M}),z_{solv})$$ 

### 2.2 关键组件
* **分子编码器 (Molecular Encoder):** 负责将变长原子集 $\mathcal{M}$ 转换为特征向量]。
    * 可采用 **Set/DeepSets** 或 **SE(3)-equivariant GNN/EGNN** 。
    * 推荐方案：结合局部邻域编码（提取最近 $K$ 个原子的特征）与全局池化（获取整体分子 embedding）。
* **查询点解码器 (Query Decoder):** * 输出标量 SDF 。
    * 使用 **Fourier features** 或位置编码来处理坐标。
    * 推荐使用 **SIREN** 或高频编码，以更好地表示分子的尖锐曲率区域。

---

## 3. 物理驱动的损失函数
[cite_start]最推荐的方案是直接最小化自由能泛函并结合几何正则化 [cite: 46]。

### 3.1 自由能泛函 (Neural Version)
利用光滑 Heaviside 函数 $H_{\epsilon}$ 构造可微分能量项：
[cite_start]$$G_{\theta}=P\int(1-H_{\epsilon}(\phi_{\theta}))dx+\gamma_{0}\int\delta_{\epsilon}(\phi_{\theta})|\nabla\phi_{\theta}|dx-2\gamma_{0}\tau\int H(\phi_{\theta})\delta_{\epsilon}(\phi_{\theta})|\nabla\phi_{\theta}|dx+\rho_{0}\int U(x)H_{\epsilon}(\phi_{\theta})dx$$ [cite: 49]
* 该能量包含：**体积项、面积项、平均曲率修正、LJ 体积分** 。
* 其中平均曲率 $H$ 由 level-set 形式计算：$H=\frac{1}{2}\nabla\cdot(\frac{\nabla\phi}{|\nabla\phi|})$ 。

### 3.2 总损失函数
$$\mathcal{L}=\lambda_{E}G_{\theta}+\lambda_{eik}\mathcal{L}_{eik}+\lambda_{contain}\mathcal{L}_{contain}+\lambda_{sign}\mathcal{L}_{sign}+\lambda_{reg}\mathcal{L}_{reg}$$ [cite: 57]
* **Eikonal Loss:** $\mathcal{L}_{eik}=\mathbb{E}_{x}(|||\nabla\phi_{\theta}(x)||-1|)$，确保输出符合符号距离函数性质 [cite: 59]。

---

## 4. 训练策略与采样
### 4.1 阶段式训练
1. **第一阶段：几何预训练。** 学习分子包络 SDF（参考 SASA/SES 或原子并集）。
2. **第二阶段：物理微调。** 加入自由能损失，使表面从几何包络调整到物理平衡界面。

### 4.2 采样策略
建议在四个区域进行采样：
1. **近界面点:** 用于 SDF 监督、Eikonal 约束和曲率计算。
2. **原子球内部点:** 用于 containment 约束 。
3. **远场点:** 用于符号约束和体积分估计 。
4. **LJ 势剧变区:** 对靠近原子但未进入硬核的区域进行过采样，以应对高梯度引起的数值不稳定。